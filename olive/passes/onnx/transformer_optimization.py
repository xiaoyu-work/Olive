# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import os
from copy import deepcopy
from typing import Any, Dict, List, Union

import numpy as np
import onnx.helper
import onnx.numpy_helper

from olive.hardware.accelerator import AcceleratorSpec, Device
from olive.model import ONNXModel
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.passes.pass_config import PassConfigParam
from olive.strategy.search_parameter import Boolean, Categorical

logger = logging.getLogger(__name__)


class OrtTransformersOptimization(Pass):
    """Optimize transformer based models in scenarios where ONNX Runtime does not apply the optimization at load time.
    It is based on onnxruntime.transformers.optimizer."""

    @staticmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        from onnxruntime.transformers.fusion_options import FusionOptions

        is_gpu = accelerator_spec.accelerator_type == Device.GPU

        config = {
            "model_type": PassConfigParam(
                type_=str,
                required=True,
                description=(
                    "Transformer based model type, including bert (exported by PyTorch), gpt2 (exported by PyTorch), "
                    "bert_tf (BERT exported by tf2onnx), bert_keras (BERT exported by keras2onnx), and "
                    "unet/vae/clip (stable diffusion)."
                ),
            ),
            "num_heads": PassConfigParam(type_=int, default_value=0, description="Number of attention heads."),
            "hidden_size": PassConfigParam(type_=int, default_value=0, description="Number of hidden nodes."),
            # TODO: Figure out what the expected type is
            "optimization_options": PassConfigParam(
                type_=Union[Dict[str, Any], FusionOptions],
                default_value=None,
                description="Optimization options that turn on/off some fusions.",
            ),
            "opt_level": PassConfigParam(
                type_=Any,
                default_value=None,
                searchable_values=Categorical([0, 1, 2, 99]),
                description=(
                    "Graph optimization level of Onnx Runtime: "
                    "0 - disable all (default), 1 - basic, 2 - extended, 99 - all."
                ),
            ),
            "use_gpu": PassConfigParam(type_=bool, default_value=is_gpu, description="Flag for GPU inference."),
            "only_onnxruntime": PassConfigParam(
                type_=bool,
                default_value=False,
                searchable_values=Boolean(),
                description="Whether only use onnxruntime to optimize model, and no python fusion.",
            ),
            "float16": PassConfigParam(
                type_=bool, default_value=False, description="Whether half-precision float will be used."
            ),
            "input_int32": PassConfigParam(
                type_=bool, default_value=False, description="Whether int32 tensors will be used as input."
            ),
            "keep_io_types": PassConfigParam(
                type_=bool,
                default_value=True,
                description="Keep input and output tensors in their original data type",
            ),
            "force_fp32_ops": PassConfigParam(
                type_=List[str], default_value=None, description="Operators that are forced to run in float32"
            ),
        }
        config.update(get_external_data_config())
        return config

    def validate_search_point(
        self, search_point: Dict[str, Any], accelerator_spec: AcceleratorSpec, with_fixed_value: bool = False
    ) -> bool:
        if with_fixed_value:
            search_point = self.config_at_search_point(search_point or {})
        if search_point.get("float16"):
            if accelerator_spec.execution_provider == "TensorrtExecutionProvider":
                logger.info(
                    "TensorRT has its own float16 implementation, please avoid to use float16 in transformers "
                    "optimization. Suggest to set 'trt_fp16_enable' as True in OrtPerfTuning."
                )
                return False
            if accelerator_spec.execution_provider == "CPUExecutionProvider":
                logger.info("CPUExecutionProvider does not support float16 very well, please avoid to use float16.")
                return False
        if search_point.get("use_gpu") and accelerator_spec.execution_provider == "CPUExecutionProvider":
            logger.info("CPUExecutionProvider does not support GPU inference, please avoid to use use_gpu.")
            return False
        return True

    @staticmethod
    def _set_fusion_options(run_config: Dict[str, Any]):
        from onnxruntime.transformers.fusion_options import FusionOptions

        fusion_options = FusionOptions(run_config["model_type"])
        fusion_options.__dict__.update(run_config["optimization_options"])
        run_config["optimization_options"] = fusion_options

    def _run_for_config(
        self, model: ONNXModel, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> ONNXModel:
        from onnxruntime.transformers import optimizer as transformers_optimizer

        # start with a copy of the config
        run_config = deepcopy(config)
        del (
            run_config["float16"],
            run_config["input_int32"],
            run_config["keep_io_types"],
            run_config["force_fp32_ops"],
        )
        for key in get_external_data_config():
            del run_config[key]

        output_model_path = ONNXModel.resolve_path(os.path.join(output_model_path, os.path.basename(model.model_path)))

        optimization_options = config["optimization_options"]
        if optimization_options:
            self._set_fusion_options(run_config)

        optimizer = transformers_optimizer.optimize_model(input=model.model_path, **run_config)

        # TODO: Remove once this is merged into ORT's fusion_attention_unet.py script
        if optimization_options and optimization_options.get("use_lora_multi_head_attention", False):
            from onnxruntime.transformers.onnx_model_bert import BertOnnxModel

            from olive.passes.onnx.fusion_attention_unet_lora import FusionAttentionUnetLora

            # The model type doesn't matter here, so get the base model
            base_model = BertOnnxModel(optimizer.model, run_config["num_heads"], run_config["hidden_size"])

            # Self Attention
            self_attention_fusion = FusionAttentionUnetLora(
                base_model, base_model.hidden_size, base_model.num_heads, False, True, False
            )
            self_attention_fusion.apply()

            # Cross Attention
            cross_attention_fusion = FusionAttentionUnetLora(
                base_model, base_model.hidden_size, base_model.num_heads, True, False, True
            )
            cross_attention_fusion.apply()

        if optimization_options:
            from onnxruntime.transformers.onnx_model_bert import BertOnnxModel

            from olive.passes.onnx.lora_weights_renamer import LoraWeightsRenamer

            lora_weights_strategy = optimization_options.get("lora_weights_strategy", "baked")
            lora_weights_prefix = optimization_options.get("lora_weights_prefix", "")

            if lora_weights_strategy != "baked":
                # The model type doesn't matter here, so get the base model
                base_model = BertOnnxModel(optimizer.model, run_config["num_heads"], run_config["hidden_size"])
                lora_weights_renamer = LoraWeightsRenamer(
                    base_model, lora_weights_strategy == "input_binding", lora_weights_prefix
                )
                lora_weights_renamer.apply()

                if lora_weights_strategy == "input_binding":
                    for graph_input in optimizer.model.graph.input:
                        if graph_input.name.endswith("lora.up.weight"):
                            graph_input.type.tensor_type.shape.dim[0].ClearField("dim_value")
                            graph_input.type.tensor_type.shape.dim[0].dim_param = "lora_rank"

                            tensor_shape = (1, graph_input.type.tensor_type.shape.dim[1].dim_value)
                            tensor_dtype = onnx.helper.tensor_dtype_to_np_dtype(graph_input.type.tensor_type.elem_type)
                            tensor = np.zeros(tensor_shape).astype(tensor_dtype)
                            initializer = onnx.numpy_helper.from_array(tensor, graph_input.name)
                            optimizer.model.graph.initializer.add().CopyFrom(initializer)
                        elif graph_input.name.endswith("lora.down.weight"):
                            graph_input.type.tensor_type.shape.dim[1].ClearField("dim_value")
                            graph_input.type.tensor_type.shape.dim[1].dim_param = "lora_rank"

                            tensor_shape = (graph_input.type.tensor_type.shape.dim[0].dim_value, 1)
                            tensor_dtype = onnx.helper.tensor_dtype_to_np_dtype(graph_input.type.tensor_type.elem_type)
                            tensor = np.zeros(tensor_shape).astype(tensor_dtype)
                            initializer = onnx.numpy_helper.from_array(tensor, graph_input.name)
                            optimizer.model.graph.initializer.add().CopyFrom(initializer)
                        elif graph_input.name == "lora_network_alpha_per_rank" or graph_input.name == "lora_scale":
                            tensor_dtype = onnx.helper.tensor_dtype_to_np_dtype(graph_input.type.tensor_type.elem_type)
                            tensor = np.ones([]).astype(tensor_dtype)
                            initializer = onnx.numpy_helper.from_array(tensor, graph_input.name)
                            optimizer.model.graph.initializer.add().CopyFrom(initializer)

        if config["float16"]:
            op_block_list = config["force_fp32_ops"]
            optimizer.convert_float_to_float16(keep_io_types=config["keep_io_types"], op_block_list=op_block_list)

        if config["input_int32"]:
            optimizer.change_graph_inputs_to_int32()

        # Topologically sort the graph at the end since previous optimizations may have broken it
        optimizer.topological_sort()

        # save the model to the output path and return the model
        return model_proto_to_olive_model(optimizer.model, output_model_path, config)
