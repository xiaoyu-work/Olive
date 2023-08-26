# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import re

import numpy as np
import onnx
from onnxruntime.transformers.onnx_model import OnnxModel


class LoraWeightsRenamer:
    """
    Rename the LoRA weights to make them easily interchangeable.
    """

    def __init__(self, model: OnnxModel, prefix: str):
        self.model = model
        self.prefix = prefix
        self.shape_infer = self.model.infer_runtime_shape(update=True)

    def apply(self):
        if self.prefix == "unet":
            self._rename_unet_weights()
        else:
            assert self.prefix == "text_encoder"
            self._rename_text_encoder_weights()

        self.model.update_graph()

    def _add_mul_initializer(self, mul_node, initializer_name):
        lora_scale_already_added = False
        initializer = self.model.get_constant_value(mul_node.input[1])

        if initializer is not None:
            mul_node.input[1] = initializer_name

            if not lora_scale_already_added:
                initializer_data = np.ones((1,), dtype=initializer.dtype)
                new_initializer = onnx.helper.make_tensor(
                    initializer_name,
                    onnx.helper.np_dtype_to_tensor_dtype(initializer.dtype),
                    initializer.shape,
                    initializer_data,
                )
                self.model.add_initializer(new_initializer)
                lora_scale_already_added = True

    def _add_matmul_initializer(self, matmul_node, new_name):
        initializer = self.model.get_initializer(matmul_node.input[1])

        # When the initializer is missing, we add an empty optimizer that can be overriden later on
        if initializer is None:
            input_shape = self.shape_infer.get_edge_shape(matmul_node.input[1])
            tensor_dtype = self.model.get_dtype(matmul_node.input[1])

            if tensor_dtype is None:
                tensor_dtype = self.shape_infer.known_vi_[matmul_node.input[1]].type.tensor_type.elem_type

            np_dtype = onnx.helper.tensor_dtype_to_np_dtype(tensor_dtype)
            initializer_data = np.zeros(input_shape, dtype=np_dtype)
            new_initializer = onnx.helper.make_tensor(new_name, tensor_dtype, input_shape, initializer_data)
            self.model.add_initializer(new_initializer)
        else:
            initializer.name = new_name

        matmul_node.input[1] = new_name

    def _rename_initializers(self, matmul_node, weight_name):
        self._add_matmul_initializer(matmul_node, weight_name)

        network_alpha_node = self.model.get_children(matmul_node)[0]
        if network_alpha_node.op_type == "Mul":
            self._add_mul_initializer(network_alpha_node, "lora_network_alpha_per_rank")
            lora_scale_node = self.model.get_children(network_alpha_node)[0]
            assert lora_scale_node.op_type == "Mul"
            self._add_mul_initializer(lora_scale_node, "lora_scale")

    def _rename_unet_weights(self):
        matmul_nodes = self.model.get_nodes_by_op_type("MatMul")
        for matmul_node in matmul_nodes:
            if "lora" in matmul_node.name:
                result = re.search(
                    r"/([a-z]+_blocks\.\d+|mid_block)"
                    r"/(attentions\.\d+)"
                    r"/(transformer_blocks\.\d+)"
                    r"/(attn\d+)"
                    r"/(to_[a-z]+_lora)"
                    r"/(up|down)"
                    "/MatMul",
                    matmul_node.name,
                )

                if result is not None:
                    new_name = (
                        f"{self.prefix}"
                        f".{result.group(1)}"
                        f".{result.group(2)}"
                        f".{result.group(3)}"
                        f".{result.group(4)}"
                        ".processor"
                        f".{result.group(5)}"
                        f".{result.group(6)}"
                        ".weight"
                    )

                    self._rename_initializers(matmul_node, new_name)

    def _rename_text_encoder_weights(self):
        matmul_nodes = self.model.get_nodes_by_op_type("MatMul")
        for matmul_node in matmul_nodes:
            if "lora" in matmul_node.name:
                result = re.search(
                    r"/(text_model)"
                    r"/(encoder)"
                    r"/(layers.\d+)"
                    r"/(self_attn)"
                    r"/([a-z]+)_proj"
                    r"/lora_linear_layer"
                    r"/(up|down)"
                    "/MatMul",
                    matmul_node.name,
                )

                if result is not None:
                    new_name = (
                        f"{self.prefix}"
                        f".{result.group(1)}"
                        f".{result.group(2)}"
                        f".{result.group(3)}"
                        f".{result.group(4)}"
                        f".to_{result.group(5)}_lora"
                        f".{result.group(6)}"
                        ".weight"
                    )

                    self._rename_initializers(matmul_node, new_name)
