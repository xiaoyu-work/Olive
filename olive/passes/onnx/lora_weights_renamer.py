# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import re
from logging import getLogger

import onnx
from onnxruntime.transformers.fusion_base import Fusion
from onnxruntime.transformers.onnx_model import OnnxModel

logger = getLogger(__name__)


class LoraWeightsRenamer(Fusion):
    """
    Rename the LoRA weights to make them easily interchangeable.
    """

    def __init__(self, model: OnnxModel, lora_weights_as_inputs: bool, prefix: str):
        self.model = model
        self.lora_weights_as_inputs = lora_weights_as_inputs
        self.prefix = prefix

    def apply(self):
        if self.prefix == "unet":
            self._rename_unet_weights()
        else:
            assert self.prefix == "text_encoder"
            self._rename_text_encoder_weights()

        self.model.update_graph()

    def _rename_scalars(self, suffix):
        network_alpha_already_added = False
        lora_scale_already_added = False

        mul_nodes = self.model.get_nodes_by_op_type("Mul")
        for mul_node in mul_nodes:
            if mul_node.name.endswith(suffix):
                initializer = self.model.get_initializer(mul_node.input[1])
                mul_node.input[1] = "lora_network_alpha_per_rank"

                if not network_alpha_already_added:
                    # The first Mul node is the network_alpha divided by the rank
                    new_input = onnx.helper.make_tensor_value_info(
                        "lora_network_alpha_per_rank", initializer.data_type, initializer.dims
                    )
                    self.model.add_input(new_input)
                    network_alpha_already_added = True

                # The next node is the LoRA scale
                next_node = self.model.get_children(mul_node)[0]

                if next_node.op_type == "Mul":
                    initializer = self.model.get_initializer(next_node.input[1])
                    next_node.input[1] = "lora_scale"

                    if not lora_scale_already_added:
                        new_input = onnx.helper.make_tensor_value_info(
                            "lora_scale", initializer.data_type, initializer.dims
                        )
                        self.model.add_input(new_input)
                        lora_scale_already_added = True

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
                    initializer = self.model.get_initializer(matmul_node.input[1])
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
                    matmul_node.input[1] = new_name

                    if self.lora_weights_as_inputs:
                        new_input = onnx.helper.make_tensor_value_info(
                            new_name, initializer.data_type, initializer.dims
                        )
                        self.model.add_input(new_input)
                    else:
                        initializer.name = new_name

        if self.lora_weights_as_inputs:
            self._rename_scalars("_lora/Mul")

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
                    initializer = self.model.get_initializer(matmul_node.input[1])
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
                    matmul_node.input[1] = new_name

                    if self.lora_weights_as_inputs:
                        new_input = onnx.helper.make_tensor_value_info(
                            new_name, initializer.data_type, initializer.dims
                        )
                        self.model.add_input(new_input)
                    else:
                        initializer.name = new_name

        if self.lora_weights_as_inputs:
            self._rename_scalars("lora_linear_layer/Mul")
