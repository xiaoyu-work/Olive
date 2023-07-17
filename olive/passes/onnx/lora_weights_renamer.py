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

    def __init__(self, model: OnnxModel):
        self.model = model

    def apply(self):
        matmul_nodes = self.model.get_nodes_by_op_type("MatMul")

        for node in matmul_nodes:
            if "lora" in node.name:
                result = re.search(
                    (
                        r"/([a-z]+_blocks\.\d+|mid_block)"
                        r"/(attentions\.\d+)"
                        r"/(transformer_blocks\.\d+)"
                        r"/(attn\d+)"
                        r"/(to_[a-z]+_lora)"
                        r"/(up|down)"
                        "/MatMul",
                    ),
                    node.name,
                )

                if result is not None:
                    initializer = self.model.get_initializer(node.input[1])
                    new_name = (
                        f"{result.group(1)}"
                        f".{result.group(2)}"
                        f".{result.group(3)}"
                        f".{result.group(4)}"
                        ".processor"
                        f".{result.group(5)}"
                        f".{result.group(6)}"
                        ".weight"
                    )
                    initializer.name = new_name
                    new_input = onnx.helper.make_tensor_value_info(new_name, initializer.data_type, initializer.dims)
                    self.model.add_input(new_input)
                    node.input[1] = new_name

        self.model.update_graph()
