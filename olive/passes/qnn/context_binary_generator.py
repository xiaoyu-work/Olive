# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Union

from olive.constants import ModelFileFormat
from olive.hardware import AcceleratorSpec
from olive.model import ONNXModelHandler, PyTorchModelHandler, QNNModelHandler, TensorFlowModelHandler
from olive.passes.olive_pass import Pass
from olive.passes.pass_config import PassConfigParam
from olive.qnn.utils.local import run_qnn_command

logger = logging.getLogger(__name__)


class QNNContextBinaryGenerator(Pass):
    @staticmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {
            "backend": PassConfigParam(
                type_=str,
                required=True,
                description=("Path to a QNN backend .so library to create the context binary."),
            ),
            "binary_file": PassConfigParam(
                type_=str,
                required=False,
                description=(
                    "Name of the binary file to save the context binary to."
                    " Saved in the same path as --output_dir option with .bin"
                    " as the binary file extension. If not provided, no backend binary"
                    " is created."
                ),
            ),
            "extra_args": PassConfigParam(
                type_=str, default_value=None, description="Extra arguments to qnn-context-binary-generator"
            ),
        }

    @staticmethod
    def _validators() -> Dict[str, Callable[..., Any]]:
        pass

    def _run_for_config(
        self,
        model: Union[TensorFlowModelHandler, PyTorchModelHandler, ONNXModelHandler],
        data_root: str,
        config: Dict[str, Any],
        output_model_path: str,
    ) -> QNNModelHandler:
        main_cmd = "qnn-context-binary-generator"

        # input model path's name without suffix
        input_model_path = Path(model.model_path).resolve()
        # TODO(trajep): find .so file in the same directory as the model
        output_model_path = Path(output_model_path).resolve()

        cmd_list = [
            main_cmd,
            f"--model {input_model_path}",
            f"--backend {config['backend']}",
            f"--output_dir {output_model_path}",
            f"--binary_file {config['binary_file']}" if config["binary_file"] else "",
            config["extra_args"],
        ]
        run_qnn_command(
            " ".join(cmd_list),
            dev=True,
        )
        return QNNModelHandler(output_model_path, model_file_format=ModelFileFormat.QNN_LIB)
