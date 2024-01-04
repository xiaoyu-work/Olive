# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import platform
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from olive.common.config_utils import validate_config
from olive.constants import Framework, ModelFileFormat
from olive.hardware.accelerator import Device
from olive.model.config import IoConfig
from olive.model.config.registry import model_handler_registry
from olive.model.handler.base import OliveModelHandler
from olive.qnn import QNNInferenceSession, QNNSessionOptions

logger = logging.getLogger(__name__)


@model_handler_registry("QNNModel")
class QNNModelHandler(OliveModelHandler):
    json_config_keys: Tuple[str, ...] = ("io_config", "model_file_format")

    def __init__(
        self,
        model_path: str,
        model_attributes: Optional[Dict[str, Any]] = None,
        io_config: Union[Dict[str, Any], IoConfig, str, Callable] = None,
        model_file_format: ModelFileFormat = ModelFileFormat.QNN_CPP,
    ):
        super().__init__(
            framework=Framework.QNN,
            model_file_format=model_file_format,
            model_path=model_path,
            model_attributes=model_attributes,
        )
        # io config for conversion to onnx
        self.io_config = (
            validate_config(io_config, IoConfig).dict() if isinstance(io_config, (IoConfig, dict)) else io_config
        )

    def _get_model_lib_path(self):
        if self.model_file_format == ModelFileFormat.QNN_CPP:
            logger.debug("QNNModelHandler: model_path is the cpp file for QNN_CPP model format.")
        elif self.model_file_format == ModelFileFormat.QNN_LIB:
            # self.model_path is the folder containing the lib file, the structure is like:
            # - self.model_path
            #   - aarch64-android
            #     - libmodel.so
            #   - x86_64-linux-clang
            #     - libmodel.so
            model_attributes = self.model_attributes or {}
            model_lib_suffix = None
            lib_targets = model_attributes.get("lib_targets")
            if lib_targets is None:
                logger.debug(
                    "QNNModelHandler: lib_targets is not provided, using default lib_targets x86_64-linux-clang"
                )
                if platform.system() == "Linux":
                    lib_targets = "x86_64-linux-clang"
                    model_lib_suffix = ".so"
                elif platform.system() == "Windows":
                    lib_targets = "x86_64-windows-msvc"
                    model_lib_suffix = ".dll"
            model_folder = Path(self.model_path) / lib_targets
            model_paths = list(model_folder.glob(f"*{model_lib_suffix}"))
            if len(model_paths) == 0:
                raise FileNotFoundError(f"No model file found in {model_folder}")
            elif len(model_paths) > 1:
                raise FileNotFoundError(f"Multiple model files found in {model_folder}: {model_paths}")
            return str(model_paths[0])
        else:
            raise ValueError(f"Unsupported model file format {self.model_file_format}")
        return None

    def load_model(self, rank: int = None):
        raise NotImplementedError("QNNModelHandler does not support load_model")

    def prepare_session(
        self,
        inference_settings: Union[Dict[str, Any], None] = None,
        device: Device = Device.CPU,
        execution_providers: Union[str, List[str]] = None,
        rank: Union[int, None] = None,
    ):
        session_options = QNNSessionOptions(**inference_settings or {})
        return QNNInferenceSession(self._get_model_lib_path(), self.io_config, session_options)
