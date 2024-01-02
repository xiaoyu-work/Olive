# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import Any, Dict, List

from olive.hardware.accelerator import Device
from olive.model.config.registry import model_handler_registry
from olive.model.handler.base import OliveModelHandler


@model_handler_registry("QNNModel")
class QNNModelHandler(OliveModelHandler):
    def __init__(
        self,
    ):
        ...

    def load_model(self, rank: int = None):
        ...

    def prepare_session(
        self,
        inference_settings: Dict[str, Any] | None = None,
        device: Device = Device.CPU,
        execution_providers: str | List[str] = None,
        rank: int | None = None,
    ):
        ...

    def to_json(self, check_object: bool = False):
        ...
