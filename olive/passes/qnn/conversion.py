# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from typing import Any, Dict

from olive.hardware import AcceleratorSpec
from olive.model import QNNModelHandler
from olive.passes.olive_pass import Pass
from olive.passes.pass_config import PassConfigParam


class SNPEConversion(Pass):
    @staticmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        ...

    def _run_for_config(
        self,
        model,
        data_root: str,
        config: Dict[str, Any],
        output_model_path: str,
    ) -> QNNModelHandler:
        ...
