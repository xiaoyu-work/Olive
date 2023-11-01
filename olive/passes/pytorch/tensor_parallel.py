# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# Automatically distribute PyTorch model using Tensor Parallelism
# --------------------------------------------------------------------------

import logging
from typing import Any, Dict

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import DistributedPyTorchModel, PyTorchModel
from olive.passes import Pass
from olive.passes.olive_pass import PassConfigParam

logger = logging.getLogger(__name__)


class PyTorchTensorParallel(Pass):
    @staticmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        # Note : The default world_size should be the no of gpus (AceleratorSpec.Device == GPU)
        # in the target OliveSystem
        return {"world_size": PassConfigParam(type_=int, default_value=0, description="world size")}

    def _run_for_config(
        self, model: PyTorchModel, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> DistributedPyTorchModel:
        self.world_size = config["world_size"]

        # 1. Load the model
        pytorch_model = model.load_model()
        # 2. Replace the layers
        self.replace_layers()
        # 3. Split the weights
        self.split_weights(pytorch_model)
        # 4. Save the weights for each rank
        for r in self.world_size:
            self.load_rank_weights(pytorch_model, r)
            output_filepath = str(output_model_path / f"{r:02d}")
            pytorch_model.save_pretrained(output_filepath)
        # 5. Restore layers that were replaced
        self.restore_layers()
        # 6. Construct DistributedPyTorchModel from saved wegihts for each rank

        # return PyTorchModel with updated model path
        model_config = model.to_json()["config"]
        model_config["model_paths"] = output_model_path
        model_config["world_size"] = self.world_size
        return DistributedPyTorchModel(**model_config)

    def replace_layers(self):
        raise NotImplementedError

    def restore_layers(self):
        raise NotImplementedError

    def split_weights(self, m):
        raise NotImplementedError

    def load_rank_weights(self, r, ws):
        raise NotImplementedError
