# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from test.unit_test.utils import get_pytorch_model

from olive.data.template import dummy_data_config_template
from olive.hardware.accelerator import AcceleratorSpec
from olive.passes.olive_pass import FullPassConfig, create_pass_from_dict
from olive.passes.pytorch.quantization_aware_training import QuantizationAwareTraining


def test_quantization_aware_training_pass_default(tmp_path):
    # setup
    input_model = get_pytorch_model()
    config = {
        "train_data_config": dummy_data_config_template([[1]]),
        "checkpoint_path": str(tmp_path / "checkpoint"),
    }

    p = create_pass_from_dict(QuantizationAwareTraining, config, disable_search=True)
    output_folder = str(tmp_path / "onnx")

    # execute
    p.run(input_model, None, output_folder)


def test_optional_ep(tmp_path):
    accl = AcceleratorSpec("cpu", None)
    p = create_pass_from_dict(
        QuantizationAwareTraining,
        {"train_data_config": dummy_data_config_template([[1]])},
        accelerator_spec=accl,
    )
    qat_json = p.to_json()
    pass_config = FullPassConfig.from_json(qat_json)
    sp = pass_config.create_pass()
    assert sp.accelerator_spec.execution_provider is None
