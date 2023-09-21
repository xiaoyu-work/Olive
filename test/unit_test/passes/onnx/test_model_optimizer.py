# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from test.unit_test.utils import get_onnx_model

from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx import OnnxModelOptimizer
import pytest


def test_onnx_model_optimizer_pass(tmp_path):
    # setup
    input_model = get_onnx_model()
    p = create_pass_from_dict(OnnxModelOptimizer, {}, disable_search=True)
    output_folder = str(tmp_path / "onnx")

    # execute
    p.run(input_model, None, output_folder)


@pytest.mark.parametrize("use_sess", [True, False])
def test_onnx_model_temp_path(use_sess):
    from pathlib import Path
    import tempfile
    import onnx
    import onnxruntime
    from optimum.exporters.onnx import main_export
    curr_path = Path(__file__).parent
    main_export(
        "distilbert-base-uncased-finetuned-sst-2-english",
        output=str(curr_path)
    )
    input_model = onnx.load(str(curr_path / "model.onnx"))
    with tempfile.TemporaryDirectory(prefix="pre-quant") as temp_dir:
        temp_path = Path(temp_dir)
        output_model_path_1 = temp_path / "input_model1.onnx"
        onnx.save_model(input_model, str(output_model_path_1), save_as_external_data=True)
        if use_sess:
            sess_option = onnxruntime.SessionOptions()
            sess_option.optimized_model_filepath = str(temp_path / "optimized.onnx")
            sess_option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
            sess = onnxruntime.InferenceSession(output_model_path_1, sess_option, providers=["CPUExecutionProvider"])
            sess._reset_session(providers=["CPUExecutionProvider"], provider_options=None)
        else:
            ...
    print("test_onnx_model_temp_path done")
