# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import shutil
import zipfile
from pathlib import Path
from test.unit_test.utils import (
    get_accuracy_metric,
    get_onnxconversion_pass,
    get_pytorch_model,
    get_pytorch_model_config,
)
from unittest.mock import patch

import onnx
import pytest

from olive.engine import Engine
from olive.engine.footprint import Footprint
from olive.engine.packaging.packaging_config import (
    AzureMLPackagingConfig,
    InferenceServerConfig,
    ModelPackageConfig,
    PackagingConfig,
    PackagingType,
)
from olive.engine.packaging.packaging_generator import generate_output_artifacts
from olive.evaluator.metric import AccuracySubType
from olive.evaluator.olive_evaluator import OliveEvaluator
from olive.workflows.run.config import RunConfig, RunEngineConfig

# def test_generate_zipfile_output():
#     # setup
#     p = get_onnxconversion_pass()
#     metric = get_accuracy_metric(AccuracySubType.ACCURACY_SCORE)
#     evaluator = OliveEvaluator(metrics=[metric])
#     options = {
#         "cache_dir": "./cache",
#         "clean_cache": True,
#         "search_strategy": {
#             "execution_order": "joint",
#             "search_algorithm": "random",
#         },
#         "clean_evaluation_cache": True,
#     }
#     engine = Engine(options, evaluator=evaluator)
#     engine.register(p)

#     input_model = get_pytorch_model()

#     packaging_config = PackagingConfig()
#     packaging_config.type = PackagingType.Zipfile
#     packaging_config.name = "OutputModels"

#     output_dir = Path(__file__).parent / "outputs"

#     # execute
#     engine.run(input_model=input_model, packaging_config=packaging_config, output_dir=output_dir)

#     # assert
#     artifacts_path = output_dir / "OutputModels.zip"
#     assert artifacts_path.exists()
#     with zipfile.ZipFile(artifacts_path, "r") as zip_ref:
#         zip_ref.extractall(output_dir)
#     assert (output_dir / "SampleCode").exists()
#     assert (output_dir / "CandidateModels").exists()

#     # cleanup
#     shutil.rmtree(output_dir)


# def test_generate_zipfile_artifacts_no_search():
#     # setup
#     p = get_onnxconversion_pass()
#     options = {
#         "cache_dir": "./cache",
#         "clean_cache": True,
#         "clean_evaluation_cache": True,
#     }
#     engine = Engine(options)
#     engine.register(p)

#     input_model = get_pytorch_model()

#     packaging_config = PackagingConfig()
#     packaging_config.type = PackagingType.Zipfile
#     packaging_config.name = "OutputModels"

#     output_dir = Path(__file__).parent / "outputs"

#     # execute
#     engine.run(input_model=input_model, packaging_config=packaging_config, output_dir=output_dir)

#     # assert
#     artifacts_path = output_dir / "OutputModels.zip"
#     assert artifacts_path.exists()
#     with zipfile.ZipFile(artifacts_path, "r") as zip_ref:
#         zip_ref.extractall(output_dir)
#     assert (output_dir / "SampleCode").exists()
#     assert (output_dir / "CandidateModels").exists()

#     # cleanup
#     shutil.rmtree(output_dir)


@patch("olive.azureml.azure_ml_client.AzureMLClient")
def test_generate_azureml_output(mock_azureml_client):
    # setup
    # onnx will save with external data when tensor size is greater than 1024(default threshold)
    mock_sys_getsizeof.return_value = mocked_size_value
    metric = get_accuracy_metric(AccuracySubType.ACCURACY_SCORE)
    evaluator_config = OliveEvaluatorConfig(metrics=[metric])
    options = {
        "cache_dir": "./cache",
        "clean_cache": False,
        "search_strategy": {
            "execution_order": "joint",
            "search_algorithm": "random",
        },
        "clean_evaluation_cache": True,
    }
    engine = Engine(options, evaluator=evaluator)
    engine.register(p)
    packaging_config = PackagingConfig()
    packaging_config.type = PackagingType.AzureML
    inference_server_config = InferenceServerConfig(type="AzureMLOnline", code_folder="code", scoring_script="score.py")
    model_package_config = ModelPackageConfig(
        target_environment_name="olive_target_environment",
        target_environment_version="1",
        inferencing_server=inference_server_config,
        base_environent_id="azureml:aml-packaging-test:1",
    )
    azureml_config = AzureMLPackagingConfig(
        model_name="olive_model",
        model_version="1",
        model_package=model_package_config,
    )
    packaging_config.azureml_config = azureml_config

    mock_azureml_client.ml_client.models.create_or_update.return_value = "olive_model"
    mock_azureml_client.ml_client.models.begin_package.return_value = "olive_model_package"
    mock_azureml_client.ml_client.environments.get.return_value = "olive_target_environment"
    mock_azureml_client.ml_client.online_endpoints.get.return_value = "package-test-endpoint"
    mock_azureml_client.ml_client.online_endpoints.begin_create_or_update.return_value = "package-test-deployment"
    mock_azureml_client.ml_client.online_deployments.begin_create_or_update.return_value = "package-test-deployment"

    input_model_config = get_pytorch_model_config()

    output_dir = Path(__file__).parent / "outputs"

    # execute
    engine.run(
        input_model_config=input_model_config,
        accelerator_specs=[DEFAULT_CPU_ACCELERATOR],
        data_root=None,
        packaging_config=packaging_config,
        output_dir=output_dir,
    )

    # assert
    assert mock_azureml_client.ml_client.models.begin_package.call_once()
