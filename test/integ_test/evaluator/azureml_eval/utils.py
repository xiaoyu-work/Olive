# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import shutil
from pathlib import Path
from test.integ_test.utils import download_azure_blob, get_olive_workspace_config

from torchvision import datasets
from torchvision.transforms import ToTensor

from olive.azureml.azureml_client import AzureMLClientConfig
from olive.common.config_utils import validate_config
from olive.data.config import DataComponentConfig, DataConfig
from olive.evaluator.metric import AccuracySubType, LatencySubType, Metric, MetricType
from olive.systems.azureml import AzureMLDockerConfig, AzureMLSystem

# pylint: disable=redefined-outer-name

_current_dir, _models_dir, _data_dir, _user_script = None, None, None, None


def get_directories():
    global _current_dir, _models_dir, _data_dir, _user_script

    _current_dir = Path(__file__).resolve().parent

    _models_dir = _current_dir / "models"
    _models_dir.mkdir(parents=True, exist_ok=True)

    _data_dir = _current_dir / "data"
    _data_dir.mkdir(parents=True, exist_ok=True)

    _user_script = _current_dir / "user_script.py"

    return _current_dir, _models_dir, _data_dir, _user_script


def _get_metric_data_config(name, post_process=None):
    data_config = DataConfig(
        name=name,
        type="HuggingfaceContainer",
        user_script=str(_user_script),
        load_dataset_config=DataComponentConfig(
            type="mnist_dataset",
            params={"data_dir": str(_data_dir)},
        ),
        pre_process_data_config=DataComponentConfig(type="skip_pre_process"),
    )
    if post_process:
        data_config.post_process_data_config = DataComponentConfig(type=post_process)
    return validate_config(data_config, DataConfig)


def get_accuracy_metric():
    sub_types = [{"name": AccuracySubType.ACCURACY_SCORE, "metric_config": {"task": "multiclass", "num_classes": 10}}]
    return Metric(
        name="accuracy",
        type=MetricType.ACCURACY,
        sub_types=sub_types,
        data_config=_get_metric_data_config("accuracy_metric_data_config", "mnist_post_process"),
    )


def get_latency_metric():
    return Metric(
        name="latency",
        type=MetricType.LATENCY,
        sub_types=[{"name": LatencySubType.AVG}],
        data_config=_get_metric_data_config("latency_metric_data_config"),
    )


def download_models():
    pytorch_model_config = {
        "container": "olivetest",
        "blob": "models/model.pt",
        "download_path": _models_dir / "model.pt",
    }
    download_azure_blob(**pytorch_model_config)

    onnx_model_config = {
        "container": "olivetest",
        "blob": "models/model.onnx",
        "download_path": _models_dir / "model.onnx",
    }
    download_azure_blob(**onnx_model_config)


def download_data():
    datasets.MNIST(_data_dir, download=True, transform=ToTensor())


def get_pytorch_model():
    return str(_models_dir / "model.pt")


def get_onnx_model():
    return str(_models_dir / "model.onnx")


def delete_directories():
    global _current_dir, _models_dir, _data_dir, _user_script
    shutil.rmtree(_data_dir)
    shutil.rmtree(_models_dir)
    _current_dir = _models_dir = _data_dir = _user_script = None


def get_aml_target():
    aml_compute = "cpu-cluster"
    current_path = Path(__file__).absolute().parent
    conda_file_location = current_path / "conda.yaml"
    azureml_client_config = AzureMLClientConfig(**get_olive_workspace_config())
    docker_config = AzureMLDockerConfig(
        base_image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
        conda_file_path=conda_file_location,
    )
    return AzureMLSystem(
        azureml_client_config=azureml_client_config,
        aml_compute=aml_compute,
        aml_docker_config=docker_config,
        is_dev=True,
    )
