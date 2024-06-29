# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import shutil
from pathlib import Path
from test.integ_test.utils import download_azure_blob
from zipfile import ZipFile

from olive.common.config_utils import validate_config
from olive.data.config import DataComponentConfig, DataConfig
from olive.evaluator.metric import AccuracySubType, LatencySubType, Metric, MetricType

# pylint: disable=redefined-outer-name

_current_dir, _models_dir, _data_dir, _user_script = None, None, None, None


def get_directories():
    global _current_dir, _models_dir, _data_dir

    _current_dir = Path(__file__).resolve().parent

    _models_dir = _current_dir / "models"
    _models_dir.mkdir(parents=True, exist_ok=True)

    _data_dir = _current_dir / "data"
    _data_dir.mkdir(parents=True, exist_ok=True)

    _user_script = _current_dir / "user_script.py"

    return _current_dir, _models_dir, _data_dir, _user_script


def _get_metric_data_config(name, dataset, post_process=None):
    data_config = DataConfig(
        name=name,
        type="HuggingfaceContainer",
        user_script=str(_user_script),
        load_dataset_config=DataComponentConfig(
            type=dataset,
            params={"data_dir": _data_dir},
        ),
        pre_process_data_config=DataComponentConfig(type="skip_pre_process"),
    )
    if post_process:
        data_config.post_process_data_config = DataComponentConfig(type=post_process)
    return validate_config(data_config, DataConfig)


def get_accuracy_metric(post_process, dataset="mnist_dataset"):
    sub_types = [{"name": AccuracySubType.ACCURACY_SCORE, "metric_config": {"task": "multiclass", "num_classes": 10}}]
    return Metric(
        name="accuracy",
        type=MetricType.ACCURACY,
        sub_types=sub_types,
        data_config=_get_metric_data_config("accuracy_metric_data_config", dataset, post_process),
    )


def get_latency_metric(dataset="mnist_dataset"):
    sub_types = [{"name": LatencySubType.AVG}]
    return Metric(
        name="latency",
        type=MetricType.LATENCY,
        sub_types=sub_types,
        data_config=_get_metric_data_config("latency_metric_data_config", dataset),
    )


def get_hf_accuracy_metric(post_process="prajjwal1_post_process", dataset="prajjwal1_dataset"):
    return get_accuracy_metric(post_process, dataset)


def get_hf_latency_metric(dataset="prajjwal1_dataset"):
    return get_latency_metric(dataset)


def get_pytorch_model():
    download_path = _models_dir / "model.pt"
    pytorch_model_config = {
        "container": "olivetest",
        "blob": "models/model.pt",
        "download_path": download_path,
    }
    download_azure_blob(**pytorch_model_config)
    return {"model_path": str(download_path)}


def get_huggingface_model():
    return {"hf_config": {"model_class": "AutoModelForSequenceClassification", "model_name": "prajjwal1/bert-tiny"}}


def get_onnx_model():
    download_path = _models_dir / "model.onnx"
    onnx_model_config = {
        "container": "olivetest",
        "blob": "models/model.onnx",
        "download_path": download_path,
    }
    download_azure_blob(**onnx_model_config)
    return {"model_path": str(download_path)}


def get_openvino_model():
    download_path = _models_dir / "openvino.zip"
    openvino_model_config = {
        "container": "olivetest",
        "blob": "models/openvino.zip",
        "download_path": download_path,
    }
    download_azure_blob(**openvino_model_config)
    with ZipFile(download_path) as zip_ref:
        zip_ref.extractall(_models_dir)
    return {"model_path": str(_models_dir / "openvino")}


def delete_directories():
    global _current_dir, _models_dir, _data_dir, _user_script
    shutil.rmtree(_data_dir)
    shutil.rmtree(_models_dir)
    _current_dir = _models_dir = _data_dir = _user_script = None
