# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from enum import Enum
from typing import List

from olive.common.config_utils import ConfigBase

logger = logging.getLogger(__name__)


class PackagingType(str, Enum):
    """Output Artifacts type."""

    Zipfile = "Zipfile"
    AzureMLDeployment = "AzureMLDeployment"
    AzureMLDatastore = "AzureMLDatastore"


class InferencingServerType(str, Enum):
    """Inferencing server type"""

    AzureMLOnline = "AzureMLOnline"
    AzureMLBatch = "AzureMLBatch"


class AzureMLInputType(str, Enum):
    UriFile = "UriFile"
    UriFolder = "UriFolder"


class AzureMLModeType(str, Enum):
    Download = "download"
    ReadonlyMount = "readonly_mount"


class ModelPackageInput(ConfigBase):
    type: AzureMLInputType
    path: str
    Mode: AzureMLModeType = AzureMLModeType.Download
    mount_path: str = None


class InferenceServerConfig(ConfigBase):
    type: InferencingServerType
    code_folder: str
    scoring_script: str


class BaseEnvironmentType(str, Enum):
    EnvironmentAsset = "EnvironmentAsset"


class ModelConfigurationConfig(ConfigBase):
    mode: str = "Download"  # readonly_mount or Download (default)
    mount_path: str = None  # Relative mount path


class ModelPackageConfig(ConfigBase):
    target_environment_name: str
    target_environment_version: str = "1"
    inferencing_server: InferenceServerConfig
    base_environent_id: str
    inputs: List[ModelPackageInput] = None
    model_configurations: ModelConfigurationConfig = None
    environment_variables: dict = None
    tags: dict = None


class DeploymentConfig(ConfigBase):
    endpoint_name: str
    deployment_name: str
    instance_type: str = "Standard_DS3_v2"
    compute: str = None
    instance_count: int = 1
    mini_batch_size: int = 10  # AzureMLBatch only
    extra_config: dict = None


class CommonPackagingConfig(ConfigBase):
    export_in_mlflow_format: bool = False


class AzureMLPackagingConfig(CommonPackagingConfig):
    model_package: ModelPackageConfig
    model_name: str = None
    model_version: str = None
    deployment_config: DeploymentConfig = None


class PackagingConfig(ConfigBase):
    """Olive output artifacts generation config."""

    type: PackagingType = PackagingType.Zipfile
    name: str = "OutputModels"
    config: CommonPackagingConfig = None
