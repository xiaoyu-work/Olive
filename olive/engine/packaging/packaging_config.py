# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from enum import Enum
from typing import Union

from olive.common.config_utils import ConfigBase
from olive.model import OutputModelFormat


class PackagingType(str, Enum):
    """Output Artifacts type."""

    Zipfile = "Zipfile"


class PackagingConfig(ConfigBase):
    """Olive output artifacts generation config."""

    type: PackagingType = PackagingType.Zipfile  # noqa: A003
    name: str = "OutputModels"
    output_model_format: Union[str, OutputModelFormat] = OutputModelFormat.RAW_ONNX
