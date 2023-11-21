# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import hashlib
import inspect
import io
import json
import logging
import pickle
import platform
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)


def run_subprocess(cmd, env=None, cwd=None, check=False):  # pragma: no cover
    logger.debug(f"Running command: {cmd} with env: {env}")

    windows = platform.system() == "Windows"
    cmd = shlex.split(cmd, posix=not windows)
    if windows:
        path = env.get("PATH") if env else None
        cmd_exe = shutil.which(cmd[0], path=path)
        cmd[0] = cmd_exe
    out = subprocess.run(cmd, env=env, cwd=cwd, capture_output=True, check=False)
    returncode = out.returncode
    stdout = out.stdout.decode("utf-8")
    stderr = out.stderr.decode("utf-8")
    if check and returncode != 0:
        raise RuntimeError(f"Command '{cmd}' failed with return code {returncode} and error: {stderr}")

    return returncode, stdout, stderr


def get_package_name_from_ep(execution_provider):
    provider_package_mapping = {
        "CPUExecutionProvider": ("onnxruntime", "ort-nightly"),
        "CUDAExecutionProvider": ("onnxruntime-gpu", "ort-nightly-gpu"),
        "TensorrtExecutionProvider": ("onnxruntime-gpu", "ort-nightly-gpu"),
        "ROCMExecutionProvider": ("onnxruntime-gpu", "ort-nightly-gpu"),
        "OpenVINOExecutionProvider": ("onnxruntime-openvino", None),
        "DmlExecutionProvider": ("onnxruntime-directml", "ort-nightly-directml"),
    }
    return provider_package_mapping.get(execution_provider, ("onnxruntime", "ort-nightly"))


def hash_string(string):  # pragma: no cover
    md5_hash = hashlib.md5()
    md5_hash.update(string.encode())
    return md5_hash.hexdigest()


def hash_io_stream(f):  # pragma: no cover
    md5_hash = hashlib.md5()
    # Read and update hash in chunks of 4K
    for byte_block in iter(lambda: f.read(4096), b""):
        md5_hash.update(byte_block)
    return md5_hash.hexdigest()


def hash_file(filename):  # pragma: no cover
    with open(filename, "rb") as f:  # noqa: PTH123
        return hash_io_stream(f)


def hash_dict(dictionary):  # pragma: no cover
    md5_hash = hashlib.md5()
    encoded_dictionary = json.dumps(dictionary, sort_keys=True).encode()
    md5_hash.update(encoded_dictionary)
    return md5_hash.hexdigest()


def hash_function(function):  # pragma: no cover
    md5_hash = hashlib.md5()
    try:
        source = inspect.getsource(function)
    except OSError:
        logger.warning(f"Could not get source code for {function.__name__}. Hash will be based on name only.")
        source = function.__name__
    md5_hash.update(source.encode())
    return md5_hash.hexdigest()


def hash_object(obj):  # pragma: no cover
    f = io.BytesIO()
    pickle.dump(obj, f)
    return hash_io_stream(f)


def unflatten_dict(dictionary):  # pragma: no cover
    """Unflatten a dictionary with keys of the form "a.b.c" into a nested dictionary."""
    result = {}
    for key, value in dictionary.items():
        parts = list(key)
        d = result
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return result


def flatten_dict(dictionary, stop_condition=None):  # pragma: no cover
    """Flatten a nested dictionary into a dictionary with keys of the form (a,b,c)."""
    result = {}
    for key, value in dictionary.items():
        if stop_condition is not None and stop_condition(value):
            result[(key,)] = value
        elif isinstance(value, dict):
            result.update({(key, *k): v for k, v in flatten_dict(value, stop_condition).items()})
        else:
            result[(key,)] = value
    return result


def retry_func(func, args=None, kwargs=None, max_tries=3, delay=5, backoff=2, exceptions=None):
    """Retry a function call using an exponential backoff.

    Args:
        func: Function to call.
        args: Arguments to pass to the function.
        kwargs: Keyword arguments to pass to the function.
        max_tries: Maximum number of retries.
        delay: Initial delay between retries in seconds.
        backoff: Backoff multiplier e.g. value of 2 will double the delay each retry.
        exceptions: Exceptions to catch. If None, catch all exceptions. Can be a single exception or a tuple
            of exceptions.
    """
    args = args or []
    kwargs = kwargs or {}
    exceptions = exceptions or Exception
    num_tries, sleep_time = 0, delay
    while num_tries < max_tries:
        try:
            logger.debug(f"Calling function '{func.__name__}'. Try {num_tries + 1} of {max_tries}...")
            out = func(*args, **kwargs)
            logger.debug("Succeeded.")
            return out
        except exceptions as e:
            num_tries += 1
            if num_tries == max_tries:
                logger.error(f"Failed with error: {e}", exc_info=True)
                raise e
            logger.debug(f"Failed. Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)
            sleep_time *= backoff
    return None


def tensor_data_to_device(data, device: str):
    if device is None:
        return data

    from torch import Tensor

    if isinstance(data, Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: tensor_data_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [tensor_data_to_device(v, device) for v in data]
    elif isinstance(data, tuple):
        return tuple(tensor_data_to_device(v, device) for v in data)
    elif isinstance(data, set):
        return {tensor_data_to_device(v, device) for v in data}
    else:
        return data


def resolve_torch_dtype(dtype):
    """Get torch dtype from string or torch dtype.

    :param dtype: dtype to resolve. Can be a string (float16, torch.float16, etc) or torch dtype.
    :return: torch dtype.
    """
    import torch

    if isinstance(dtype, str):
        dtype = dtype.replace("torch.", "")
        try:
            dtype = getattr(torch, dtype)
        except AttributeError as e:
            raise AttributeError(f"Invalid dtype '{dtype}'.") from e
    assert isinstance(dtype, torch.dtype), f"dtype must be a string or torch.dtype, got {type(dtype)}."
    return dtype


def get_attr(module, attr, fail_on_not_found=False):
    """Get attribute from module.

    :param module: module to get attribute from.
    :param attr: attribute name, can be a string with dot notation. If empty, return module.
    :param fail_on_not_found: if True, raise AttributeError if attribute is not found.
    :return: attribute
    """
    if not attr:
        # return module if attr is empty
        return module

    attr = attr.split(".")
    for a in attr:
        try:
            module = getattr(module, a)
        except AttributeError as e:
            not_found_message = f"Attribute {attr} not found."
            if fail_on_not_found:
                raise AttributeError(not_found_message) from e
            else:
                logger.warning(not_found_message)
                return None
    return module


def find_submodules(module, submodule_types, full_name=False):
    """Find all submodules of a given type in a module.

    :param module: module to search.
    :param submodule_type: type of submodule to search for. Can be a single type or a tuple of types.
    :param full_name: if True, return full name of submodule. Otherwise, return last part of submodule name.
    :return: list of submodules names.
    """
    submodules = set()
    for name, submodule in module.named_modules():
        if isinstance(submodule, submodule_types):
            if full_name:
                submodules.add(name)
            else:
                submodules.add(name.split(".")[-1])
    return list(submodules)


def model_proto_to_file(
    model,
    output_path: Union[str, Path],
    save_as_external_data: Optional[bool] = False,
    all_tensors_to_one_file: Optional[bool] = True,
    external_data_name: Optional[Union[str, Path]] = None,
    size_threshold: Optional[int] = 1024,
    convert_attribute: Optional[bool] = False,
) -> bool:
    """Save the ONNX model to the specified path.

    :param model: The ONNX model to save.
    :param output_path: The path to save the ONNX model to.
    :param save_as_external_data: If True, save tensor data to separate files instead of directly in the ONNX file.
        Large models (>2GB) may be forced to save external data regardless of the value of this parameter.
    :param all_tensors_to_one_file: Effective only if save_as_external_data is True. If True, save all tensors to one
        external file specified by 'external_data_name'. If False, save each tensor to a file named with the tensor
        name.
    :param external_data_name: Effective only if all_tensors_to_one_file is True and save_as_external_data is True.
        If not specified, the external data file will be named with <model_path_name>.data

    :return: True if the model has external data, False otherwise.
    """
    import onnx

    output_path = Path(output_path)
    if output_path.exists():
        logger.info(f"Deleting existing onnx file: {output_path}")
        output_path.unlink()

    # parent directory of .onnx file
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if not save_as_external_data:
        try:
            # save model
            onnx.save_model(model, str(output_path))
            return False
        except ValueError as e:
            # there are different types of error message for large model (>2GB) based on onnx version
            # just try to save as external data
            # if it fails, raise the original error
            try:
                logger.debug(f"Model save failed with error: {e}. Trying to save as external data.")
                model_proto_to_file(model, output_path, True, all_tensors_to_one_file, external_data_name)
                logger.warning(
                    "Model is too large to save as a single file but 'save_as_external_data' is False. Saved tensors"
                    " as external data regardless."
                )
                return True
            except Exception:
                raise e from None

    # location for external data
    external_data_path = output_dir / (external_data_name if external_data_name else f"{output_path.name}.data")
    location = external_data_path.name if all_tensors_to_one_file else None

    if all_tensors_to_one_file:
        if external_data_path.exists():
            # Delete the external data file. Otherwise, data will be appended to existing file.
            logger.info(f"Deleting existing external data file: {external_data_path}")
            external_data_path.unlink()
    else:
        if any(output_dir.iterdir()):
            raise RuntimeError(f"Output directory ({output_dir}) for external data is not empty.")

    # save model
    onnx.save_model(
        model,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=all_tensors_to_one_file,
        location=location,
        size_threshold=size_threshold,
        convert_attribute=convert_attribute,
    )
    return True
