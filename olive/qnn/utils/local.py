# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import os
import platform
import time
from pathlib import Path
from typing import Tuple

from olive.common.utils import run_subprocess

logger = logging.getLogger(__name__)


def get_qnn_root() -> str:
    """Get the QNN root directory from the QNN_ROOT environment variable."""
    try:
        qnn_root = os.environ["QNN_SDK_ROOT"]
        logger.debug(f"QNN_SDK_ROOT is set to {qnn_root}")
    except KeyError:
        raise ValueError("QNN_SDK_ROOT is not set") from None

    return qnn_root


def get_qnn_target_arch(fail_on_unsupported: bool = True) -> str:
    """Get the QNN target architecture from the system and processor.

    fail_on_unsupported: Whether to raise an exception if the system or processor is not supported
    """
    system = platform.system()

    qnn_target_arch = None
    if system == "Linux":
        machine = platform.machine()
        if machine == "x86_64":
            qnn_target_arch = "x64-Linux"
        elif machine == "aarch64":
            qnn_target_arch = "ARM64-Linux"
        else:
            if fail_on_unsupported:
                raise ValueError(f"Unsupported machine {machine} on system {system}")
    elif system == "Windows":
        processor_identifier = os.environ.get("PROCESSOR_IDENTIFIER", "")
        qnn_target_arch = "ARM64-Windows" if "ARM" in processor_identifier else "x64-Windows"
    else:
        if fail_on_unsupported:
            raise ValueError(f"Unsupported system {system}")
    logger.debug(f"QNN target architecture: {qnn_target_arch}")

    return qnn_target_arch


def get_qnn_win_arch_name(qnn_root: str, qnn_target_arch: str) -> str:
    """Get the QNN ARM64-Windows architecture name from the QNN root directory.

    qnn_root: The unzipped QNN SDK directory
    qnn_target_arch: The QNN target architecture
    """
    if not Path(qnn_root).exists():
        raise ValueError(f"QNN root directory {qnn_root} does not exist")
    # for qnn: there are: aarch64-windows-*, arm64x-windows-*
    # TODO(trajep): what is the difference between aarch64 and arm64x?
    prefix_map = {"x64-Windows": "x86_64-windows-", "ARM64-Windows": "aarch64-windows-"}
    prefix = prefix_map[qnn_target_arch]

    arm_windows_archs = list(Path(qnn_root).glob(f"lib/{prefix}*"))
    if len(arm_windows_archs) == 0:
        raise FileNotFoundError(f"SNPE_ROOT {qnn_root} missing {prefix}*")

    arm_windows_arch = arm_windows_archs[0].name
    logger.debug(f"SNPE {qnn_target_arch} arch name: {arm_windows_arch}")

    return arm_windows_arch


def get_qnn_env(dev: bool = False) -> dict:
    """Get the QNN environment variables.

    dev: Whether to use the QNN development environment. Only supported on x64-Linux
    """
    qnn_root = get_qnn_root()
    qnn_target_arch = get_qnn_target_arch()
    if "Linux" in qnn_target_arch:
        target_arch_name = "x86_64-linux-clang"
    else:
        target_arch_name = get_qnn_win_arch_name(qnn_root, qnn_target_arch)

    if dev and qnn_target_arch != "x64-Linux":
        raise ValueError("QNN development environment is only supported on x64-Linux(Ubuntu or WSL)")

    bin_path = str(Path(f"{qnn_root}/bin/{target_arch_name}"))
    lib_path = str(Path(f"{qnn_root}/lib/{target_arch_name}"))
    python_path = str(Path(f"{qnn_root}/lib/python"))

    env = {}
    delimiter = os.path.pathsep
    if platform.system() == "Linux":
        env["LD_LIBRARY_PATH"] = lib_path
        bin_path += delimiter + "/usr/bin"
    elif platform.system() == "Windows":
        if qnn_target_arch == "ARM64-Windows":
            bin_path = str(Path(f"{qnn_root}/olive-arm-win"))
            if not Path(bin_path).exists():
                raise FileNotFoundError(
                    f"Path {bin_path} does not exist. Please run 'python -m olive.snpe.configure' to add the"
                    " missing folder"
                )
        else:
            bin_path += delimiter + lib_path
    env["PYTHONPATH"] = python_path
    env["PATH"] = bin_path
    env["QNN_SDK_ROOT"] = qnn_root

    for paths in env.values():
        for path in paths.split(delimiter):
            if not Path(path).exists():
                raise FileNotFoundError(f"Path {str(Path(path))} does not exist")

    return env


def run_qnn_command(
    cmd: str, dev: bool = False, runs: int = 1, sleep: int = 0, log_error: bool = True
) -> Tuple[str, str]:
    """Run a SNPE command.

    cmd: The command to run
    dev: Whether to use the SNPE development environment. Only supported on x64-Linux
    runs: The number of times to run the command
    sleep: The number of seconds to sleep between runs
    log_error: Whether to log an error if the command fails
    """
    env = get_qnn_env(dev)
    full_cmd = cmd

    for run in range(runs):
        run_log_msg = "" if runs == 1 else f" (run {run + 1}/{runs})"
        logger.debug(f"Running SNPE command{run_log_msg}: {full_cmd}")
        returncode, stdout, stderr = run_subprocess(full_cmd, env)
        logger.debug(f"Return code: {returncode} \n Stdout: {stdout} \n Stderr: {stderr}")
        if returncode != 0:
            break
        if sleep > 0 and run < runs - 1:
            time.sleep(sleep)

    if returncode != 0:
        error_msg = (
            f"Error running qnn command. \n Command: {full_cmd} \n Return code: {returncode} \n Stdout: {stdout} \n"
            f" Stderr: {stderr}"
        )
        if log_error:
            logger.error(error_msg)
        raise RuntimeError(error_msg)

    return stdout, stderr
