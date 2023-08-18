# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import json
import os
import shutil
from pathlib import Path

import onnxruntime as ort
from packaging import version

from olive.model import CompositeOnnxModel, ONNXModel
from olive.workflows import run as olive_run


def optimize(onnx_file: str, optimized_model_dir: Path):
    from google.protobuf import __version__ as protobuf_version

    # protobuf 4.x aborts with OOM when optimizing LLaMA
    if version.parse(protobuf_version) > version.parse("3.20.3"):
        print("This script requires protobuf 3.20.3. Please ensure your package version matches requirements.txt.")
        exit(1)

    ort.set_default_logger_severity(4)
    script_dir = Path(__file__).resolve().parent

    model_info = dict()

    # Optimize the model with Olive
    print(f"\nOptimizing {onnx_file}")

    olive_config = None
    with open(script_dir / "config_llama_v2.json", "r") as fin:
        olive_config = json.load(fin)

    olive_config["input_model"]["config"]["model_path"] = onnx_file
    olive_run(olive_config)

    # TODO: rename the 0 prefix in the path when the hardware accelerator feature is implemented.
    footprints_file_path = Path(__file__).resolve().parent / "footprints/llama_v2_gpu-dml_footprints.json"
    with footprints_file_path.open("r") as footprint_file:
        footprints = json.load(footprint_file)
        optimization_footprint = None
        for _, footprint in footprints.items():
            if footprint["from_pass"] == "OrtTransformersOptimization":
                optimization_footprint = footprint

        assert optimization_footprint

        optimized_olive_model = ONNXModel(**optimization_footprint["model_config"]["config"])

        model_info = {
            "optimized": {
                "path": Path(optimized_olive_model.model_path),
            },
        }

        print(f"Optimized Model   : {model_info['optimized']['path']}")

    merged_model_path = str(model_info["optimized"]["path"])
    merged_weights_path = merged_model_path + ".data"

    merged_model_name = os.path.basename(merged_model_path)
    merged_weights_name = merged_model_name + ".data"

    print(f"Copying the optimized model to {optimized_model_dir}")
    os.makedirs(optimized_model_dir, exist_ok=True)
    shutil.copyfile(merged_model_path, optimized_model_dir / merged_model_name)
    shutil.copyfile(merged_weights_path, optimized_model_dir / merged_weights_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimize", action="store_true", help="Runs the optimization step")
    parser.add_argument("--clean_cache", action="store_true", help="Deletes the Olive cache")
    parser.add_argument("--onnx_file", type=str, help="Path to the .onnx model file", required=True)
    parser.add_argument("--embedding_file", type=str, help="Path to the .pth embeddings file")
    parser.add_argument("--tokenizer_file", type=str, help="Path to the .model tokenizer file")
    parser.add_argument(
        "--prompt", default="Explain to me the difference between nuclear fission and fusion.", type=str
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    optimized_model_dir = script_dir / "models" / "optimized" / os.path.splitext(os.path.basename(args.onnx_file))[0]

    if args.clean_cache:
        shutil.rmtree(script_dir / "cache", ignore_errors=True)

    if args.optimize:
        shutil.rmtree(script_dir / "footprints", ignore_errors=True)
        shutil.rmtree(optimized_model_dir, ignore_errors=True)

    if args.optimize or not optimized_model_dir.exists():
        optimize(args.onnx_file, optimized_model_dir)

    if not args.optimize:
        print("This example doesn't support inference yet")
