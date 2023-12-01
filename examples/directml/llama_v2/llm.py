# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import argparse
import json
import os
import shutil
import sys
import urllib.request
import warnings
from pathlib import Path

import config
import torch
import transformers
from chat_app.app import launch_chat_app
from run_llm_io_binding import run_llm_io_binding
from transformers import AutoTokenizer

from olive.model import ONNXModel
from olive.workflows import run as olive_run


def optimize(optimized_model_dir: Path, model_name: str, model_type: str):
    script_dir = Path(__file__).resolve().parent
    model_info = {}
    submodel_names = ["argmax_sampling", model_name]

    for submodel_name in submodel_names:
        print(f"\nOptimizing {submodel_name}")

        olive_config = None
        config_file_name = "config_llm.json" if submodel_name == model_name else f"config_{submodel_name}.json"

        with Path.open(script_dir / config_file_name) as fin:
            olive_config = json.load(fin)

            # ORT-DML doesn't support SimplifiedLayerNorm or SkipSimplifiedLayerNorm yet, so only enable the fusions if
            # LayerNorm is selected
            if submodel_name == model_name:
                olive_config["input_model"]["config"]["model_path"] = model_name
                olive_config["engine"]["output_name"] = model_name

                olive_config["passes"]["optimize"]["config"]["num_heads"] = config.num_heads
                olive_config["passes"]["optimize"]["config"]["hidden_size"] = config.hidden_size

                if config.model_name == "falcon":
                    # TODO (pavignol): Enable attention and rotary embedding once GQA is working
                    olive_config["passes"]["optimize"]["config"]["optimization_options"]["enable_attention"] = False
                    olive_config["passes"]["optimize"]["config"]["optimization_options"][
                        "use_multi_head_attention"
                    ] = False
                    olive_config["passes"]["optimize"]["config"]["optimization_options"][
                        "enable_rotary_embeddings"
                    ] = False

                    olive_config["passes"]["optimize"]["config"]["optimization_options"]["enable_layer_norm"] = True
                    del olive_config["passes"]["optimize"]["config"]["force_fp32_nodes"]

                # Fewer than 32 layers can be provided for debugging purposes so we have to remove them from the config
                if config.num_layers < 32:
                    model_components = olive_config["input_model"]["config"]["model_components"]
                    for model_component in model_components:
                        layer_range = range(config.num_layers, 32)

                        # Remove the extra inputs
                        key_inputs_to_remove = {f"cache.{idx}.key" for idx in layer_range}
                        value_inputs_to_remove = {f"cache.{idx}.value" for idx in layer_range}
                        input_names = model_component["config"]["io_config"]["input_names"]
                        input_names = [x for x in input_names if x not in key_inputs_to_remove]
                        input_names = [x for x in input_names if x not in value_inputs_to_remove]
                        model_component["config"]["io_config"]["input_names"] = input_names

                        # Remove the extra outputs
                        key_output_to_remove = {f"cache_out.{idx}.key" for idx in layer_range}
                        value_output_to_remove = {f"cache_out.{idx}.value" for idx in layer_range}
                        output_names = model_component["config"]["io_config"]["output_names"]
                        output_names = [x for x in output_names if x not in key_output_to_remove]
                        output_names = [x for x in output_names if x not in value_output_to_remove]
                        model_component["config"]["io_config"]["output_names"] = output_names

                        # Remove the dynamic axes
                        for idx in layer_range:
                            del model_component["config"]["io_config"]["dynamic_axes"][f"cache.{idx}.key"]
                            del model_component["config"]["io_config"]["dynamic_axes"][f"cache.{idx}.value"]

        olive_run(olive_config)

        footprints_file_path = (
            Path(__file__).resolve().parent / "footprints" / f"{submodel_name}_gpu-dml_footprints.json"
        )
        with footprints_file_path.open("r") as footprint_file:
            footprints = json.load(footprint_file)

            conversion_footprint = None
            optimizer_footprint = None
            merging_footprint = None
            for footprint in footprints.values():
                if footprint["from_pass"] == "OnnxConversion":
                    conversion_footprint = footprint
                elif footprint["from_pass"] == "OrtTransformersOptimization":
                    optimizer_footprint = footprint
                elif footprint["from_pass"] == "OptimumMerging":
                    merging_footprint = footprint

            assert conversion_footprint is not None

            if submodel_name == model_name:
                assert optimizer_footprint is not None
                assert merging_footprint is not None
                optimized_olive_model = ONNXModel(**merging_footprint["model_config"]["config"])
            else:
                optimized_olive_model = ONNXModel(**conversion_footprint["model_config"]["config"])

            model_info[submodel_name] = {
                "optimized": {
                    "path": Path(optimized_olive_model.model_path),
                },
            }

            print(f"Optimized Model   : {model_info[submodel_name]['optimized']['path']}")

    print("Copying optimized models...")
    for submodel_name in submodel_names:
        src_path = model_info[submodel_name]["optimized"]["path"]
        dst_path = optimized_model_dir / submodel_name / src_path.name
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copyfile(src_path, dst_path)

        src_weights_path = src_path.with_suffix(".onnx.data")
        if src_weights_path.is_file():
            dst_weights_path = dst_path.with_suffix(".onnx.data")
            shutil.copyfile(src_weights_path, dst_weights_path)

    raw_data_folder = Path(__file__).resolve().parent / "raw_model_data" / model_name / model_type
    raw_data_folder.mkdir(exist_ok=True, parents=True)
    src_tokenizer_path = raw_data_folder / "tokenizer.model"
    dst_tokenizer_path = optimized_model_dir / "tokenizer.model"
    shutil.copyfile(src_tokenizer_path, dst_tokenizer_path)

    print(f"The optimized pipeline is located here: {optimized_model_dir}")


def download_llama_v2_checkpoint(model_type: str):
    raw_data_folder = Path(__file__).resolve().parent / "raw_model_data" / "llama_v2" / model_type
    raw_data_folder.mkdir(exist_ok=True, parents=True)

    license_path = raw_data_folder / "LICENSE"
    use_policy_path = raw_data_folder / "USE_POLICY.md"
    tokenizer_path = raw_data_folder / "tokenizer.model"
    weights_path = raw_data_folder / "weights.pth"

    opener = urllib.request.build_opener()
    opener.addheaders = [("User-agent", "wget")]
    urllib.request.install_opener(opener)

    if not (
        license_path.is_file() and use_policy_path.is_file() and tokenizer_path.is_file() and weights_path.is_file()
    ):
        email_url = input(
            "URL from the e-mail that was received after requesting access from "
            "https://ai.meta.com/resources/models-and-libraries/llama-downloads/ (only valid for 24h): "
        )

    if not license_path.is_file():
        print("Downloading LICENSE")
        urllib.request.urlretrieve(email_url.replace("*", "LICENSE"), license_path)

    if not use_policy_path.is_file():
        print("Downloading Acceptable Usage Policy")
        urllib.request.urlretrieve(email_url.replace("*", "USE_POLICY.md"), use_policy_path)

    if not tokenizer_path.is_file():
        print("Downloading tokenizer")
        urllib.request.urlretrieve(email_url.replace("*", "tokenizer.model"), tokenizer_path)

    if not weights_path.is_file():
        print(f"Downloading llama-2-{model_type}")
        urllib.request.urlretrieve(email_url.replace("*", f"llama-2-{model_type}/consolidated.00.pth"), weights_path)


def download_falcon_checkpoint(model_type: str):
    raw_data_folder = Path(__file__).resolve().parent / "raw_model_data" / "falcon" / model_type
    raw_data_folder.mkdir(exist_ok=True, parents=True)

    weights_path = raw_data_folder / "weights.pth"

    if not weights_path.is_file():
        model_suffix = model_type.replace("chat", "instruct")
        model = f"tiiuae/falcon-{model_suffix}"
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

        pipeline = transformers.pipeline(
            "text-generation", model=model, tokenizer=tokenizer, torch_dtype=torch.float32, device="cpu"
        )

        torch.save(pipeline.model.state_dict(), weights_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimize", action="store_true", help="Runs the optimization step")
    parser.add_argument("--interactive", action="store_true", help="Run with a GUI")
    parser.add_argument(
        "--expose_locally",
        action="store_true",
        help="Expose the web UI on the local network (does nothing if --interactive is not supplied)",
    )
    parser.add_argument("--prompt", default="What is the lightest element?", type=str)
    parser.add_argument("--max_seq_len", default=2048, type=int, help="The size of the cache")
    parser.add_argument("--device_id", default=0, type=int, help="GPU device to use during inference")
    parser.add_argument(
        "--max_gen_len", default=256, type=int, help="The maximum number of tokens that can be included in an answer"
    )
    parser.add_argument(
        "--model_name",
        default="llama_v2",
        choices=["llama_v2", "falcon"],
        type=str,
    )
    parser.add_argument(
        "--model_type",
        default="7b-chat",
        choices=["7b", "7b-chat"],
        help="Which model to convert. The 7b model is the original one without any finetuning, and the 7b-chat "
        "version is the finetuned model optimized for chat.",
        type=str,
    )
    parser.add_argument(
        "--num_layers",
        default=32,
        help="This is a debugging option to be able to quickly generate and optimize an ONNX model with fewer layers "
        "than 32 that barely takes any memory and is easy to load in Netron. This value should ALWAYS be 32 for "
        "production purposes.",
        type=int,
    )
    args = parser.parse_args()

    config.model_name = args.model_name
    config.model_type = args.model_type
    config.num_layers = args.num_layers

    config.hidden_size = {
        "llama_v2": 4096,
        "falcon": 4544,
    }[args.model_name]

    config.num_heads = {
        "llama_v2": 32,
        "falcon": 71,
    }[args.model_name]

    script_dir = Path(__file__).resolve().parent
    optimized_model_dir = script_dir / "models" / "optimized" / args.model_name

    if args.optimize or not optimized_model_dir.exists():
        if args.model_name == "llama_v2":
            download_llama_v2_checkpoint(args.model_type)
        elif args.model_name == "falcon":
            download_falcon_checkpoint(args.model_type)
        else:
            print(f"Model '{args.model_name}' is not supported yet")
            sys.exit(1)

        optimize(optimized_model_dir, args.model_name, args.model_type)

    if not args.optimize:
        if args.interactive:
            launch_chat_app(args.expose_locally)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                run_llm_io_binding(
                    args.prompt,
                    args.max_seq_len,
                    args.max_gen_len,
                    args.device_id,
                    model_name=args.model_name,
                )
