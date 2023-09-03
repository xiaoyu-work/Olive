# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import config
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from huggingface_hub import model_info
from lora_weights_conversion import convert_kohya_lora_to_diffusers
from transformers.models.clip.modeling_clip import CLIPTextModel


# Helper latency-only dataloader that creates random tensors with no label
class RandomDataLoader:
    def __init__(self, create_inputs_func, batchsize, torch_dtype):
        self.create_input_func = create_inputs_func
        self.batchsize = batchsize
        self.torch_dtype = torch_dtype

    def __getitem__(self, idx):
        label = None
        return self.create_input_func(self.batchsize, self.torch_dtype), label


def get_base_model_name(model_name):
    if model_name.endswith(".safetensors"):
        return model_name

    return model_info(model_name).cardData.get("base_model", model_name)


def is_lora_model(model_name):
    # TODO: might be a better way to detect (e.g. presence of LORA weights file)
    return model_name != get_base_model_name(model_name)


# Merges LoRA weights into the layers of a base model
def merge_lora_weights(base_model, lora_model_id, submodel_name="unet", scale=1.0):
    from collections import defaultdict
    from functools import reduce

    from diffusers.loaders import LORA_WEIGHT_NAME
    from diffusers.models.attention_processor import LoRAAttnProcessor
    from diffusers.utils import DIFFUSERS_CACHE
    from diffusers.utils.hub_utils import _get_model_file

    # Load LoRA weights
    model_file = _get_model_file(
        lora_model_id,
        weights_name=LORA_WEIGHT_NAME,
        cache_dir=DIFFUSERS_CACHE,
        force_download=False,
        resume_download=False,
        proxies=None,
        local_files_only=False,
        use_auth_token=None,
        revision=None,
        subfolder=None,
        user_agent={
            "file_type": "attn_procs_weights",
            "framework": "pytorch",
        },
    )
    lora_state_dict = torch.load(model_file, map_location="cpu")

    alpha = None
    if all((k.startswith("lora_te_") or k.startswith("lora_unet_")) for k in lora_state_dict.keys()):
        lora_state_dict, alpha = convert_kohya_lora_to_diffusers(lora_state_dict)

    # All keys in the LoRA state dictionary should have 'lora' somewhere in the string.
    keys = list(lora_state_dict.keys())
    assert all("lora" in k for k in keys)

    if any(key.startswith(submodel_name) for key in keys):
        # New format (https://github.com/huggingface/diffusers/pull/2918) supports LoRA weights in both the
        # unet and text encoder where keys are prefixed with 'unet' or 'text_encoder', respectively.
        submodel_state_dict = {k: v for k, v in lora_state_dict.items() if k.startswith(submodel_name)}
    else:
        # Old format. Keys will not have any prefix. This only applies to unet, so exit early if this is
        # optimizing the text encoder.
        if submodel_name != "unet":
            return
        submodel_state_dict = lora_state_dict

    # Group LoRA weights into attention processors
    attn_processors = {}
    lora_grouped_dict = defaultdict(dict)
    for key, value in submodel_state_dict.items():
        attn_processor_key, sub_key = ".".join(key.split(".")[:-3]), ".".join(key.split(".")[-3:])
        lora_grouped_dict[attn_processor_key][sub_key] = value

    for key, value_dict in lora_grouped_dict.items():
        rank = value_dict["to_k_lora.down.weight"].shape[0]
        cross_attention_dim = value_dict["to_k_lora.down.weight"].shape[1]
        hidden_size = value_dict["to_k_lora.up.weight"].shape[0]

        attn_processors[key] = LoRAAttnProcessor(
            hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=rank, network_alpha=alpha
        )
        attn_processors[key].load_state_dict(value_dict)

    # Merge LoRA attention processor weights into existing Q/K/V/Out weights
    for name, proc in attn_processors.items():
        processor_len = -len(".processor")
        if name.startswith(f"{submodel_name}."):
            submodel_name_len = len(f"{submodel_name}.")
            attention_name = name[submodel_name_len:processor_len]
        else:
            attention_name = name[:processor_len]

        attention = reduce(getattr, attention_name.split(sep="."), base_model)
        if submodel_name == "unet":
            attention.to_q.weight.data += scale * torch.mm(proc.to_q_lora.up.weight, proc.to_q_lora.down.weight)
            attention.to_k.weight.data += scale * torch.mm(proc.to_k_lora.up.weight, proc.to_k_lora.down.weight)
            attention.to_v.weight.data += scale * torch.mm(proc.to_v_lora.up.weight, proc.to_v_lora.down.weight)
            attention.to_out[0].weight.data += scale * torch.mm(
                proc.to_out_lora.up.weight, proc.to_out_lora.down.weight
            )
        else:
            attention.self_attn.q_proj.weight.data += scale * torch.mm(
                proc.to_q_lora.up.weight, proc.to_q_lora.down.weight
            )
            attention.self_attn.k_proj.weight.data += scale * torch.mm(
                proc.to_k_lora.up.weight, proc.to_k_lora.down.weight
            )
            attention.self_attn.v_proj.weight.data += scale * torch.mm(
                proc.to_v_lora.up.weight, proc.to_v_lora.down.weight
            )
            attention.self_attn.out_proj.weight.data += scale * torch.mm(
                proc.to_out_lora.up.weight, proc.to_out_lora.down.weight
            )


# This generates a dictionary with all LoRA weights initialized to 0 for SD models. The dictionary contains kohya
# weights since it forces diffusers to insert the network alpha nodes into the model.
def generate_lora_weights():
    lora_weights = {}

    # We use a rank of 1 to generate the placeholder weights, but they can be overriden by weights of any rank during
    # session creation
    lora_rank = 1

    # Generate the text encoder weights
    text_encoder_weight_kinds = ("q", "k", "v", "out")
    for i in range(12):
        for kind in text_encoder_weight_kinds:
            prefix = f"lora_te_text_model_encoder_layers_{i}_self_attn_{kind}_proj"

            alpha_name = f"{prefix}.alpha"
            lora_weights[alpha_name] = torch.ones([], dtype=torch.float16)

            down_name = f"{prefix}.lora_down.weight"
            lora_weights[down_name] = torch.zeros((lora_rank, 768), dtype=torch.float16)

            up_name = f"{prefix}.lora_up.weight"
            lora_weights[up_name] = torch.zeros((768, lora_rank), dtype=torch.float16)

    # Generate the unet weights
    unet_weight_kinds = ("q", "k", "v", "out_0")
    block_positions = ("down", "mid", "up")
    for block_pos in block_positions:
        block_start = {
            "down": 0,
            "mid": 0,
            "up": 1,
        }[block_pos]

        block_count = {
            "down": 3,
            "mid": 1,
            "up": 3,
        }[block_pos]

        attention_count = {
            "down": 2,
            "mid": 1,
            "up": 3,
        }[block_pos]

        block_end = block_start + block_count
        processor_count = 2

        for block_index in range(block_start, block_end):
            block_prefix = (
                f"lora_unet_{block_pos}_block" if block_count == 1 else f"lora_unet_{block_pos}_blocks_{block_index}"
            )

            for attention_index in range(attention_count):
                for proc_index in range(1, 1 + processor_count):
                    prefix = f"{block_prefix}_attentions_{attention_index}_transformer_blocks_0_attn{proc_index}"

                    for kind in unet_weight_kinds:
                        up_outer_dim = {
                            "down": 320 * (2**block_index),
                            "mid": 1280,
                            "up": 320 * (2 ** (block_end - block_index - 1)),
                        }[block_pos]

                        if proc_index == 2 and (kind == "k" or kind == "v"):
                            down_outer_dim = 768
                        else:
                            down_outer_dim = up_outer_dim

                        alpha_name = f"{prefix}_to_{kind}.alpha"
                        lora_weights[alpha_name] = torch.ones([], dtype=torch.float16)

                        down_name = f"{prefix}_to_{kind}.lora_down.weight"
                        lora_weights[down_name] = torch.zeros((lora_rank, down_outer_dim), dtype=torch.float16)

                        up_name = f"{prefix}_to_{kind}.lora_up.weight"
                        lora_weights[up_name] = torch.zeros((up_outer_dim, lora_rank), dtype=torch.float16)

    return lora_weights


# -----------------------------------------------------------------------------
# TEXT ENCODER
# -----------------------------------------------------------------------------


def text_encoder_inputs(batchsize, torch_dtype):
    inputs = {
        "input_ids": torch.zeros((batchsize, 77), dtype=torch_dtype),
    }

    return inputs


def text_encoder_load(model_name):
    base_model_id = get_base_model_name(model_name)

    if config.lora_weights_strategy == "inserted":
        if base_model_id.endswith(".safetensors"):
            pipe = StableDiffusionPipeline.from_single_file(base_model_id)
        else:
            pipe = StableDiffusionPipeline.from_pretrained(base_model_id)

        if is_lora_model(model_name):
            pipe.load_lora_weights(model_name)
        else:
            lora_weights = generate_lora_weights()
            pipe.load_lora_weights(lora_weights)

        return pipe.text_encoder

    model = CLIPTextModel.from_pretrained(base_model_id, subfolder="text_encoder")

    if config.lora_weights_strategy == "folded":
        if is_lora_model(model_name):
            merge_lora_weights(model, model_name, "text_encoder")
        else:
            merge_lora_weights(model, config.lora_weights_file, "text_encoder")

    return model


def text_encoder_conversion_inputs(model):
    return text_encoder_inputs(1, torch.int32)


def text_encoder_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(text_encoder_inputs, batchsize, torch.int32)


# -----------------------------------------------------------------------------
# TEXT ENCODER 2
# -----------------------------------------------------------------------------


def text_encoder_2_inputs(batchsize, torch_dtype):
    return torch.zeros((batchsize, 77), dtype=torch_dtype)


def text_encoder_2_load(model_name):
    base_model_id = get_base_model_name(model_name)
    model = CLIPTextModel.from_pretrained(base_model_id, subfolder="text_encoder_2")
    return model


def text_encoder_2_conversion_inputs(model):
    return text_encoder_2_inputs(1, torch.int32)


def text_encoder_2_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(text_encoder_2_inputs, batchsize, torch.int32)


# -----------------------------------------------------------------------------
# UNET
# -----------------------------------------------------------------------------


def unet_inputs(batchsize, torch_dtype, is_conversion_inputs=False):
    # TODO: Rename onnx::Concat_4 to text_embeds and onnx::Shape_5 to time_ids
    inputs = {
        "sample": torch.rand((batchsize, 4, 64, 64), dtype=torch_dtype),
        "timestep": torch.rand((batchsize,), dtype=torch_dtype),
        "encoder_hidden_states": torch.rand((batchsize, 77, config.image_size + 256), dtype=torch_dtype),
        "return_dict": False,
    }

    if is_conversion_inputs:
        inputs["additional_inputs"] = {
            "added_cond_kwargs": {
                "text_embeds": torch.rand((1, 1280), dtype=torch_dtype),
                "time_ids": torch.rand((1, 5), dtype=torch_dtype),
            }
        }
    else:
        inputs["onnx::Concat_4"] = torch.rand((1, 1280), dtype=torch_dtype)
        inputs["onnx::Shape_5"] = torch.rand((1, 5), dtype=torch_dtype)

    return inputs


def unet_load(model_name):
    base_model_id = get_base_model_name(model_name)

    if config.lora_weights_strategy == "inserted":
        if base_model_id.endswith(".safetensors"):
            pipe = StableDiffusionPipeline.from_single_file(base_model_id)
        else:
            pipe = StableDiffusionPipeline.from_pretrained(base_model_id)

        if is_lora_model(model_name):
            pipe.load_lora_weights(model_name)
        else:
            lora_weights = generate_lora_weights()
            pipe.load_lora_weights(lora_weights)

        return pipe.unet

    model = UNet2DConditionModel.from_pretrained(base_model_id, subfolder="unet")

    if config.lora_weights_strategy == "folded":
        if is_lora_model(model_name):
            merge_lora_weights(model, model_name, "unet")
        else:
            merge_lora_weights(model, config.lora_weights_file, "unet")

    return model


def unet_conversion_inputs(model):
    return tuple(unet_inputs(1, torch.float32, True).values())


def unet_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(unet_inputs, batchsize, torch.float16)


# -----------------------------------------------------------------------------
# VAE ENCODER
# -----------------------------------------------------------------------------


def vae_encoder_inputs(batchsize, torch_dtype):
    return {
        "sample": torch.rand((batchsize, 3, config.image_size, config.image_size), dtype=torch_dtype),
        "return_dict": False,
    }


def vae_encoder_load(model_name):
    base_model_id = get_base_model_name(model_name)

    if base_model_id.endswith(".safetensors"):
        pipe = StableDiffusionPipeline.from_single_file(base_model_id)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(base_model_id)

    pipe.vae.forward = lambda sample, return_dict: pipe.vae.encode(sample, return_dict)[0].sample()
    return pipe.vae


def vae_encoder_conversion_inputs(model):
    return tuple(vae_encoder_inputs(1, torch.float32).values())


def vae_encoder_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(vae_encoder_inputs, batchsize, torch.float16)


# -----------------------------------------------------------------------------
# VAE DECODER
# -----------------------------------------------------------------------------


def vae_decoder_inputs(batchsize, torch_dtype):
    return {
        "latent_sample": torch.rand((batchsize, 4, 64, 64), dtype=torch_dtype),
        "return_dict": False,
    }


def vae_decoder_load(model_name):
    base_model_id = get_base_model_name(model_name)

    if base_model_id.endswith(".safetensors"):
        pipe = StableDiffusionPipeline.from_single_file(base_model_id)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(base_model_id)

    pipe.vae.forward = pipe.vae.decode
    return pipe.vae


def vae_decoder_conversion_inputs(model):
    return tuple(vae_decoder_inputs(1, torch.float32).values())


def vae_decoder_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(vae_decoder_inputs, batchsize, torch.float16)


# -----------------------------------------------------------------------------
# SAFETY CHECKER
# -----------------------------------------------------------------------------


def safety_checker_inputs(batchsize, torch_dtype):
    return {
        "clip_input": torch.rand((batchsize, 3, 224, 224), dtype=torch_dtype),
        "images": torch.rand((batchsize, config.image_size, config.image_size, 3), dtype=torch_dtype),
    }


def safety_checker_load(model_name):
    base_model_id = get_base_model_name(model_name)

    if base_model_id.endswith(".safetensors"):
        pipe = StableDiffusionPipeline.from_single_file(base_model_id)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(base_model_id)

    pipe.safety_checker.forward = pipe.safety_checker.forward_onnx
    return pipe.safety_checker


def safety_checker_conversion_inputs(model):
    return tuple(safety_checker_inputs(1, torch.float32).values())


def safety_checker_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(safety_checker_inputs, batchsize, torch.float16)
