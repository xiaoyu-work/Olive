# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import config
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from huggingface_hub import model_info
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

    # All keys in the LoRA state dictionary should have 'lora' somewhere in the string.
    keys = list(lora_state_dict.keys())
    assert all("lora" in k for k in keys)

    if all(key.startswith(submodel_name) for key in keys):
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
            hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=rank
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
        attention.to_q.weight.data += scale * torch.mm(proc.to_q_lora.up.weight, proc.to_q_lora.down.weight)
        attention.to_k.weight.data += scale * torch.mm(proc.to_k_lora.up.weight, proc.to_k_lora.down.weight)
        attention.to_v.weight.data += scale * torch.mm(proc.to_v_lora.up.weight, proc.to_v_lora.down.weight)
        attention.to_out[0].weight.data += scale * torch.mm(proc.to_out_lora.up.weight, proc.to_out_lora.down.weight)


# -----------------------------------------------------------------------------
# TEXT ENCODER
# -----------------------------------------------------------------------------

# This only works for SD 1.4/1.5/2.0/2.1 and .bin weights formats
# TODO: Make it more generic and support more weight types
def generate_text_encoder_lora_inputs():
    lora_inputs = {}

    weight_kinds = ("q", "k", "v", "out")
    weight_positions = ("down", "up")

    for i in range(12):
        for kind in weight_kinds:
            for pos in weight_positions:
                input_shape = (128, 768) if pos == "up" else (768, 128)
                input_name = f"text_encoder.text_model.encoder.layers.{i}.self_attn.to_{kind}_lora.{pos}.weight"
                lora_inputs[input_name] = torch.rand(input_shape, dtype=torch.float16)

    return lora_inputs


def text_encoder_inputs(batchsize, torch_dtype):
    inputs = {
        "input_ids": torch.zeros((batchsize, 77), dtype=torch_dtype),
    }

    if config.lora_weights_strategy == "input_binding":
        inputs.update(generate_text_encoder_lora_inputs())

    return inputs


def text_encoder_load(model_name):
    base_model_id = get_base_model_name(model_name)

    if config.lora_weights_strategy in ("input_binding", "initializers"):
        if base_model_id.endswith(".safetensors"):
            pipe = StableDiffusionPipeline.from_single_file(base_model_id)
        else:
            pipe = StableDiffusionPipeline.from_pretrained(base_model_id)

        if config.lora_weights_file is not None:
            use_safetensors = config.lora_weights_file.endswith(".safetensors")
            pipe.load_lora_weights(config.lora_weights_file, use_safetensors=use_safetensors)
        elif is_lora_model(model_name):
            pipe.load_lora_weights(model_name)

        return pipe.text_encoder

    model = CLIPTextModel.from_pretrained(base_model_id, subfolder="text_encoder")

    if config.lora_weights_strategy == "baked":
        if config.lora_weights_file is not None:
            merge_lora_weights(model, config.lora_weights_file, "text_encoder")
        elif is_lora_model(model_name):
            merge_lora_weights(model, model_name, "text_encoder")

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

# This only works for SD 1.4/1.5/2.0/2.1 and .bin weights formats
# TODO: Make it more generic and support more weight types
def generate_unet_lora_inputs(dtype):
    lora_inputs = {}

    weight_kinds = ("q", "k", "v", "out")
    weight_positions = ("down", "up")

    for i in range(256):
        is_cross_attention = (i % 16) // 8 == 1
        is_k_weight = (i % 8) // 2 == 1
        is_v_weight = (i % 8) // 2 == 2
        is_down_pos = i % 2 == 0

        input_name = "unet"

        if i < 16:
            input_name += ".mid_block.attentions.0"

            if is_down_pos:
                input_shape = (768, 4) if (is_cross_attention and (is_k_weight or is_v_weight)) else (1280, 4)
            else:
                input_shape = (4, 1280)
        elif i < 112:
            block_index = (i - 16) // 32
            input_name += f".down_blocks.{block_index}.attentions.{((i - 16) % 32) // 16}"

            if is_down_pos:
                input_shape = (
                    (768, 4) if (is_cross_attention and (is_k_weight or is_v_weight)) else (320 * (2**block_index), 4)
                )
            else:
                input_shape = (4, 320 * (2**block_index))
        else:
            block_index = (i - 112) // 48
            input_name += f".up_blocks.{block_index + 1}.attentions.{((i - 112) % 48) // 16}"

            if is_down_pos:
                input_shape = (
                    (768, 4)
                    if (is_cross_attention and (is_k_weight or is_v_weight))
                    else (320 * (2 ** (2 - block_index)), 4)
                )
            else:
                input_shape = (4, 320 * (2 ** (2 - block_index)))

        input_name += f".transformer_blocks.0.attn{(i % 16) // 8 + 1}"
        input_name += f".processor.to_{weight_kinds[(i % 8) // 2]}_lora"
        input_name += f".{weight_positions[i % 2]}.weight"
        lora_inputs[input_name] = torch.rand(input_shape, dtype=dtype)

    return lora_inputs


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

        if config.lora_weights_strategy == "input_binding":
            inputs.update(generate_unet_lora_inputs(torch_dtype))

    return inputs


def unet_load(model_name):
    base_model_id = get_base_model_name(model_name)

    if config.lora_weights_strategy in ("input_binding", "initializers"):
        if base_model_id.endswith(".safetensors"):
            pipe = StableDiffusionPipeline.from_single_file(base_model_id)
        else:
            pipe = StableDiffusionPipeline.from_pretrained(base_model_id)

        if config.lora_weights_file is not None:
            use_safetensors = config.lora_weights_file.endswith(".safetensors")
            pipe.load_lora_weights(config.lora_weights_file, use_safetensors=use_safetensors)
        elif is_lora_model(model_name):
            pipe.load_lora_weights(model_name)

        return pipe.unet

    model = UNet2DConditionModel.from_pretrained(base_model_id, subfolder="unet")

    if config.lora_weights_strategy == "baked":
        if config.lora_weights_file is not None:
            merge_lora_weights(model, config.lora_weights_file, "unet")
        elif is_lora_model(model_name):
            merge_lora_weights(model, model_name, "unet")

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
