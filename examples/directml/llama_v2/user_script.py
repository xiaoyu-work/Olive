# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from collections import OrderedDict
from pathlib import Path

import config
import torch
from argmax_sampling_model import ArgmaxSampling
from decoder_model import DecoderModel


# Helper latency-only dataloader that creates random tensors with no label
class RandomDataLoader:
    def __init__(self, create_inputs_func, batch_size):
        self.create_input_func = create_inputs_func
        self.batch_size = batch_size

    def __getitem__(self, idx):
        label = None
        return self.create_input_func(self.batch_size), label


# -----------------------------------------------------------------------------
# ARGMAX SAMPLING
# -----------------------------------------------------------------------------


def load_argmax_sampling_model(model_path):
    model = ArgmaxSampling()
    model.eval()
    return model


def argmax_sampling_inputs(model):
    batch_size = 2

    vocab_size = {
        "llama_v2": 32000,
        "falcon": 65024,
    }[config.model_name]

    return torch.zeros((batch_size, vocab_size), dtype=torch.float16)


def argmax_sampling_data_loader(data_dir, batch_size, *args, **kwargs):
    return RandomDataLoader(argmax_sampling_inputs, batch_size)


# -----------------------------------------------------------------------------
# DECODER
# -----------------------------------------------------------------------------


def get_or_create_llama_v2_decoder_model():
    vocab_size = 32000
    scale_type = "SquareRootHeadDim"

    # Lazily load the decoder model the first time it's requested. This is necessary because both the cache and
    # no_cache models need to share the same instance in order to export their common weights with the same names.
    # Not doing so would result identical weights having different names in both models, which makes merging them
    # very difficult.
    if config.decoder_model is None:
        config.decoder_model = DecoderModel(
            config.num_layers,
            vocab_size,
            config.hidden_size,
            config.num_heads,
            scale_type,
            config.model_name,
        )
        config.decoder_model.eval()

        script_dir = Path(__file__).resolve().parent
        weights_path = script_dir / "raw_model_data" / config.model_name / config.model_type / "weights.pth"
        state_dict = torch.load(weights_path)

        # Permutation for sliced rotary
        def permute(weight):
            return (
                weight.view(config.num_heads, config.hidden_size // config.num_heads // 2, 2, config.hidden_size)
                .transpose(1, 2)
                .reshape(config.hidden_size, config.hidden_size)
            )

        for layer_idx in range(config.num_layers):
            state_dict[f"layers.{layer_idx}.attention.wq.weight"] = permute(
                state_dict[f"layers.{layer_idx}.attention.wq.weight"]
            )
            state_dict[f"layers.{layer_idx}.attention.wk.weight"] = permute(
                state_dict[f"layers.{layer_idx}.attention.wk.weight"]
            )

        # We don't use rope.freqs
        del state_dict["rope.freqs"]
        strict = config.num_layers == 32
        config.decoder_model.load_state_dict(state_dict, strict=strict)

    return config.decoder_model


def get_or_create_falcon_decoder_model():
    vocab_size = 65024
    scale_type = "SquareRootHeadDim"

    # Lazily load the decoder model the first time it's requested. This is necessary because both the cache and
    # no_cache models need to share the same instance in order to export their common weights with the same names.
    # Not doing so would result identical weights having different names in both models, which makes merging them
    # very difficult.
    if config.decoder_model is None:
        config.decoder_model = DecoderModel(
            config.num_layers,
            vocab_size,
            config.hidden_size,
            config.num_heads,
            scale_type,
            config.model_name,
        )
        config.decoder_model.eval()

        script_dir = Path(__file__).resolve().parent
        weights_path = script_dir / "raw_model_data" / config.model_name / config.model_type / "weights.pth"

        state_dict = torch.load(weights_path)
        new_dict = OrderedDict()
        head_dim = config.hidden_size // config.num_heads

        # Falcon has weights in qkv format, so split them to match our architecture
        for layer_idx in range(config.num_layers):
            # Split the qkv weight in 3 and rename them
            qkv_weight = state_dict[f"transformer.h.{layer_idx}.self_attention.query_key_value.weight"]
            qkv_weight = qkv_weight.view(config.num_heads + 2, -1, config.hidden_size)

            new_dict[f"layers.{layer_idx}.attention.wq.weight"] = (
                qkv_weight[:-2, :, :].reshape([config.hidden_size, config.hidden_size]).to(torch.float32)
            )
            new_dict[f"layers.{layer_idx}.attention.wk.weight"] = (
                qkv_weight[[-2], :, :].reshape([head_dim, config.hidden_size]).to(torch.float32)
            )
            new_dict[f"layers.{layer_idx}.attention.wv.weight"] = (
                qkv_weight[[-1], :, :].reshape([head_dim, config.hidden_size]).to(torch.float32)
            )

            # Rename the attention linear weight
            attn_linear_weight = state_dict[f"transformer.h.{layer_idx}.self_attention.dense.weight"]
            new_dict[f"layers.{layer_idx}.attention.wo.weight"] = attn_linear_weight.to(torch.float32)

            # Rename the attention norm weight
            attn_norm_weight = state_dict[f"transformer.h.{layer_idx}.input_layernorm.weight"]
            new_dict[f"layers.{layer_idx}.attention_norm.weight"] = attn_norm_weight.to(torch.float32)

            # Rename the attention norm bias
            attn_norm_bias = state_dict[f"transformer.h.{layer_idx}.input_layernorm.bias"]
            new_dict[f"layers.{layer_idx}.attention_norm.bias"] = attn_norm_bias.to(torch.float32)

            # Rename the to_4h weight
            to_4h_weight = state_dict[f"transformer.h.{layer_idx}.mlp.dense_h_to_4h.weight"]
            new_dict[f"layers.{layer_idx}.feed_forward.to_4h.weight"] = to_4h_weight.to(torch.float32)

            # Rename the to_h weight
            to_h_weight = state_dict[f"transformer.h.{layer_idx}.mlp.dense_4h_to_h.weight"]
            new_dict[f"layers.{layer_idx}.feed_forward.to_h.weight"] = to_h_weight.to(torch.float32)

            # Add the fillers
            new_dict[f"layers.{layer_idx}.ffn_norm.weight"] = torch.ones(config.hidden_size).to(torch.float32)
            new_dict[f"layers.{layer_idx}.ffn_norm.bias"] = torch.ones(config.hidden_size).to(torch.float32)
            # new_dict[f"layers.{layer_idx}.attention.wq.bias"] = (
            #     torch.zeros(config.hidden_size).reshape([config.hidden_size]).to(torch.float32)
            # )
            # new_dict[f"layers.{layer_idx}.attention.wk.bias"] = (
            #     torch.zeros(headDim).reshape([headDim]).to(torch.float32)
            # )
            # new_dict[f"layers.{layer_idx}.attention.wv.bias"] = (
            #     torch.zeros(headDim).reshape([headDim]).to(torch.float32)
            # )
            # new_dict[
            #     f"layers.{layer_idx}.attention.wo.bias"
            # ] = torch.zeros(config.hidden_size).to(torch.float32)
            # new_dict[
            #     f"layers.{layer_idx}.feed_forward.to_4h.bias"
            # ] = torch.zeros(config.hidden_size*4).to(torch.float32)
            # new_dict[
            #     f"layers.{layer_idx}.feed_forward.to_h.bias"
            # ] = torch.zeros(config.hidden_size).to(torch.float32)

        new_dict["norm.weight"] = state_dict["transformer.ln_f.weight"].to(torch.float32)
        new_dict["norm.bias"] = state_dict["transformer.ln_f.bias"].to(torch.float32)
        new_dict["output.weight"] = state_dict["lm_head.weight"].to(torch.float32)
        new_dict["tok_embeddings.weight"] = state_dict["transformer.word_embeddings.weight"]

        strict = config.num_layers == 32
        config.decoder_model.load_state_dict(new_dict, strict=strict)

    return config.decoder_model


def load_decoder_model(model_path):
    model = {
        "llama_v2": get_or_create_llama_v2_decoder_model,
        "falcon": get_or_create_falcon_decoder_model,
    }[config.model_name]()

    model.set_use_cache(False)
    return model


def decoder_inputs(model):
    batch_size = 2
    seq_len = 10
    max_seq_len = 256
    head_size = config.hidden_size // config.num_heads

    key_value_shape = {
        "llama_v2": (batch_size, config.num_heads, max_seq_len, head_size),
        "falcon": (batch_size, 1, max_seq_len, head_size),
    }[config.model_name]

    return {
        "tokens": torch.zeros((batch_size, seq_len), dtype=torch.int64),
        "position_ids": torch.zeros((batch_size, seq_len), dtype=torch.int64),
        "attn_mask": torch.zeros((batch_size, max_seq_len), dtype=torch.int32),
        "cache": [
            {
                "key": torch.rand(key_value_shape, dtype=torch.float32),
                "value": torch.rand(key_value_shape, dtype=torch.float32),
            }
            for _ in range(config.num_layers)
        ],
    }


# -----------------------------------------------------------------------------
# DECODER WITH PAST
# -----------------------------------------------------------------------------


def load_decoder_with_past_model(model_path):
    model = {
        "llama_v2": get_or_create_llama_v2_decoder_model,
        "falcon": get_or_create_falcon_decoder_model,
    }[config.model_name]()

    model.set_use_cache(True)
    return model


def decoder_with_past_inputs(model):
    batch_size = 2
    max_seq_len = 256
    head_size = config.hidden_size // config.num_heads

    key_value_shape = {
        "llama_v2": (batch_size, config.num_heads, max_seq_len, head_size),
        "falcon": (batch_size, 1, max_seq_len, head_size),
    }[config.model_name]

    return {
        "tokens_increment": torch.zeros((batch_size, 1), dtype=torch.int64),
        "position_ids_increment": torch.zeros((batch_size, 1), dtype=torch.int64),
        "attn_mask": torch.zeros((batch_size, max_seq_len), dtype=torch.int32),
        "cache": [
            {
                "key": torch.rand(key_value_shape, dtype=torch.float32),
                "value": torch.rand(key_value_shape, dtype=torch.float32),
            }
            for _ in range(config.num_layers)
        ],
    }


# -----------------------------------------------------------------------------
# MERGED DECODERS
# -----------------------------------------------------------------------------


def merged_decoders_inputs(model):
    batch_size = 2
    max_seq_len = 256
    head_size = config.hidden_size // config.num_heads
    seq_len = 10

    key_value_shape = {
        "llama_v2": (batch_size, config.num_heads, max_seq_len, head_size),
        "falcon": (batch_size, 1, max_seq_len, head_size),
    }[config.model_name]

    inputs = {
        "tokens": torch.zeros((batch_size, seq_len), dtype=torch.int64),
        "position_ids": torch.zeros((batch_size, seq_len), dtype=torch.int64),
        "attn_mask": torch.zeros((batch_size, max_seq_len), dtype=torch.int32),
    }

    for layer_idx in range(config.num_layers):
        inputs[f"cache.{layer_idx}.key"] = torch.rand(key_value_shape, dtype=torch.float32)
        inputs[f"cache.{layer_idx}.value"] = torch.rand(key_value_shape, dtype=torch.float32)

    inputs["tokens_increment"] = torch.zeros((batch_size, 1), dtype=torch.int64)
    inputs["position_ids_increment"] = torch.zeros((batch_size, 1), dtype=torch.int64)
    inputs["use_cache_branch"] = torch.ones((1,), dtype=torch.bool)

    return inputs


def merged_decoders_data_loader(data_dir, batch_size, *args, **kwargs):
    return RandomDataLoader(merged_decoders_inputs, batch_size)
