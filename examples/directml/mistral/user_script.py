# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import config
import torch


# Helper latency-only dataloader that creates random tensors with no label
class RandomDataLoader:
    def __init__(self, create_inputs_func, batch_size, torch_dtype):
        self.create_input_func = create_inputs_func
        self.batch_size = batch_size
        self.torch_dtype = torch_dtype

    def __getitem__(self, idx):
        label = None
        return self.create_input_func(self.batch_size, self.torch_dtype), label


def mistral_inputs(batch_size, torch_dtype):
    num_layers = 32
    num_key_value_heads = 8
    max_seq_len = 256
    num_heads = 32
    head_size = config.hidden_size // num_heads
    seq_len = 10

    inputs = {
        "input_ids": torch.randint(10, (batch_size, seq_len), dtype=torch.int64),
        "attention_mask": torch.randint(10, (batch_size, max_seq_len), dtype=torch.int64),
    }

    for layer_index in range(num_layers):
        inputs[f"past_key_values.{layer_index}.key"] = torch.rand(
            (batch_size, num_key_value_heads, max_seq_len, head_size), dtype=torch_dtype
        )
        inputs[f"past_key_values.{layer_index}.value"] = torch.rand(
            (batch_size, num_key_value_heads, max_seq_len, head_size), dtype=torch_dtype
        )

    inputs["use_cache_branch"] = torch.ones((1,), dtype=torch.bool)

    return inputs


def mistral_data_loader(data_dir, batch_size, *args, **kwargs):
    return RandomDataLoader(mistral_inputs, batch_size, torch.float16)
