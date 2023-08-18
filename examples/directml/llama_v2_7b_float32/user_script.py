# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
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


def llama_v2_inputs(batch_size, torch_dtype):
    max_seq_len = 8
    num_layers = 32
    num_heads = 32
    hidden_size = 4096

    inputs = {
        "x": torch.rand((batch_size, 1, hidden_size), dtype=torch_dtype),
        "attn_mask": -10000.0 * torch.triu(torch.ones((batch_size, hidden_size // 2, hidden_size // 2)), diagonal=1),
        "k_cache": torch.rand((batch_size, num_layers, max_seq_len, num_heads, hidden_size // num_heads), dtype=torch_dtype),
        "v_cache": torch.rand((batch_size, num_layers, max_seq_len, num_heads, hidden_size // num_heads), dtype=torch_dtype),
        "pos": max_seq_len * torch.ones([], dtype=torch.int64),
    }

    return inputs


def llama_v2_data_loader(data_dir, batch_size, *args, **kwargs):
    return RandomDataLoader(llama_v2_inputs, batch_size, torch.float16)
