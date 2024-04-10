# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# This program will run the ONNX version of the LLM.
import time

import numpy as np
import onnxruntime
from transformers import AutoTokenizer

PHI_2_CHAT_TEMPLATE = """"{% for message in messages %}"
{% if message['role'] == 'user' %}
Human: {{ message['content'] }}\nAI:{% endif %}
{% if message['role'] == 'assistant' %}
{{ message['content'] }}{% endif %}
{% endfor %}"""

def run_llm_io_binding(
    onnx_model_path: str,
    prompt: str,
    max_seq_len: int = 2048,
    max_gen_len: int = 256,
    device: str = "dml",
    device_id: int = 0,
    ignore_eos: bool = False,
) -> str:
    onnxruntime.set_default_logger_severity(3)

    execution_provider = {
        "dml": "DmlExecutionProvider",
        "cuda": "CUDAExecutionProvider",
    }[device]

    # Create the ONNX session
    providers = [
        (
            execution_provider,
            {
                "device_id": device_id,
            },
        )
    ]

    if device == "cuda":
        providers[0][1]["enable_cuda_graph"] = True

    llm_session_options = onnxruntime.SessionOptions()
    llm_session_options.add_session_config_entry("ep.dml.enable_graph_capture", "1")

    llm_session = onnxruntime.InferenceSession(
        onnx_model_path,
        sess_options=llm_session_options,
        providers=providers,
    )

    data_type = np.float16
    num_layers = 0
    for inputs_meta in llm_session._inputs_meta:
        if inputs_meta.name.startswith("past_key_"):
            num_layers += 1
            num_key_value_heads = inputs_meta.shape[1]
            head_dim = inputs_meta.shape[3]

    # Initialize the tokenizer and produce the initial tokens.
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/phi-2",
        trust_remote_code=True,
        padding_side="left",
    )
    tokenizer.chat_template = PHI_2_CHAT_TEMPLATE

    initial_input_ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], return_tensors="np")
    initial_input_ids = np.asarray(initial_input_ids, dtype=np.int64)
    initial_input_ids = onnxruntime.OrtValue.ortvalue_from_numpy(initial_input_ids, device)

    position_ids_ortvalue = onnxruntime.OrtValue.ortvalue_from_shape_and_type((1, 1), np.int64, device)
    attention_mask_ortvalue = onnxruntime.OrtValue.ortvalue_from_shape_and_type((1, max_seq_len), np.int64, device)
    input_ids_ortvalue = onnxruntime.OrtValue.ortvalue_from_shape_and_type((1, 1), np.int64, device)

    sequence_length = initial_input_ids.shape()[1]

    # Create the LLM model's I/O binding
    llm_io_binding = llm_session.io_binding()

    # Create the K and V caches.
    cache_shape = (1, num_key_value_heads, max_seq_len, head_dim)
    initial_cache = np.zeros(cache_shape, dtype=data_type)
    k_caches = []
    v_caches = []

    for _ in range(num_layers):
        k_caches.append(onnxruntime.OrtValue.ortvalue_from_numpy(initial_cache, device))
        v_caches.append(onnxruntime.OrtValue.ortvalue_from_numpy(initial_cache, device))

    llm_io_binding.bind_output("logits", device)
    llm_io_binding.bind_ortvalue_input("input_ids", initial_input_ids)

    initial_position_ids = np.arange(sequence_length, dtype=np.int64).reshape((1, sequence_length))
    llm_io_binding.bind_cpu_input("position_ids", initial_position_ids)

    for layer_idx in range(num_layers):
        llm_io_binding.bind_ortvalue_input(f"past_key_{layer_idx}", k_caches[layer_idx])
        llm_io_binding.bind_ortvalue_input(f"past_value_{layer_idx}", v_caches[layer_idx])
        llm_io_binding.bind_ortvalue_output(f"present_key_{layer_idx}", k_caches[layer_idx])
        llm_io_binding.bind_ortvalue_output(f"present_value_{layer_idx}", v_caches[layer_idx])

    attention_mask = np.pad(np.ones((1, sequence_length), dtype=np.int64), ((0, 0), (0, max_seq_len - sequence_length)))
    llm_io_binding.bind_ortvalue_input("attention_mask", attention_mask_ortvalue)

    run_options = onnxruntime.RunOptions()

    before_time = time.perf_counter()

    # Iteratively generate tokens.
    output_tokens = []
    for idx in range(max_gen_len):
        if idx > 0:
            position_ids = np.array(sequence_length - 1, dtype=np.int64, ndmin=2)
            position_ids_ortvalue.update_inplace(position_ids)

        if idx == 1:
            llm_io_binding.bind_ortvalue_input("position_ids", position_ids_ortvalue)
            llm_io_binding.bind_ortvalue_input("input_ids", input_ids_ortvalue)

        attention_mask[0, sequence_length - 1] = 1
        attention_mask_ortvalue.update_inplace(attention_mask)

        # Run the LLM
        if idx == 1:
            run_options.add_run_config_entry("gpu_graph_id", "1")

        llm_session.run_with_iobinding(llm_io_binding, run_options)

        # Decide the next token using your preferred sampling strategy.
        logits = llm_io_binding.get_outputs()[0].numpy()[:, -1, :]
        next_token = np.argmax(logits, axis=-1, keepdims=True)
        output_tokens.append(next_token.item())

        # Set the token for the next iteration
        input_ids_ortvalue.update_inplace(next_token)

        # Stop if/when we get an ENDOFTEXT token before reaching maximum sequence length
        if not ignore_eos and output_tokens[-1] == tokenizer.eos_token_id:
            break

        if idx == 0:
            llm_io_binding.bind_output("logits", device)

        sequence_length += 1
        sequence_length = min(sequence_length, max_seq_len)

    after_time = time.perf_counter()
    duration = after_time - before_time
    tokens_per_second = idx / duration

    # Only print the tokens/s when ignore_eos is provided for benchmarking purposes
    if ignore_eos:
        print(f"Execution took {duration:0.4f} seconds (generated {tokens_per_second:0.2f} tokens per second)")

    output_str = tokenizer.decode(output_tokens, skip_special_tokens=True)
    return [output_str]


def run(
    prompt,
    onnx_model_path,
    device_id,
    max_length=200,
):
    texts = run_llm_io_binding(onnx_model_path, prompt=prompt[0], max_gen_len=max_length, device_id=device_id)

    for i, text in enumerate(texts):
        print(f"Prompt: {prompt[i]}")  # noqa: T201
        yield text
