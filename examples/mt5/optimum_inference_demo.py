import argparse
import gc
import time
from functools import partial

import evaluate
import onnxruntime as ort
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
from transformers import AutoTokenizer

from olive.constants import Framework
from olive.model import OptimumModel, PyTorchModel

ort.set_default_logger_severity(3)

dataset_name = "cnn_dailymail"
subset = "1.0.0"
split = "validation"


"""
Only Use for onnx model from ort script
# mixed precision
! CUDA_VISIBLE_DEVICES=0 python -m onnxruntime.transformers.convert_generation -m google/mt5-base --model_type mt5 \
    --output ~/Olive/examples/mt5/onnx_models/mt5_beam_search.onnx \
    --use_gpu --past_present_share_buffer --use_decoder_masked_attention -e

from olive.model import ONNXModel
model = ONNXModel(
    model_path="~/Olive/examples/mt5/onnx_models/mt5_beam_search.onnx"
)
evaluate_accuracy(model, "", 2, "cuda")
"""
max_length = 1000
min_length = 1
num_beams = 4
num_return_sequences = 1
length_penalty = 1
repetition_penalty = 1


dataset_count = 10


# -------------------- dataset -------------------
def tokenize_and_align_labels(examples, model_name):
    data = [f"summarize: {example}" for example in examples["article"]]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenized_inputs = tokenizer(
        data,
        truncation=True,
        padding=True,
        return_tensors="pt",
        # max_length=128,
        # return_overflowing_tokens=True,
    )
    # pre process
    tokenized_inputs["labels"] = examples["highlights"]
    return tokenized_inputs


def create_evaluation_dataset(model_name):
    raw_dataset = load_dataset(dataset_name, subset, split=split)
    tokenize_func = partial(tokenize_and_align_labels, model_name=model_name)
    tokenized_datasets = raw_dataset.map(
        tokenize_func,
        batched=True,
        remove_columns=raw_dataset.column_names,
    )
    tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask"], output_all_columns=True)

    class _Dateset(Dataset):
        def __init__(self, dataset):
            self.dataset = dataset

        def __getitem__(self, index):
            return self.dataset[index], self.dataset[index]["labels"]

        def __len__(self):
            return dataset_count
            # return len(self.dataset)

    return _Dateset(tokenized_datasets)


def create_dataloader(data_dir="", batch_size=2, model_name="google/mt5-base"):
    def _collate_fn(batch):
        batch = default_collate(batch)
        batch[0].update(
            {
                "max_length": torch.IntTensor([max_length]),
                "min_length": torch.IntTensor([min_length]),
                "num_beams": torch.IntTensor([num_beams]),
                "num_return_sequences": torch.IntTensor([num_return_sequences]),
                "length_penalty": torch.FloatTensor([length_penalty]),
                "repetition_penalty": torch.FloatTensor([repetition_penalty]),
            }
        )
        batch[0]["input_ids"] = batch[0]["input_ids"].to(torch.int32)
        return batch

    dataset = create_evaluation_dataset(model_name)
    return DataLoader(dataset, batch_size=batch_size, collate_fn=_collate_fn)


def evaluate_accuracy(model, data_dir, batch_size, device, model_name="google/mt5-base"):
    prepared_model = model.prepare_session(inference_settings=None, device=device)
    dataloader = create_dataloader(batch_size=batch_size, model_name=model_name)
    rough = evaluate.load("rouge")

    pre = []
    ref = []
    latencies = []

    for raw_item in tqdm(dataloader):
        ref.extend(raw_item[-1])
        item = raw_item[0]
        if model.framework == Framework.PYTORCH:
            prepared_model.to(torch.device("cuda") if device == "cuda" else torch.device("cpu"))
            input_ids = (
                item["input_ids"].to(torch.int64).to(torch.device("cuda") if device == "cuda" else torch.device("cpu"))
            )
            t = time.perf_counter()
            ort_outputs = prepared_model.generate(input_ids)
            latencies.append(time.perf_counter() - t)
            outputs = post_process(ort_outputs, is_ort=False, model_name=model_name)
        elif model.framework == Framework.ONNX:
            inputs = {k: v.cpu().numpy() for k, v in item.items() if k != "labels" and k != "attention_mask"}
            ort_outputs = prepared_model.run(None, inputs)[0]
            outputs = post_process(ort_outputs, model_name=model_name)
        pre.extend(outputs)
    _rls = _evaluate(pre, ref, rough)
    rls = _rls["rouge1"]
    print(pre)
    print(ref)
    return rls, round(dataset_count / sum(latencies), 5)


def _evaluate(pre, ref, computer_func=None):
    if computer_func is None:
        return None
    return computer_func.compute(predictions=pre, references=ref)


def post_process(model_output, is_ort=True, model_name="google/mt5-base"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if is_ort:
        (batch_size, num_sequences, _) = model_output.shape
    else:
        (batch_size, num_sequences) = model_output.shape
    ort_decoded_sequences = []
    for i in range(batch_size):
        _sequences = []
        for j in range(num_sequences):
            decoded_sequence = tokenizer.decode(model_output[i][j], skip_special_tokens=True)
            _sequences.append(decoded_sequence)
        ort_decoded_sequences.append(" ".join(_sequences))
    return ort_decoded_sequences


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MT5 optimization")
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/mt5-base",
        help="Model name, google/mt5-base or google/mt5-large",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device",
    )
    parser.add_argument(
        "--olive_output_model_path",
        type=str,
        default="",
        help="Model path",
    )
    args = parser.parse_args()

    olive_model = OptimumModel(
        model_path=args.olive_output_model_path,
        model_components=["decoder_model_merged.onnx", "encoder_model.onnx"],
        hf_config={
            "model_class": "ORTModelForSeq2SeqLM",
            "model_resource_hub": "optimum.onnxruntime",
            "model_loading_args": {
                "provider": "CUDAExecutionProvider" if args.device == "cuda" else "CPUExecutionProvider",
            },
        },
    )
    rough_val, throughput = evaluate_accuracy(olive_model, "", args.batch_size, args.device, args.model_name)
    print("*" * 20)
    print("Olive optimized MT5 model - rough1: ", rough_val)
    print("Olive optimized MT5 model - throughput (qps): ", throughput)
    print("*" * 20)

    del olive_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    torch_model = PyTorchModel(
        model_path="google/mt5-base",
        hf_config={"model_class": "MT5ForConditionalGeneration", "torch_dtype": torch.float16},
    )
    rough_val, throughput = evaluate_accuracy(torch_model, "", args.batch_size, args.device, args.model_name)
    print("*" * 20)
    print("Torch Fp16 MT5 model - rough1: ", rough_val)
    print("Torch Fp16 MT5 model - throughput (qps): ", throughput)
    print("*" * 20)
    del torch_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
