# Llama2 optimization
This folder contains sample use cases of Olive to optimize a [MT5](https://huggingface.co/google/mt5-base)

## Optimization Workflows
### Optimize using ONNX Runtime Tools
Performs optimization pipeline:
- GPU, FP16: *PyTorch Model -> Optimum Conversion fp16 -> Transformers Optimized Onnx Model -> Optimum Merge decoder*


**Note:** mt5-small is not supported for onnxruntime transformers optimization as the `hidden_size / num_heads != 0`.

## Prerequisites
### Clone the repository and install Olive

Refer to the instructions in the [examples README](../README.md) to clone the repository and install Olive.

### Install extra dependencies
Install the necessary python packages:
```
python -m pip install -r requirements.txt
```

## Run the config to optimize the model
```bash
python -m olive.workflows.run --config mt5_optimum.json

# or run with specific gpu
CUDA_VISIBLE_DEVICES=1,2 python -m olive.workflows.run --config mt5_optimum.json
```

Basically, the output model will be in `cache/models/*_OptimumMerging-*-gpu-cuda/output_model` folder if you do not want to wait Olive packaging the model into zip file.


## Preparation for model evaluation
Since the MT5 model is used for text summarization that requires us to use Optimum's inference API to generate the output for optimized model. Please refer to [Optimum's inference API](https://huggingface.co/docs/optimum/v1.2.1/en/onnxruntime/modeling_ort) for more details.

To use the inference API, we need to move/download the model config into the output model directory. For example, if we want to use the optimized model in `output/mt5_optimum` for inference, we need to ensure `config.json` in `output/mt5_optimum`.
There are several ways to do that:
1. directly copy it locally: e.g. in linux, `cp cache/models/0_OptimumConversion.*/output_model/config.json cache/models/3_OptimumMerging-*-gpu-cuda/output_model/config.json`
2. download it from huggingface: e.g. `AutoConfig.from_pretrained("meta-mt5/mt5-base").save_pretrained("cache/models/3_OptimumMerging-<your_path>-gpu-cuda/output_model")`

## Model evaluation
```
python optimum_inference_demo.py --model_name google/mt5-base --batch_size 2 --device cuda --olive_output_model_path <your model path>
```

In our test, for v100-32GB, when batch_size is 2, we have:
- Olive optimized MT5 model - throughput (qps):  **5.02517**
- Torch Fp16 MT5 model - throughput (qps):  3.09879
