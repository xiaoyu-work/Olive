from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
model = ORTModelForCausalLM.from_pretrained(
    r"models\optimized\mistralai\Mistral-7B-Instruct-v0.1",
    provider="DmlExecutionProvider",
    use_io_binding=False,
)

inputs = tokenizer("My name is Philipp and I live in Germany.", return_tensors="pt").to("cpu")
gen_tokens = model.generate(**inputs, do_sample=True, temperature=0.9, min_length=20, max_length=20)
print(tokenizer.batch_decode(gen_tokens))
