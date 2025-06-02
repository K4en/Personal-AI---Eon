import torch
from transformers import (
AutoTokenizer,
AutoModelForCausalLM,
pipeline
)
from peft import PeftModel

from mistral_lora_train import bnb_config

# Config
base_model = "mistralai/Mistral-7B-Instruct-v0.2"
adapter_dir = "./mistral_lora_Eon_qlora"


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(adapter_dir)

base = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto",
    offload_folder="offload",
    offload_state_dict=True,
)

# Inject Lora adapter
model = PeftModel.from_pretrained(base, adapter_dir, device_map="auto",
    offload_folder="offload",
    offload_state_dict=True,)
model.eval()
tokenizer.pad_token = tokenizer.eos_token

from transformers import TextGenerationPipeline
# Setup Pipeline
pipe = TextGenerationPipeline(
    model,
    tokenizer,
    max_new_tokens=128,
    do_sample=True,
    top_p=0.9,
    temperature=0.7
)
if __name__ == "__main__":
    prompt = "What's the meaning of life?"
    output = pipe(prompt)[0]["generated_text"]
    print(output)