from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline
import torch

# Model path
model_path = "./mistral_lora_Eon_qlora"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)

# Wrap in simple pipeline
pipe = TextGenerationPipeline(model, tokenizer, device=0, max_new_tokens=64, do_sample=True, top_p=0.95, temperature=0.95)

# Custom prompts

prompts = [
    "Translate to French: Hello, how are you today?",
    "Write me a short poem about the solar system",
    "Explain quantum computing in simple terms",
    "Plan a workout routine to get from 78kg to 85kg for a 190cm tall person with high metabolism rate."
]

for p in prompts:
    out = pipe(p)[0]["generated_text"]
    print(f"\n Prompt: {p} \n Response: {out}\n{'-'*60}")