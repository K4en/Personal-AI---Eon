import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset
from torch.utils.data import DataLoader
from peft import PeftModel

from mistral_lora_train import bnb_config

#Paths
base_model_name = "mistralai/Mistral-7B-Instruct-v0.2"
adapter_path = "./mistral_lora_Eon_qlora"
dataset_file = "chatgpt_converted_dataset.jsonl"
batch_size = 1

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(adapter_path)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

# Load and split dataset
ds = load_dataset("json", data_files=dataset_file, split="train[:10%]")
def tokenize(example):
    enc = tokenizer(example["prompt"] +"\n" + example["response"],
                    truncation=True, padding="max_length", max_length=512)
    enc["labels"] = enc["input_ids"].copy()
    return enc

eval_ds = ds.map(tokenize, remove_columns=ds.column_names)

# Use HF data collator so batches already tensors
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

loader = DataLoader(eval_ds, batch_size=1, collate_fn=collator)

# Compute cross-entropy loss
total_loss = 0.0
total_tokens = 0

with torch.no_grad():
    for batch in loader:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(**batch)
        # loss is average over batch, multiply by tokens to get sum
        non_pads = batch["labels"].ne(tokenizer.pad_token_id).sum().item()
        total_loss += outputs.loss.item() * non_pads
        total_tokens += non_pads

perplexity = torch.exp(torch.tensor(total_loss) / total_tokens)
print(f"Perplexity: {perplexity}")