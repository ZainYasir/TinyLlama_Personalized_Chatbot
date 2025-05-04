from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import json

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def format(example):
    prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
    return {"text": prompt}

def tokenize(example):
    output = tokenizer(
        example["text"],
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return {
        "input_ids": output["input_ids"][0],
        "attention_mask": output["attention_mask"][0],
        "labels": output["input_ids"][0].clone(),  # Important fix
    }

# Load your JSONL file
data = []
with open("/kaggle/input/zains-dataset/puck_knowledge_10k.jsonl", "r") as f:
    for line in f:
        data.append(json.loads(line))

dataset = Dataset.from_list(data)
dataset = dataset.map(format)
dataset = dataset.map(tokenize, remove_columns=dataset.column_names)
dataset.save_to_disk("puck_tokenized")
