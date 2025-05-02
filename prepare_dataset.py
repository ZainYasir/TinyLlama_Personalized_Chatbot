from datasets import load_dataset
from transformers import AutoTokenizer
import os

def get_tokenized_dataset(model_name="TinyLlama/TinyLlama-1.1B-Chat"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure pad_token is defined

    # Load dataset
    data = load_dataset("json", data_files="/kaggle/working/TinyLlama_Personalized_Chatbot/puck_knowledge_10k.jsonl")["train"]

    def format(example):
        # Customize this according to your JSONL format
        return f"<|user|>: {example['question']}\n<|assistant|>: {example['answer']}"

    def tokenize(example):
        text = format(example)
        return tokenizer(text, truncation=True, padding="max_length", max_length=512)

    tokenized = data.map(tokenize, batched=False)
    return tokenized, tokenizer
