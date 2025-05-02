from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraModel
import torch
import os
from datasets import load_dataset

# Load dataset (tokenized data)
train_data_path = "data/tokenized_train_data.jsonl"
train_data = load_dataset("json", data_files={"train": train_data_path}, split="train")

# Load TinyLlama model
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Apply LoRA with 4-bit precision
lora_model = LoraModel(model, r=8, alpha=16, dropout=0.1)  # Adjust parameters as needed

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',            # output directory
    evaluation_strategy="epoch",       # Evaluate every epoch
    per_device_train_batch_size=4,     # Adjust based on available GPU memory
    per_device_eval_batch_size=4,
    num_train_epochs=3,                # Adjust epochs as needed
    logging_dir='./logs',              # logging directory
    logging_steps=10,                  # log every 10 steps
    save_steps=500,                    # save checkpoint every 500 steps
    fp16=True,                         # enable mixed-precision training for faster performance
    logging_first_step=True           # Log the first step
)

# Initialize Trainer
trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=train_data,
)

# Start fine-tuning
trainer.train()
