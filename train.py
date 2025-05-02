import torch
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModelForCausalLM, PeftConfig
import bitsandbytes as bnb
from datasets import load_dataset
import os

# Load the dataset
dataset = load_dataset("json", data_files="puck_knowledge_10k.jsonl", split="train")

# Load Tokenizer (Ensure tokenizer is not deprecated)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")

# Load the model with the correct configuration
config = PeftConfig.from_pretrained("path_to_peft_config")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")

# Use bitsandbytes configuration for quantization (e.g., 8-bit or 4-bit)
model = model.from_pretrained(
    "EleutherAI/gpt-neo-2.7B",
    load_in_8bit=True,  # Enables 8-bit quantization
    device_map="auto"  # Automatically distribute model across available devices (if using GPU/TPU)
)

# Applying LoRA with proper configuration
lora_model = PeftModelForCausalLM.from_pretrained(
    model,
    config=config
)

# Tokenizing the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Tokenize the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Set up training arguments with `use_cache=False`
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    save_steps=500,
    logging_dir="./logs",
    logging_steps=100,
    load_best_model_at_end=True,
    save_total_limit=2,
    gradient_checkpointing=True,
    use_cache=False,  # Ensure no cache with gradient checkpointing
    fp16=True,  # Mixed precision for efficiency
)

# Set up the Trainer
trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,
    tokenizer=tokenizer,
    compute_metrics=None,  # Add custom metrics if needed
)

# Start the training
trainer.train()

# Save the base model and adapter weights
model.save_pretrained("./results/base_model")  # Save the base model
tokenizer.save_pretrained("./results/base_model")  # Save the tokenizer

# Save the LoRA adapter
lora_model.save_adapter("./results/lora_adapter")  # Save the LoRA adapter weights
import torch
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModelForCausalLM, PeftConfig
import bitsandbytes as bnb
from datasets import load_dataset
import os

# Load the dataset
dataset = load_dataset("json", data_files="puck_knowledge_10k.jsonl", split="train")

# Load Tokenizer (Ensure tokenizer is not deprecated)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")

# Load the model with the correct configuration
config = PeftConfig.from_pretrained("path_to_peft_config")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")

# Use bitsandbytes configuration for quantization (e.g., 8-bit or 4-bit)
model = model.from_pretrained(
    "EleutherAI/gpt-neo-2.7B",
    load_in_8bit=True,  # Enables 8-bit quantization
    device_map="auto"  # Automatically distribute model across available devices (if using GPU/TPU)
)

# Applying LoRA with proper configuration
lora_model = PeftModelForCausalLM.from_pretrained(
    model,
    config=config
)

# Tokenizing the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Tokenize the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Set up training arguments with `use_cache=False`
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    save_steps=500,
    logging_dir="./logs",
    logging_steps=100,
    load_best_model_at_end=True,
    save_total_limit=2,
    gradient_checkpointing=True,
    use_cache=False,  # Ensure no cache with gradient checkpointing
    fp16=True,  # Mixed precision for efficiency
)

# Set up the Trainer
trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,
    tokenizer=tokenizer,
    compute_metrics=None,  # Add custom metrics if needed
)

# Start the training
trainer.train()

# Save the base model and adapter weights
model.save_pretrained("./results/base_model")  # Save the base model
tokenizer.save_pretrained("./results/base_model")  # Save the tokenizer

# Save the LoRA adapter
lora_model.save_adapter("./results/lora_adapter")  # Save the LoRA adapter weights
