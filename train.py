import torch
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from prepare_dataset import get_tokenized_dataset

# Load data
tokenized_dataset, tokenizer = get_tokenized_dataset()

# Load model with 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat",
    load_in_4bit=True,
    device_map="auto",
    trust_remote_code=True
)

# Prepare model for LoRA
model = prepare_model_for_kbit_training(model)

# LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # for TinyLlama; may adjust after inspecting model
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Training args
training_args = TrainingArguments(
    output_dir="tinylama-lora-4bit-output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    push_to_hub=False,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# Train
trainer.train()

# Save adapter
model.save_pretrained("tinylama-lora-4bit-adapter")
