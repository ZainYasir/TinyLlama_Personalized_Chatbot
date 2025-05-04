from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from datasets import load_from_disk
import torch

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Load model & tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
DEVICE = torch.device("cuda:0")  # Force everything on GPU 0

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map={"": DEVICE},  # force entire model on cuda:0
    torch_dtype=torch.float16,
    load_in_4bit=True
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.to(DEVICE)
# Prepare for LoRA
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)

# Load dataset
dataset = load_from_disk("puck_tokenized")

# Training setup
training_args = TrainingArguments(
    output_dir="puck_lora_output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    learning_rate=2e-4,
    bf16=False,
    fp16=True,
    remove_unused_columns=False,
    report_to="none"  # Disable wandb
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer
)

trainer.train()
