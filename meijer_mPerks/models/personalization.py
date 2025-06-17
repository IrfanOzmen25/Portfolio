from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from data.synthetic_rewards import get_synthetic_coupon_dataset
import torch

dataset = get_synthetic_coupon_dataset()
# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("gpt-2")
model = AutoModelForCausalLM.from_pretrained("gpt-2")

# Make sure tokenizer uses padding if necessary
tokenizer.pad_token = tokenizer.eos_token

# Function to fine-tune GPT-2
def finetune_llm(dataset, epochs=3):
    """
    dataset: list of dicts, e.g. [{'text': "20% off frozen meals for loyalty users"}, ...]
    """
    # Convert to Hugging Face dataset
    raw_dataset = dataset.from_list(dataset)

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=64)

    tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)

    # Define data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # no masking for causal LM
    )

    # Training args
    training_args = TrainingArguments(
        output_dir="./llm-coupon-model",
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        save_steps=100,
        save_total_limit=2,
        logging_steps=10,
        logging_dir="./logs",
        fp16=torch.cuda.is_available(),
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model("./llm-coupon-model")

# Function to generate personalized text
def generate_text(prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        do_sample=True,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
