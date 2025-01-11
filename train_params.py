from transformers import TrainingArguments

training_args = TrainingArguments(
    gradient_accumulation_steps=4,  # Accumulate gradients over 4 steps
    per_device_train_batch_size=4,  # Reduce per-device batch size
    learning_rate=2e-5,
    num_train_epochs=3,
    fp16=True,  # Enable mixed precision for speed and memory efficiency
    output_dir="./fine_tuned_llama",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    logging_dir="./logs",
)
