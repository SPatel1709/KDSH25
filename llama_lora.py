from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

model_name = "LLaMA-3.1"  
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Add LoRA to the model
lora_config = LoraConfig(
    r=16,  # LoRA rank
    lora_alpha=32,  
    target_modules=["q_proj", "v_proj"],  
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# Print the model with LoRA adapters
print(model)

