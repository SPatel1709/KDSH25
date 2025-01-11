# Freeze all layers except classification head
for name, param in model_with_head.named_parameters():
    param.requires_grad = "classification_head" in name

# Later, unfreeze specific layers
for name, param in model_with_head.named_parameters():
    if "transformer.h.10" in name or "transformer.h.11" in name:  # Example
        param.requires_grad = True
