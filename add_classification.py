import torch
import torch.nn as nn

class LLaMAForClassification(nn.Module):
    def __init__(self, base_model, num_labels=2):
        super().__init__()
        self.base_model = base_model
        self.classification_head = nn.Sequential(
            nn.Linear(base_model.config.hidden_size, num_labels),
            nn.Softmax(dim=-1)
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # Extract hidden states
        
        cls_hidden_state = hidden_states[:, 0, :]
        logits = self.classification_head(cls_hidden_state)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return loss, logits

        return logits

# Wrap the base model with the classification head
num_labels = 2  # Publishable and Non-Publishable
model_with_head = LLaMAForClassification(model, num_labels)
