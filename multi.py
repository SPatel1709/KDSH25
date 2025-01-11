from torch.nn import CrossEntropyLoss

class MultiTaskModel(nn.Module):
    def __init__(self, base_model, num_labels=2):
        super().__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(base_model.config.hidden_size, num_labels)
        self.topic_classifier = nn.Linear(base_model.config.hidden_size, 10)  # Example: 10 topics

    def forward(self, input_ids, attention_mask=None, labels=None, topic_labels=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        cls_hidden_state = hidden_states[:, 0, :]

        logits = self.classifier(cls_hidden_state)
        topic_logits = self.topic_classifier(cls_hidden_state)

        loss = 0
        if labels is not None:
            loss_fn = CrossEntropyLoss()
            loss += loss_fn(logits, labels)
        if topic_labels is not None:
            loss_fn = CrossEntropyLoss()
            loss += loss_fn(topic_logits, topic_labels)

        return loss, logits
