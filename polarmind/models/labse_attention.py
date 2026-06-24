import torch
import torch.nn as nn
from transformers import AutoModel


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        scores = self.attention(hidden_state).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        return torch.sum(hidden_state * weights.unsqueeze(-1), dim=1)


class LaBSEAttentionClassifier(nn.Module):
    """LaBSE with per-layer attention pooling and concatenated representations."""

    def __init__(self, model_name: str, num_labels: int = 2, num_layers_to_pool: int = 6):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.num_layers_to_pool = num_layers_to_pool
        self.attention_poolers = nn.ModuleList(
            [AttentionPooling(self.hidden_size) for _ in range(num_layers_to_pool)]
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size * num_layers_to_pool, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_labels),
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        selected_layers = outputs.hidden_states[-self.num_layers_to_pool :]
        pooled = [pooler(layer, attention_mask) for layer, pooler in zip(selected_layers, self.attention_poolers)]
        concatenated = torch.cat(pooled, dim=1)
        logits = self.classifier(concatenated)

        if labels is None:
            return logits

        loss_fn = nn.CrossEntropyLoss()
        return loss_fn(logits, labels), logits
