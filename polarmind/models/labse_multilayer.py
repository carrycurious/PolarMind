import torch
import torch.nn as nn
from transformers import AutoModel


class LaBSEMultiLayerClassifier(nn.Module):
    """LaBSE encoder with learnable weighted pooling over the last N layers."""

    def __init__(self, model_name: str, num_labels: int = 2, num_layers_to_aggregate: int = 6):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.hidden_size = self.encoder.config.hidden_size
        self.num_layers = num_layers_to_aggregate
        self.layer_weights = nn.Parameter(torch.ones(self.num_layers))

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_labels),
        )

    def mean_pooling(self, hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
        summed = torch.sum(hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        selected_layers = outputs.hidden_states[-self.num_layers :]
        pooled_layers = [self.mean_pooling(layer, attention_mask) for layer in selected_layers]
        stacked = torch.stack(pooled_layers, dim=-1)
        weights = torch.softmax(self.layer_weights, dim=0)
        weighted = torch.matmul(stacked, weights)
        return self.classifier(weighted)
