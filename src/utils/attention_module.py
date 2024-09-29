import torch
import torch.nn as nn


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attn_weights = nn.Parameter(torch.randn(hidden_size, 1))  # Параметры для обучаемого attention
        self.softmax = nn.Softmax(dim=1)

    def forward(self, outputs_1, outputs_2, outputs_3):
        # Конкатенируем выходы
        outputs = torch.stack([outputs_1, outputs_2, outputs_3], dim=1)  # [batch_size, 3, hidden_size]

        # Применяем attention
        attention_scores = torch.matmul(outputs, self.attn_weights).squeeze(-1)  # [batch_size, 3]
        attention_weights = self.softmax(attention_scores)  # [batch_size, 2]

        # Умножаем выходы на attention веса
        weighted_outputs = outputs * attention_weights.unsqueeze(-1)  # [batch_size, 2, hidden_size]

        # Усредняем взвешенные выходы
        combined_output = weighted_outputs.sum(dim=1)  # [batch_size, hidden_size]

        return combined_output
