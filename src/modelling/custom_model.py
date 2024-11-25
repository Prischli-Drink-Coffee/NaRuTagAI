import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch
import pandas as pd
from src.utils.custom_logging import setup_logging

log = setup_logging()


class GraphReader:
    def __init__(self, graph_path):
        # Читаем граф из файла
        self.graph_path = graph_path
        self.graph = self.load_graph()

    def load_graph(self):
        # Загружаем граф из GEXF файла
        G = nx.read_gexf(self.graph_path)
        return G

    def get_graph_data(self):
        # Получаем список узлов и рёбер графа
        nodes = list(self.graph.nodes)
        edges = list(self.graph.edges)
        log.info(f"Количество узлов: {len(nodes)}")
        log.info(f"Количество рёбер: {len(edges)}")

        # Извлекаем веса рёбер, если они есть, иначе 1.0
        edge_weights = []
        for u, v in edges:
            if 'weight' in self.graph[u][v]:
                edge_weights.append(self.graph[u][v]['weight'])
            else:
                edge_weights.append(1.0)  # Если веса нет, присваиваем 1.0

        # Преобразуем узлы в числовые индексы
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        indexed_edges = [(node_to_idx[u], node_to_idx[v]) for u, v in edges]

        # Конвертируем рёбра в формат, который использует PyTorch Geometric
        edge_index = torch.tensor(indexed_edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float32)

        log.info(f"Количество весов: {len(edge_attr)}")
        return edge_index, edge_attr, node_to_idx


class GraphAttentionNetwork(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_heads: int = 4):
        super(GraphAttentionNetwork, self).__init__()
        self.gat1 = GATConv(in_features, out_features, heads=num_heads, concat=True)
        self.gat2 = GATConv(out_features * num_heads, out_features, heads=1, concat=False)

        # Обучаемые веса рёбер
        self.edge_weight_param = nn.Parameter(torch.ones(1))

    def forward(self, sample, edge_index, edge_attr):

        x = sample
        x = x.view(-1, 1)

        # Применение GAT на первом слое
        x = self.gat1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)

        # Применение GAT на втором слое
        x = self.gat2(x, edge_index, edge_attr=edge_attr)

        # Обновление весов рёбер в зависимости от предсказаний
        edge_attr = edge_attr * self.edge_weight_param

        return x, edge_attr  # Возвращаем выход из сети и обновлённые веса рёбер


class CustomClassifierWithGAT(nn.Module):

    def __init__(self,
                 img_emb_shape: tuple = (1, 64, 1280),
                 audio_emb_shape: tuple = (1, 1500, 1280),
                 text_emb_shape: tuple = (1, 3, 1024),
                 num_categories: int = 43,
                 num_subcategories: int = 1047,
                 ):
        super(CustomClassifierWithGAT, self).__init__()
        self.img_emb_shape = img_emb_shape
        self.audio_emb_shape = audio_emb_shape
        self.text_emb_shape = text_emb_shape
        self.num_categories = num_categories
        self.num_subcategories = num_subcategories

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if len(self.audio_emb_shape) != 3:
            raise ValueError("audio_emb_shape должно быть формата (batch_size, seq_length, feature_dim)")
        if len(self.img_emb_shape) != 3:
            raise ValueError("img_emb_shape должно быть формата (batch_size, seq_length, feature_dim)")
        if len(self.text_emb_shape) != 3:
            raise ValueError("text_emb_shape должно быть формата (batch_size, seq_length, feature_dim)")

        self.total_emb_dim = self.img_emb_shape[2] + self.audio_emb_shape[2] + self.text_emb_shape[2]

        # Полносвязные слои для обработки признаков
        self.fc1 = nn.Linear(self.total_emb_dim, 4096)
        self.fc2 = nn.Linear(4096, 2048)

        # GAT для классификации
        self.gat = GraphAttentionNetwork(in_features=1, out_features=1).to(self.device)

        # Для категорий и подкатегорий
        self.category_out = nn.Linear(2048, self.num_categories)
        self.subcategory_out = nn.Linear(2048, self.num_subcategories)


    def forward(self,
                img_emb: torch.Tensor,
                audio_emb: torch.Tensor,
                text_emb: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor):

        # Обработка эмбеддингов
        audio_emb = audio_emb.mean(dim=1)
        audio_hidden = audio_emb.mean(dim=1)
        img_hidden = img_emb.mean(dim=1)
        txt_hidden = text_emb.mean(dim=1)

        # log.info(f"txt_hidden.shape: {txt_hidden.shape}")
        # log.info(f"img_hidden.shape: {img_hidden.shape}")
        # log.info(f"audio_hidden.shape: {audio_hidden.shape}")

        # # Конкатенация всех эмбеддингов
        combined_emb = torch.cat([img_hidden, audio_hidden, txt_hidden], dim=1)

        # Пропуск через полносвязные слои
        x = F.relu(self.fc1(combined_emb))
        x = F.relu(self.fc2(x))

        # Получаем логиты для категорий и подкатегорий
        category_logits = self.category_out(x)
        subcategory_logits = self.subcategory_out(x)

        # Пропускаем через GAT
        outputs = []
        for sample in subcategory_logits:
            graph_output, edge_attr = self.gat(sample, edge_index.to(self.device),
                                               edge_attr.to(self.device))
            outputs.append(graph_output)
        subcategory_logits = torch.cat(outputs, dim=1).T

        return category_logits, subcategory_logits
