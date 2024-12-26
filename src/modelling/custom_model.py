import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import pandas as pd
from src.utils.custom_logging import setup_logging

log = setup_logging()



class CustomClassifier(nn.Module):

    def __init__(self,
                 img_emb_shape: tuple = (1280, ),
                 audio_emb_shape: tuple = (1280, ),
                 text_emb_shape: tuple = (1024, ),
                 num_categories: int = 43,
                 num_subcategories: int = 1024):

        super(CustomClassifier, self).__init__()
        self.img_emb_shape = img_emb_shape
        self.audio_emb_shape = audio_emb_shape
        self.text_emb_shape = text_emb_shape
        self.num_categories = num_categories
        self.num_subcategories = num_subcategories

        self.total_emb_dim = self.img_emb_shape[0] + self.audio_emb_shape[0] + self.text_emb_shape[0]
        log.info(f"total_emb_dim: {self.total_emb_dim}")

        # Полносвязные слои для обработки признаков
        self.fc1 = nn.Linear(self.total_emb_dim, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 4096)
        self.fc4 = nn.Linear(4096, 2048)

        # Для категорий и подкатегорий
        self.subcategory_out = nn.Linear(2048, self.num_subcategories)
        self.category_out = nn.Linear(self.num_subcategories, self.num_categories)


    def forward(self,
                img_emb: torch.Tensor,
                audio_emb: torch.Tensor,
                text_emb: torch.Tensor):

        # log.info(f"txt_hidden.shape: {txt_hidden.shape}")
        # log.info(f"img_hidden.shape: {img_hidden.shape}")
        # log.info(f"audio_hidden.shape: {audio_hidden.shape}")

        # # Конкатенация всех эмбеддингов
        combined_emb = torch.cat([img_emb, audio_emb, text_emb], dim=1)

        # Пропуск через полносвязные слои
        x = F.relu(self.fc1(combined_emb))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        # Получаем логиты для категорий и подкатегорий
        subcategory_logits = self.subcategory_out(x)
        category_logits = self.category_out(subcategory_logits)

        return category_logits, subcategory_logits
