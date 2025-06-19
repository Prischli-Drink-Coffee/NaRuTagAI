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

        self.total_emb_dim = img_emb_shape[0] + audio_emb_shape[0] + text_emb_shape[0]
        dropout_rate = 0.2
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.total_emb_dim, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, 2048)
        )

        self.category_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(2048, num_categories)
        )

        self.subcategory_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(2048 + num_categories, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, num_subcategories)
        )


    def forward(self,
                img_emb: torch.Tensor,
                audio_emb: torch.Tensor,
                text_emb: torch.Tensor):

        # log.info(f"txt_hidden.shape: {txt_hidden.shape}")
        # log.info(f"img_hidden.shape: {img_hidden.shape}")
        # log.info(f"audio_hidden.shape: {audio_hidden.shape}")

        combined_emb = torch.cat([img_emb, audio_emb, text_emb], dim=1)
        shared_features = self.feature_extractor(combined_emb)
        category_logits = self.category_head(shared_features)
        combined_for_subcategory = torch.cat([shared_features, category_logits], dim=1)
        subcategory_logits = self.subcategory_head(combined_for_subcategory)

        return category_logits, subcategory_logits
