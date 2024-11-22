import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.custom_logging import setup_logging

log = setup_logging()


class CustomClassifier(nn.Module):

    def __init__(self,
                 img_emb_shape: tuple = (1, 64, 1280),
                 audio_emb_shape: tuple = (1, 1500, 1280),
                 text_emb_shape: tuple = (1, 3, 1024),
                 num_categories: int = 43,
                 num_subcategories: int = 1047):
        super(CustomClassifier, self).__init__()
        self.img_emb_shape = img_emb_shape
        self.audio_emb_shape = audio_emb_shape
        self.text_emb_shape = text_emb_shape
        self.num_categories = num_categories
        self.num_subcategories = num_subcategories

        if len(self.audio_emb_shape) != 3:
            raise ValueError("audio_emb_shape должно быть формата (batch_size, seq_length, feature_dim)")
        if len(self.img_emb_shape) != 3:
            raise ValueError("img_emb_shape должно быть формата (batch_size, seq_length, feature_dim)")
        if len(self.text_emb_shape) != 3:
            raise ValueError("text_emb_shape должно быть формата (batch_size, seq_length, feature_dim)")

        # Слой для обработки аудио
        # self.audio_rnn = nn.GRU(input_size=self.audio_emb_shape[2],
        #                         hidden_size=self.audio_emb_shape[2],
        #                         num_layers=1,
        #                         batch_first=True,
        #                         dtype=torch.float32)

        # Слой для обработки изображений
        # self.img_rnn = nn.GRU(input_size=self.img_emb_shape[2],
        #                       hidden_size=self.img_emb_shape[2],
        #                       num_layers=1,
        #                       batch_first=True,
        #                       dtype=torch.float32)

        # Слой для обработки текста
        # self.text_conv = nn.Conv1d(in_channels=self.text_emb_shape[1],
        #                            out_channels=1,
        #                            kernel_size=3,
        #                            padding=1,
        #                            dtype=torch.float32)

        self.total_emb_dim = self.img_emb_shape[2] + self.audio_emb_shape[2] + self.text_emb_shape[2]

        # Входные полносвязные слои
        self.fc1 = nn.Linear(self.total_emb_dim, 4096)
        self.fc2 = nn.Linear(4096, 2048)

        # Выходные полносвязные слои
        self.category_out = nn.Linear(2048, self.num_categories)
        self.subcategory_out = nn.Linear(2048, self.num_subcategories)

    def forward(self,
                img_emb: torch.Tensor,
                audio_emb: torch.Tensor,
                text_emb: torch.Tensor):

        # Обработка аудио через RNN
        # _, audio_hidden = self.audio_rnn(audio_emb)
        audio_hidden = audio_emb.mean(dim=1)
        audio_hidden = audio_hidden.mean(dim=1)
        # audio_hidden = audio_hidden.squeeze(0)


        # Обработка изображений через RNN
        # _, img_hidden = self.img_rnn(img_emb)
        img_hidden = img_emb.mean(dim=1)
        # img_hidden = img_hidden.squeeze(0)

        # Обработка текста через conv1d
        # txt_hidden = self.text_conv(text_emb)
        txt_hidden = text_emb.mean(dim=1)
        # txt_hidden = txt_hidden.squeeze(0)

        # log.info(f"txt_hidden.shape: {txt_hidden.shape}")
        # log.info(f"img_hidden.shape: {img_hidden.shape}")
        # log.info(f"audio_hidden.shape: {audio_hidden.shape}")

        # Конкатенация всех эмбеддингов
        combined_emb = torch.cat([img_hidden, audio_hidden, txt_hidden], dim=1)

        # Полносвязные слои для сжатия размерности
        x = F.relu(self.fc1(combined_emb))
        x = F.relu(self.fc2(x))

        # Предсказания категорий и подкатегорий
        category_logits = self.category_out(x)
        subcategory_logits = self.subcategory_out(x)

        return category_logits, subcategory_logits
