import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class Data2VecMultimodal(nn.Module):
    """
    Data2VecMultimodal: Универсальная модель для объединения трёх модальностей
    (аудио, текста и изображений) в единое латентное пространство.
    """
    def __init__(self, modalities, embed_dims, latent_dim, ema_decay=0.999):
        super(Data2VecMultimodal, self).__init__()

        assert set(modalities).issubset({'audio', 'text', 'vision'}), "Unsupported modalities!"
        self.modalities = modalities
        self.embed_dims = embed_dims
        self.latent_dim = latent_dim

        # Регрессионные головы для приведения к латентному пространству
        self.regression_heads = nn.ModuleDict({
            modality: self._build_regression_head(embed_dims[modality])
            for modality in modalities
        })

        # Attention Fusion для объединения модальностей
        self.attention_fusion = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=4, batch_first=True)

        # VAE для вероятностного представления
        self.vae_encoder = self._build_vae_encoder()
        self.vae_decoder = self._build_vae_decoder()

        # EMA для обучения
        self.ema = EMA(self, ema_decay)

        # Для оценки неопределенности (Dropout)
        self.dropout = nn.Dropout(p=0.1)


    def _build_regression_head(self, input_dim):
        return nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, self.latent_dim)
        )

    def _build_vae_encoder(self):
        return nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim * 2),
            nn.ReLU(),
            nn.Linear(self.latent_dim * 2, self.latent_dim * 2)
        )

    def _build_vae_decoder(self):
        return nn.ModuleDict({
            'audio': nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim * 2),
                nn.ReLU(),
                nn.Linear(self.latent_dim * 2, self.embed_dims['audio'])
            ),
            'text': nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim * 2),
                nn.ReLU(),
                nn.Linear(self.latent_dim * 2, self.embed_dims['text'])
            ),
            'vision': nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim * 2),
                nn.ReLU(),
                nn.Linear(self.latent_dim * 2, self.embed_dims['vision'])
            ),
        })

    def forward(self, inputs, latent=False):
        latent_representations = []

        for modality in self.modalities:
            x = inputs[modality]
            x = self.regression_heads[modality](x)
            latent_representations.append(x)

        assert all([x.shape == latent_representations[0].shape for x in latent_representations]), \
            "All latent representations must have the same shape before stacking!"

        latent_representations = torch.stack(latent_representations, dim=1)
        combined_latent, _ = self.attention_fusion(latent_representations, latent_representations, latent_representations)
        combined_latent = combined_latent.mean(dim=1)

        z_mean, z_log_var = torch.chunk(self.vae_encoder(combined_latent), 2, dim=-1)
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        z = z_mean + eps * std

        if not latent:
            reconstructed = {modality: self.vae_decoder[modality](z) for modality in self.modalities}
            reconstructed = {k: self.dropout(v) for k, v in reconstructed.items()}
        else:
            reconstructed = z

        return reconstructed, z_mean, z_log_var

    def ema_step(self):
        """
        Выполняет шаг EMA для обновления весов.
        """
        self.ema.step(self)

    def get_latent_space(self, inputs):
        """
        Извлечение латентного пространства без обновления EMA.
        """
        with torch.no_grad():
            self.eval()
            latent, _, _ = self.forward(inputs, latent=True)
        return latent


class EMA:
    """
    Класс для экспоненциального скользящего среднего (EMA).
    """
    def __init__(self, model: nn.Module, decay=0.999, skip_keys=None,
                 device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
        self.model = self.deepcopy_model(model)
        self.model.requires_grad_(False)
        self.decay = decay
        self.skip_keys = skip_keys or set()
        self.num_updates = 0
        self.device = device
        self.model.to(device)

    @staticmethod
    def deepcopy_model(model):
        try:
            return copy.deepcopy(model)
        except RuntimeError:
            tmp_path = 'tmp_model_for_ema_deepcopy.pt'
            torch.save(model, tmp_path)
            model = torch.load(tmp_path)
            os.remove(tmp_path)
            return model

    def step(self, new_model: nn.Module):
        ema_state_dict = {}
        ema_params = self.model.state_dict()
        for key, param in new_model.state_dict().items():
            ema_param = ema_params[key].float()
            if key in self.skip_keys:
                ema_param = param.to(dtype=ema_param.dtype).clone()
            else:
                ema_param.mul_(self.decay)
                ema_param.add_(param.to(dtype=ema_param.dtype), alpha=1 - self.decay)
            ema_state_dict[key] = ema_param
        self.model.load_state_dict(ema_state_dict, strict=False)
        self.num_updates += 1

    def restore(self, model: nn.Module):
        model.load_state_dict(self.model.state_dict(), strict=False)
        return model


class MultimodalLoss(nn.Module):
    """
    Лосс для Data2VecMultimodal, учитывающий:
    - VAE Loss (реконструкция и KL-дивергенция)
    - Consistency Loss между модальностями
    - Fusion Loss для общего латентного пространства
    """
    def __init__(self, beta=1.0, alpha=0.1):
        super(MultimodalLoss, self).__init__()
        self.beta = beta
        self.alpha = alpha
        self.reconstruction_loss_fn = nn.MSELoss()
        self.consistency_loss_fn = nn.MSELoss()

    def forward(self, inputs, outputs, z_mean, z_log_var):
        """
        Args:
            inputs (dict): Оригинальные входные данные.
            outputs (torch.Tensor): Восстановленные представления.
            z_mean (torch.Tensor): Среднее латентного распределения.
            z_log_var (torch.Tensor): Логарифм дисперсии латентного распределения.

        Returns:
            torch.Tensor: Итоговое значение лосса.
        """

        # Вычисление размера для паддинга
        max_size = max([x.size(1) for x in inputs.values()])  # Размер по второму измерению (размерность признаков)

        # Паддинг для каждой модальности
        padded_representations = []
        for representation in inputs.values():
            pad_size = max_size - representation.size(1)
            if pad_size > 0:
                # Паддинг слева и справа
                padded_representation = F.pad(representation, (0, pad_size))
            else:
                padded_representation = representation
            padded_representations.append(padded_representation)

        # Подсчет консистентности
        consistency_loss = 0.0
        for i in range(len(padded_representations)):
            for j in range(i + 1, len(padded_representations)):
                consistency_loss += self.consistency_loss_fn(padded_representations[i], padded_representations[j])

        # Средний лосс по всем комбинациям
        num_pairs = len(padded_representations) * (len(padded_representations) - 1) / 2
        consistency_loss /= num_pairs

        # Остальная часть кода для вычисления реконструкции и KL-лосса
        reconstruction_loss = 0.0
        for modality in inputs:
            reconstruction_loss += self.reconstruction_loss_fn(outputs[modality], inputs[modality])
        reconstruction_loss /= len(inputs)

        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
        kl_loss /= z_mean.size(0)

        total_loss = reconstruction_loss + self.beta * kl_loss + self.alpha * consistency_loss

        return total_loss
