import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from omegaconf import DictConfig


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """

        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        self.query = nn.Parameter(torch.randn(hidden_size, 1))

    def forward(self, enc_outputs):
        scores = torch.matmul(enc_outputs, self.query) / (enc_outputs.shape[-1] ** 0.5)
        weights = F.softmax(scores, dim=1)
        weighted = weights * enc_outputs
        context = torch.sum(weighted, dim=1)
        return context, weighted, weights


class AttentionModel(nn.Module):
    def __init__(self, n_features, hparam: DictConfig):
        super().__init__()

        pl.seed_everything(hparam.seed, workers=True)
        self.input_size = n_features
        self.hidden_size = hparam.transformer.n_hidden
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.input_size))
        self.embedding = nn.Linear(n_features, self.hidden_size)
        self.pos_encoding = PositionalEncoding(self.hidden_size)
        self.TransformerEncoderLayer = nn.TransformerEncoderLayer(d_model=self.hidden_size,
                                                                  dim_feedforward=hparam.transformer.n_feed_forward,
                                                                  nhead=hparam.transformer.n_head, dropout=hparam.train.dropout)
#                                                                   batch_first=True)
        self.TransformerEncoder = nn.TransformerEncoder(self.TransformerEncoderLayer, num_layers=hparam.transformer.n_layers)
        self.attention = TemporalAttention(self.hidden_size)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.hidden_size, hparam.mlp.h1),
            nn.ReLU(),
            nn.Linear(hparam.mlp.h1, hparam.mlp.h2),
            nn.ReLU(),
            nn.Linear(hparam.mlp.h2, hparam.mlp.out),
        )

    def forward(self, x):

#         batch_size = x.size(0)
#         cls_tokens = self.cls_token.expand(batch_size, -1, -1)
#         x = torch.cat((cls_tokens, x), dim=1)

        x = self.embedding(x)
        x = x.transpose(0, 1)
        x = x * (self.hidden_size ** 0.5)
        x = self.pos_encoding(x)
        x = self.TransformerEncoder(x)
        x = x.transpose(0, 1)
#         output = self.mlp(x[:, 0, :])
        
        context, weighted, weights = self.attention(x)
        output = self.mlp(context)

        return x, output

class EnsembleModel(pl.LightningModule):

    def __init__(self, n_features: int, config: DictConfig,beta=1.0, gamma=1.0):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.beta = beta
        self.gamma = gamma
        self.models = nn.ModuleList([AttentionModel(n_features, config) for _ in range(config.transformer.n_models)])
        self.criterion_reg = nn.MSELoss()
        self.criterion_triplet = [nn.TripletMarginWithDistanceLoss(margin=config.train.triplet_margin * (i + 1)) for i in range(3)]
        self.batch_norm = nn.BatchNorm1d(n_features)

    def forward(self, x, labels=None):
        x = x.permute(0, 2, 1)
        x = self.batch_norm(x)
        x = x.permute(0, 2, 1)

        all_transformers_outputs = []
        all_mlp_outputs = []
        for model in self.models:
            transformer_output, mlp_output = model(x)
            all_transformers_outputs.append(transformer_output)
            all_mlp_outputs.append(mlp_output)

        avg_transformer_output = torch.mean(torch.stack(all_transformers_outputs), dim=0)
        avg_mlp_output = torch.mean(torch.stack(all_mlp_outputs), dim=0)

        return avg_transformer_output, avg_mlp_output
    
    def update_hyperparameters(self, beta, gamma):
        self.beta = beta
        self.gamma = gamma

    def centroid_calculate(self, sequences):
        transformer_outputs_list = []
        for sequence in sequences:
            transformer_output, _ = self.forward(sequence)
            transformer_outputs_list.append(transformer_output)
        transformer_outputs_stacked = torch.stack(transformer_outputs_list)
        centroid = torch.mean(transformer_outputs_stacked, dim=0)
        return centroid

    def training_step(self, batch, batch_idx):
        original_sequence = batch["original_sequence"]
        original_label = batch["original_label"]

        out_original_transformer, out_anchor_mlp = self.forward(original_sequence)

        positive_centroid = self.centroid_calculate(batch["positive_sequences"])
        negative_centroid_1 = self.centroid_calculate(batch["negative_sequence_1s"])
        negative_centroid_2 = self.centroid_calculate(batch["negative_sequence_2s"])
        negative_centroid_3 = self.centroid_calculate(batch["negative_sequence_3s"])

        loss_reg = self.criterion_reg(out_anchor_mlp, original_label.unsqueeze(dim=1))
        loss_trip_1 = self.criterion_triplet[0](out_original_transformer, positive_centroid, negative_centroid_1)
        loss_trip_2 = self.criterion_triplet[1](out_original_transformer, positive_centroid, negative_centroid_2)
        loss_trip_3 = self.criterion_triplet[2](out_original_transformer, positive_centroid, negative_centroid_3)
        loss = self.config.train.beta * loss_reg + self.config.train.gamma * (loss_trip_1 + loss_trip_2 + loss_trip_3)
        # loss = self.beta * loss_reg + self.gamma * (loss_trip_1 + loss_trip_2 + loss_trip_3)

        self.log("train_loss_total", loss, prog_bar=True, logger=True)
        self.log("train_loss_reg", loss_reg, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        original_sequence = batch["original_sequence"]
        original_label = batch["original_label"]
        out_original_transformer, out_anchor_mlp = self.forward(original_sequence, original_label)
        loss_reg = self.criterion_reg(out_anchor_mlp, original_label.unsqueeze(dim=1))
        loss = loss_reg

        self.log("validation_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        original_sequence = batch["original_sequence"]
        original_label = batch["original_label"]
        out_original_transformer, out_anchor_mlp = self.forward(original_sequence, original_label)
        loss_reg = self.criterion_reg(out_anchor_mlp, original_label.unsqueeze(dim=1))
        loss = loss_reg

        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        adam_optimizer = optim.AdamW(self.parameters(), lr=self.config.train.lr)

        return [adam_optimizer]
