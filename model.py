# model.py

import torch
import torch.nn as nn


class IndoorLocalizationModel(nn.Module):
    def __init__(
            self,
            embedding_dim=128,
            cnn_channels=(128, 256),
            transformer_layers=4,
            nhead=8,
            dim_feedforward=256,
            dropout_rate=0.5
    ):
        """
        更复杂的模型结构，允许多层 CNN + 可配置 Transformer。

        参数：
            embedding_dim (int): RSSI 标量的初始嵌入维度。
            cnn_channels (tuple/list): CNN 各层输出通道数，例如 (128, 256, 512) 等。
            transformer_layers (int): TransformerEncoder 的层数。
            nhead (int): Multi-head Attention 的头数。
            dim_feedforward (int): Transformer 前馈网络的维度。
            dropout_rate (float): Dropout 比例。
        """
        super(IndoorLocalizationModel, self).__init__()

        # 1) 将单个 RSSI 标量映射到 embedding_dim 维
        self.embedding = nn.Linear(1, embedding_dim)

        # 2) CNN 分支
        #    构建多层 CNN
        cnn_layers = []
        in_c = embedding_dim
        for out_c in cnn_channels:
            cnn_layers.append(nn.Conv1d(in_c, out_c, kernel_size=3, stride=1, padding=1))
            cnn_layers.append(nn.BatchNorm1d(out_c))
            cnn_layers.append(nn.ReLU())
            in_c = out_c
        self.cnn = nn.Sequential(*cnn_layers)

        # 3) Transformer 分支
        #    根据指定的层数、注意力头数和前馈维度来初始化
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # 映射 Transformer 输出到与 CNN 输出同样的通道数
        self.fc_transformer = nn.Linear(embedding_dim, in_c)

        # 4) 特征融合 (CNN + Transformer)
        self.fusion_fc = nn.Sequential(
            nn.Linear(in_c * 2, in_c),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # 5) 最终回归头 (x, y, floor)
        self.regressor = nn.Linear(in_c, 3)

    def forward(self, x):
        """
        输入 x: [batch_size, 520]（归一化后的 RSSI 向量）
        """
        # 先扩展最后一维，以便输入到 embedding
        x = x.unsqueeze(-1)  # [batch_size, 520, 1]
        x = self.embedding(x)  # [batch_size, 520, embedding_dim]
        x = torch.relu(x)

        # CNN 分支
        x_cnn = x.permute(0, 2, 1)  # [batch, embedding_dim, 520]
        x_cnn = self.cnn(x_cnn)  # [batch, out_c, 520]
        x_cnn = x_cnn.mean(dim=2)  # 全局平均池化 => [batch, out_c]

        # Transformer 分支
        x_trans = x.permute(1, 0, 2)  # [520, batch, embedding_dim]
        x_trans = self.transformer_encoder(x_trans)  # [520, batch, embedding_dim]
        x_trans = x_trans.permute(1, 0, 2)  # [batch, 520, embedding_dim]
        x_trans = x_trans.mean(dim=1)  # 全局平均池化 => [batch, embedding_dim]
        x_trans = self.fc_transformer(x_trans)  # => [batch, out_c]
        x_trans = torch.relu(x_trans)

        # 特征融合
        x_fusion = torch.cat([x_cnn, x_trans], dim=1)  # [batch, out_c * 2]
        x_fusion = self.fusion_fc(x_fusion)  # [batch, out_c]

        # 最终三维回归
        output = self.regressor(x_fusion)  # [batch, 3]
        return output
