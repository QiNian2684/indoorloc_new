# model.py

import torch
import torch.nn as nn


class IndoorLocalizationModel(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(IndoorLocalizationModel, self).__init__()
        # 共享嵌入层：将每个 RSSI 标量映射到 64 维向量
        self.embedding = nn.Linear(1, 64)

        # 分支 A：1D CNN 提取局部特征
        self.conv1 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(128)

        # 分支 B：Transformer 提取全局交互特征
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            dropout=dropout_rate,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc_transformer = nn.Linear(64, 128)

        # 特征融合层
        self.fusion_fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # 回归头：将融合特征映射到三维输出 (x, y, floor)
        self.regressor = nn.Linear(128, 3)

    def forward(self, x):
        # 输入 x: [batch_size, 520]（归一化后的 RSSI 向量）
        # 扩展最后一维为 [batch_size, 520, 1]
        x = x.unsqueeze(-1)
        x = self.embedding(x)  # 形状：[batch, 520, 64]
        x = torch.relu(x)

        # 分支 A：1D CNN
        # 1D卷积要求输入形状为 [batch, channels, length]，因此需要转换维度
        x_a = x.permute(0, 2, 1)  # [batch, 64, 520]
        x_a = self.conv1(x_a)
        x_a = self.bn1(x_a)
        x_a = torch.relu(x_a)
        x_a = self.conv2(x_a)
        x_a = self.bn2(x_a)
        x_a = torch.relu(x_a)
        # 在序列维度上做全局平均池化，输出形状：[batch, 128]
        x_a = x_a.mean(dim=2)

        # 分支 B：Transformer
        # Transformer 接受的输入形状为 [seq_len, batch, embed_dim]
        x_b = x.permute(1, 0, 2)  # [520, batch, 64]
        x_b = self.transformer_encoder(x_b)
        x_b = x_b.permute(1, 0, 2)  # 转回 [batch, 520, 64]
        # 全局平均池化：在序列维度上取均值
        x_b = x_b.mean(dim=1)  # [batch, 64]
        x_b = self.fc_transformer(x_b)  # [batch, 128]
        x_b = torch.relu(x_b)

        # 特征融合：拼接两个分支输出
        x_fusion = torch.cat([x_a, x_b], dim=1)  # [batch, 256]
        x_fusion = self.fusion_fc(x_fusion)  # [batch, 128]

        # 回归输出 (x, y, floor)
        output = self.regressor(x_fusion)  # [batch, 3]
        return output
