# data.py

import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class UJIIndoorLocDataset(Dataset):
    def __init__(self, csv_file, transform=None, is_train=True):
        """
        参数：
            csv_file (str): 数据 csv 文件路径
            transform (callable, 可选): 对样本的变换
            is_train (bool): 如果为 True，则认为数据中包含目标坐标（经度、纬度、楼层）
        """
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.is_train = is_train

        # 尝试以 "WAP" 开头的列作为 RSSI 列（总数应为 520 个）
        self.rssi_columns = [col for col in self.data.columns if col.startswith('WAP')]
        if len(self.rssi_columns) == 0:
            # 若未找到，则默认前 520 列为 RSSI 数据
            self.rssi_columns = self.data.columns[:520].tolist()

        # 数据预处理：
        # 1. 将 -100 替换为 -105（缺失信号处理）
        self.data[self.rssi_columns] = self.data[self.rssi_columns].replace(-100, -105)
        # 2. 归一化：假设有效 RSSI 分布在 [-105, 0]，使用 Min-Max 标准化
        self.data[self.rssi_columns] = (self.data[self.rssi_columns] + 105) / 105

        if self.is_train:
            # 寻找目标坐标列：需要包含经度（x）、纬度（y）以及楼层信息
            possible_x = ['x', 'X', 'longitude', 'LONGITUDE']
            possible_y = ['y', 'Y', 'latitude', 'LATITUDE']
            possible_floor = ['floor', 'FLOOR']
            self.target_columns = []
            for col in possible_x:
                if col in self.data.columns:
                    self.target_columns.append(col)
                    break
            for col in possible_y:
                if col in self.data.columns:
                    self.target_columns.append(col)
                    break
            for col in possible_floor:
                if col in self.data.columns:
                    self.target_columns.append(col)
                    break
            if len(self.target_columns) != 3:
                raise ValueError("未在数据中找到合适的目标坐标列，需要包含经度、纬度和楼层")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        # 获取 RSSI 向量，形状: (520, )
        rssi = sample[self.rssi_columns].values.astype(np.float32)
        if self.transform:
            rssi = self.transform(rssi)
        if self.is_train:
            # 获取目标坐标（经度、纬度、楼层）
            target = sample[self.target_columns].values.astype(np.float32)
            return rssi, target
        else:
            return rssi
