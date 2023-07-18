import os

import torch
from torch.utils.data import Dataset
import openpyxl
import random

import config


class Matrix:
    def __init__(self):
        self.x = []
        self.y = []
        self.rows = []

    @property
    def X(self):
        return torch.Tensor(self.x)

    @property
    def Y(self):
        return torch.Tensor(self.y)


class DataSet:
    def __init__(self):
        workbook = openpyxl.load_workbook(os.path.join(config.ABSPATH, 'data/data20221109.xlsx'))
        # 数据集
        worksheet = workbook.worksheets[0]

        # 训练集 1/3
        self.train = Matrix()
        # 验证集 1/3
        self.valid = Matrix()
        # 备用集 1/3
        self.standby = Matrix()

        self.universal = Matrix()

        # 初始一个随机种子
        random.seed(1003)
        rows = list(worksheet.rows)[1:]
        # 打乱输入
        random.shuffle(rows)

        for row in rows:
            # 随机生成测试集
            random_number = random.randint(1, 3)
            x = [float(c.value) for c in row[1:5]]
            x[1] /= 100
            x[3] *= 100
            y = [float(c.value) for c in row[5:8]]
            self.universal.rows.append(row[1:8])
            # 排除边缘数据
            if y[0] > 90 or y[0] < 57:
                continue

            self.universal.x.append(x)
            self.universal.y.append(y)

            if random_number == 1:
                self.train.x.append(x)
                self.train.y.append(y)
                self.train.rows.append([*x, *y])
            elif random_number == 2:
                self.standby.x.append(x)
                self.standby.y.append(y)
            else:
                self.valid.x.append(x)
                self.valid.y.append(y)

        # 为数据集添加 GABVSG 数据
        # workbook = openpyxl.load_workbook(os.path.join(config.ABSPATH, 'data/data20230324.xlsx'))
        # worksheet = workbook.worksheets[0]
        # random.seed(1003)
        # rows = list(worksheet.rows)[1:]
        # rows = random.sample(rows, 188)
        # for row in rows:
        #     x = [float(c.value) for c in row[1:5]]
        #     x[1] /= 100
        #     x[3] *= 100
        #     y = [float(c.value) for c in row[5:8]]
        #
        #     self.train.x.append(x)
        #     self.train.y.append(y)
        #     self.train.rows.append([*x, *y])


class CustomDataset(Dataset):
    def __init__(self, x, y, augmentations=0, noise_std=0.1):
        self.augmentations = augmentations
        self.noise_std = noise_std

        # 生成虚拟样本
        virtual_x = []
        virtual_y = []
        for _ in range(augmentations):
            random_index = torch.randint(len(x), (1,)).item()
            noisy_x = x[random_index] + torch.randn_like(x[random_index]) * noise_std
            noisy_y = y[random_index] + torch.randn_like(y[random_index]) * noise_std
            virtual_x.append(noisy_x)
            virtual_y.append(noisy_y)

        # 将虚拟样本添加到原始数据中
        self.x = torch.cat([x, torch.stack(virtual_x)])
        self.y = torch.cat([y, torch.stack(virtual_y)])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


if __name__ == '__main__':
    dataSet = DataSet()
