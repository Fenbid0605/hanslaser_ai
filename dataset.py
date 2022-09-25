import os

import torch
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
        workbook = openpyxl.load_workbook(os.path.join(config.ABSPATH, 'random_data_1.xlsx'))
        # 数据集
        worksheet = workbook.worksheets[0]

        # 训练集 2/5
        self.train = Matrix()
        # 验证集 2/5
        self.valid = Matrix()
        # 备用集 1/5
        self.standby = Matrix()

        self.universal = Matrix()

        # 初始一个随机种子
        random.seed(1003)
        rows = list(worksheet.rows)[1:]
        # 打乱输入
        random.shuffle(rows)

        for row in rows:
            # 随机生成测试集
            random_number = random.randint(1, 100)
            x = [float(c.value) for c in row[0:4]]
            # x[1] /= 100
            y = [float(c.value) for c in row[4:7]]
            self.universal.rows.append(row[0:7])
            # 排除边缘数据
            if y[0] > 90 or y[0] < 57:
                continue

            self.universal.x.append(x)
            self.universal.y.append(y)

            if 10 < random_number < 80:
                self.train.x.append(x)
                self.train.y.append(y)
            elif random_number > 90:
                self.standby.x.append(x)
                self.standby.y.append(y)
            else:
                self.valid.x.append(x)
                self.valid.y.append(y)


if __name__ == '__main__':
    dataSet = DataSet()
