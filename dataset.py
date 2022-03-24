import os

import torch
import openpyxl
import random

import config


class Matrix:
    def __init__(self):
        self.x = []
        self.y = []

    @property
    def X(self):
        return torch.Tensor(self.x)

    @property
    def Y(self):
        return torch.Tensor(self.y)


class DataSet:
    def __init__(self):
        workbook = openpyxl.load_workbook(os.path.join(config.ABSPATH, 'data.xlsx'))
        # 数据集
        worksheet = workbook.worksheets[0]

        # 训练集 2/5
        self.train = Matrix()
        # 验证集 2/5
        self.valid = Matrix()
        # 备用集 1/5
        self.standby = Matrix()

        # 初始一个随机种子
        random.seed(1003)
        # 打乱输入
        rows = list(worksheet.rows)[1:]
        random.shuffle(rows)

        for row in rows:
            # 随机生成测试集
            random_number = random.randint(1, 100)
            x = [float(c.value) for c in row[0:4]]
            # x[1] /= 100
            y = [float(c.value) for c in row[4:7]]
            y[0] /= 100
            y[2] /= 10

            if 20 < random_number < 60:
                self.train.x.append(x)
                self.train.y.append(y)
            elif random_number > 80:
                self.standby.x.append(x)
                self.standby.y.append(y)
            else:
                self.valid.x.append(x)
                self.valid.y.append(y)


if __name__ == '__main__':
    dataSet = DataSet()
