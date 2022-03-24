# coding-utf8
import torch
from model import Model

from dataset import DataSet
from config import Config
from net_test import Test

config = Config()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    print(f'device: {device}')

    # 训练集
    dataSet = DataSet()

    # 初始化网络
    model = Model()

    # 训练
    model.train(dataSet)

    # 保存模型
    model.save()

    # 测试模型
    Test()
