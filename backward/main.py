# coding-utf8
import os

import openpyxl
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from dataset import DataSet
from net import Net
from rich.progress import track
from config import LR, EPOCH
import config
from test import test

import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')


def train(_model, dataSet):
    # 打印参数，便于终端看见
    config.print_config()

    x = dataSet.x_matrix.to(device)
    y = dataSet.y_matrix.to(device)

    v_x = dataSet.vx_matrix.to(device)
    v_y = dataSet.vy_matrix.to(device)

    net = _model.to(device)

    loss_func = F.mse_loss
    optimizer = torch.optim.SGD(net.parameters(), lr=LR)

    x_list = []
    train_loss_list = []
    valid_loss_list = []

    for i in track(range(EPOCH)):
        prediction = net(x).to(device)
        train_loss = loss_func(prediction, y).to(device)

        net.eval()
        prediction = net(v_x).to(device)
        valid_loss = loss_func(prediction, v_y).to(device)
        net.train()

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        train_loss_list.append(train_loss.item())
        valid_loss_list.append(valid_loss.item())
        x_list.append(i)
        if i % 100 == 0:
            print(f"EPOCH: {i} ,train_loss: {train_loss.item()} , valid_loss: {valid_loss.item()}")

        if i / 1000 == 5 and len(sys.argv) == 2 and sys.argv[1] == 'test':
            config.save_config(train_loss_list[len(train_loss_list) - 4:], True)  # 记录参数日志
            sys.exit(0)

    # 绘图
    fig, ax = plt.subplots()

    ax.plot(x_list, train_loss_list, label='train_loss')
    ax.plot(x_list, valid_loss_list, label='valid_loss')
    ax.legend()

    # 保存配置
    config.save_config(train_loss_list[len(train_loss_list) - 4:])  # 记录参数日志

    fig.suptitle('Loss')
    plt.savefig('./result/loss.png')
    plt.show()


if __name__ == '__main__':
    model = Net()

    if len(sys.argv) == 2 and sys.argv[1] == 'carry':
        print(f"{sys.argv[1]} model~")
        try:
            model.load_state_dict(torch.load('model.pl', map_location=device))
            model.train()
        except:
            print('load model fail!')
            pass
    else:
        print("new model~")

    dataSet = DataSet()

    train(model, dataSet)

    torch.save(model.state_dict(), 'model.pl')

    model.eval()

    workbook = openpyxl.load_workbook('../data/data.xlsx')
    test(model, dataSet.x_matrix.to(device), dataSet.y_matrix.to(device), 'Train')
    test(model, dataSet.vx_matrix.to(device), dataSet.vy_matrix.to(device), 'Valid')
