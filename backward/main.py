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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')


def train(_model, dataSet):
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
        train_loss_list.append(train_loss.item())

        prediction = net(v_x).to(device)
        net.eval()
        valid_loss = loss_func(prediction, v_y).to(device)
        valid_loss_list.append(valid_loss.item())
        net.train()

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        x_list.append(i)
        if i % 9999 == 0:
            print(f"EPOCH: {i} ,train_loss: {train_loss.item()} , valid_loss: {valid_loss.item()}")

    # 绘图
    fig, ax = plt.subplots()

    ax.plot(x_list, train_loss_list, label='train_loss')
    ax.plot(x_list, valid_loss_list, label='valid_loss')
    ax.legend()

    loss = train_loss_list[len(train_loss_list) - 4:]
    print(f'loss:{loss}')
    config.save_config(loss=loss) #记录参数日志

    plt.savefig('./result/loss.png')
    fig.suptitle('Loss')
    plt.show()


if __name__ == '__main__':
    model = Net()
    # print(model)

    try:
        model.load_state_dict(torch.load('model.pl', map_location=device))
        model.train()
    except:
        pass

    dataSet = DataSet()

    train(model, dataSet)

    torch.save(model.state_dict(), 'model.pl')

    model.eval()

    workbook = openpyxl.load_workbook('../data/data.xlsx')
    test(model, workbook.worksheets[0], 'Train')
    test(model, workbook.worksheets[1], 'Valid')
