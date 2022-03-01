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
from test import test_two_net

import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')


def train(_model_1, _model_2, dataSet):
    # 打印参数，便于终端看见
    config.print_config()

    x = dataSet.x_matrix.to(device)
    y_1 = dataSet.y_matrix_1.to(device)
    y_2 = dataSet.y_matrix_2.to(device)

    v_x = dataSet.vx_matrix.to(device)
    vy_1 = dataSet.vy_matrix_1.to(device)
    vy_2 = dataSet.vy_matrix_2.to(device)

    net_1 = _model_1.to(device)
    net_2 = _model_2.to(device)

    loss_func_1 = F.mse_loss
    optimizer_1 = torch.optim.SGD(net_1.parameters(), lr=LR)

    loss_func_2 = F.mse_loss
    optimizer_2 = torch.optim.SGD(net_2.parameters(), lr=LR)

    x_list = []
    train_loss_list_1 = []
    valid_loss_list_1 = []

    train_loss_list_2 = []
    valid_loss_list_2 = []

    for i in track(range(EPOCH)):
        prediction_1 = net_1(x).to(device)
        train_loss_1 = loss_func_1(prediction_1, y_1).to(device)

        net_1.eval()
        prediction_1 = net_1(v_x).to(device)
        valid_loss_1 = loss_func_1(prediction_1, vy_1).to(device)
        net_1.train()

        optimizer_1.zero_grad()
        train_loss_1.backward()
        optimizer_1.step()

        train_loss_list_1.append(train_loss_1.item())
        valid_loss_list_1.append(valid_loss_1.item())

        # Q频、Q释放
        prediction_2 = net_2(x).to(device)
        train_loss_2 = loss_func_2(prediction_2, y_2).to(device)

        net_2.eval()
        prediction_2 = net_2(v_x).to(device)
        valid_loss_2 = loss_func_2(prediction_2, vy_2).to(device)
        net_2.train()

        optimizer_2.zero_grad()
        train_loss_2.backward()
        optimizer_2.step()

        train_loss_list_2.append(train_loss_2.item())
        valid_loss_list_2.append(valid_loss_2.item())

        x_list.append(i)
        if i % 1000 == 0:
            print(f"EPOCH: {i} ,train_loss_1: {train_loss_1.item()} , valid_loss_1: {valid_loss_1.item()}")
            print(f"EPOCH: {i} ,train_loss_2: {train_loss_2.item()} , valid_loss_2: {valid_loss_2.item()}")

    # 绘图
    fig, ax = plt.subplots()

    ax.plot(x_list, train_loss_list_1, label='train_loss_1')
    ax.plot(x_list, valid_loss_list_1, label='valid_loss_1')

    ax.plot(x_list, train_loss_list_2, label='train_loss_2')
    ax.plot(x_list, valid_loss_list_2, label='valid_loss_2')

    ax.legend()

    # 保存配置
    config.save_config(train_loss_list_1[len(train_loss_list_1) - 4:], is_two_net=True)  # 记录参数日志

    fig.suptitle('Loss_')
    plt.savefig('./result/loss_.png')
    plt.show()


def train_2(_model_2, dataSet):
    # 打印参数，便于终端看见
    config.print_config()

    x = dataSet.x_matrix.to(device)
    y_2 = dataSet.y_matrix_2.to(device)

    v_x = dataSet.vx_matrix.to(device)
    vy_2 = dataSet.vy_matrix_2.to(device)

    net_2 = _model_2.to(device)

    loss_func_2 = F.mse_loss
    optimizer_2 = torch.optim.SGD(net_2.parameters(), lr=LR)

    x_list = []

    train_loss_list_2 = []
    valid_loss_list_2 = []

    for i in track(range(EPOCH)):
        # Q频、Q释放
        prediction_2 = net_2(x).to(device)
        train_loss_2 = loss_func_2(prediction_2, y_2).to(device)

        net_2.eval()
        prediction_2 = net_2(v_x).to(device)
        valid_loss_2 = loss_func_2(prediction_2, vy_2).to(device)
        net_2.train()

        optimizer_2.zero_grad()
        train_loss_2.backward()
        optimizer_2.step()

        train_loss_list_2.append(train_loss_2.item())
        valid_loss_list_2.append(valid_loss_2.item())

        x_list.append(i)
        if i % 1000 == 0:
            print(f"EPOCH: {i} ,train_loss_2: {train_loss_2.item()} , valid_loss_2: {valid_loss_2.item()}")

    # 绘图
    fig, ax = plt.subplots()

    ax.plot(x_list, train_loss_list_2, label='train_loss_2')
    ax.plot(x_list, valid_loss_list_2, label='valid_loss_2')

    ax.legend()

    # 保存配置
    config.save_config(train_loss_list_2[len(train_loss_list_2) - 4:], is_two_net=True)  # 记录参数日志

    fig.suptitle('Loss_2')
    plt.savefig('./result/loss_2.png')
    plt.show()


if __name__ == '__main__':
    model_1 = Net()
    model_2 = Net()

    if len(sys.argv) == 2 and sys.argv[1] == 'carry':
        print(f"{sys.argv[1]} model~")
        try:
            model_1.load_state_dict(torch.load('model_1.pl', map_location=device))
            model_2.load_state_dict(torch.load('model_2.pl', map_location=device))
            model_1.train()
            model_2.train()
        except:
            print('load model fail!')
            pass
    else:
        print("new model~")

    dataSet = DataSet()

    train(model_1, model_2, dataSet)
    # train_2(model_2, dataSet)

    torch.save(model_1.state_dict(), 'model_1.pl')
    torch.save(model_2.state_dict(), 'model_2.pl')

    model_1.eval()
    model_2.eval()

    workbook = openpyxl.load_workbook('../data/data.xlsx')
    test_two_net(model_1, model_2, dataSet.x_matrix.to(device), dataSet.y_matrix.to(device), 'Train')
    test_two_net(model_1, model_2, dataSet.vx_matrix.to(device), dataSet.vy_matrix.to(device), 'Valid')
