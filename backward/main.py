# coding-utf8
import os

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from dataset import DataSet
from net import Net
from rich.progress import track
from config import LR, EPOCH

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')


def train(_model, dataSet):
    x = dataSet.x_matrix.to(device)
    y = dataSet.y_matrix.to(device)

    # ff  批数据处理
    # torch_dataset = Data.TensorDataset(x, y)
    # loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    net = _model.to(device)

    loss_func = F.mse_loss
    optimizer = torch.optim.SGD(net.parameters(), lr=LR)
    # 固定步长衰减
    step_lr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.65)

    x_list = []
    loss_list = []
    for i in track(range(EPOCH)):
        # for step, (b_x, b_y) in enumerate(loader):  # step-批次
        prediction = net(x).to(device)
        loss = loss_func(prediction, y).to(device)
        loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        step_lr.step()

        x_list.append(i)

    # 绘图
    fig, ax = plt.subplots()

    ax.plot(x_list, loss_list, label='loss')
    ax.legend()
    plt.savefig('./loss.png')
    fig.suptitle('Loss')
    plt.show()


if __name__ == '__main__':
    model = Net()
    print(model)

    dataSet = DataSet()

    train(model, dataSet)

    torch.save(model.state_dict(), 'model.pl')
