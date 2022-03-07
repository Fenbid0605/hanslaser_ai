# coding-utf8
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from dataset import DataSet_backward
from net_backward import Net
from rich.progress import track
from config import LR
import config

import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')


def train(_model, y_matrix, vy_matrix, CONFIG, net_name="NET_?"):
    # 打印参数，便于终端看见
    config.print_config_net(CONFIG)

    x = dataSet.x_matrix.to(device)
    y = y_matrix.to(device)

    v_x = dataSet.vx_matrix.to(device)
    v_y = vy_matrix.to(device)

    net = _model.to(device)

    loss_func = F.mse_loss
    optimizer = torch.optim.SGD(net.parameters(), lr=LR)

    x_list = []
    train_loss_list = []
    valid_loss_list = []

    for i in track(range(int(CONFIG.EPOCH))):
        # 打标速度
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
        if i % 1000 == 0:
            print(f"{net_name}, EPOCH: {i} ,train_loss: {train_loss.item()} , valid_loss: {valid_loss.item()}")

    # 绘图
    fig, ax = plt.subplots()

    ax.plot(x_list, train_loss_list, label='train_loss')
    ax.plot(x_list, valid_loss_list, label='valid_loss')

    ax.legend()

    # 保存配置
    config.save_config_net(train_loss_list[len(train_loss_list) - 4:], config.NET1(), f"{net_name}")  # 记录参数日志

    fig.suptitle('Loss')
    plt.savefig(f'./result/loss_{net_name}.png')
    plt.show()


if __name__ == '__main__':
    model_1 = Net(config.NET1())  # 打标速度
    model_2 = Net(config.NET2())  # a,q频，q释放
    dataSet = DataSet_backward()

    # 这里写的太冗余了。。。有时间改动只根据输入就可以跑，不if else了。。
    if len(sys.argv) >= 3 and sys.argv[1] == 'model_1':
        if sys.argv[2] == 'carry':
            print(f"{sys.argv[1]} carry model~")
            try:
                model_1.load_state_dict(torch.load('model_1.pl', map_location=device))
                model_1.train()
                train(model_1, dataSet.y_matrix_1, dataSet.vy_matrix_1, config.NET1(), "NET1")
                torch.save(model_1.state_dict(), 'model_1.pl')
            except:
                print('load model fail!')
            pass
        else:
            print("new model1")
            model_1.train()
            train(model_1, dataSet.y_matrix_1, dataSet.vy_matrix_1, config.NET1(), "NET1")
            torch.save(model_1.state_dict(), 'model_1.pl')
    elif len(sys.argv) >= 3 and sys.argv[1] == 'model_2':
        if sys.argv[2] == 'carry':
            print(f"{sys.argv[1]} carry model~")
            try:
                model_2.load_state_dict(torch.load('model_2.pl', map_location=device))
                model_2.train()
                train(model_2, dataSet.y_matrix_2, dataSet.vy_matrix_2, config.NET2(), "NET2")
                torch.save(model_2.state_dict(), 'model_2.pl')
            except:
                print('load model fail!')
            pass
        else:
            print("new model2")
            model_2.train()
            train(model_2, dataSet.y_matrix_2, dataSet.vy_matrix_2, config.NET2(), "NET2")
            torch.save(model_2.state_dict(), 'model_2.pl')
    else:
        print("输入格式错误哈, python3 main_net.py [model_1/_2] [carry/new]")
