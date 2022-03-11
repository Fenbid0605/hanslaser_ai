# coding-utf8
import torch
import torch.utils.data as Data  # ff
import torch.nn.functional as F
from matplotlib import pyplot as plt

from dataset import DataSet
from net import Net
from rich.progress import track
from config import Config
from test import Test

config = Config()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(_model, data_set):
    x = data_set.train.X
    y = data_set.train.Y

    v_x = data_set.valid.X.to(device)
    v_y = data_set.valid.Y.to(device)

    # ff  批数据处理
    torch_dataset = Data.TensorDataset(x, y)
    loader = Data.DataLoader(dataset=torch_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)

    net = _model.to(device)

    # 损失函数
    loss_func = F.mse_loss
    # 优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=config.LR)
    # 固定步长衰减
    # step_lr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.STEP, gamma=config.GAMMA)
    # LR 指数衰减
    # exp_lr = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.EXP_GAMMA)

    x_list = []
    train_loss_list = []
    valid_loss_list = []

    for i in track(range(config.EPOCH)):
        # step-批次
        for step, (b_x, b_y) in enumerate(loader):
            prediction = net(b_x.to(device)).to(device)
            train_loss = loss_func(prediction, b_y.to(device)).to(device)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        net.eval()
        prediction = net(x.to(device)).to(device)
        train_loss = loss_func(prediction, y.to(device)).to(device)

        prediction = net(v_x).to(device)
        valid_loss = loss_func(prediction, v_y).to(device)
        net.train()

        train_loss_list.append(train_loss.item())
        valid_loss_list.append(valid_loss.item())
        x_list.append(i)

        # if step_lr.get_last_lr()[0] > 2e-2:
        #     step_lr.step()

        # if i % 10 == 0:
        print(f"EPOCH: {i}, train_loss: {train_loss.item()}, "
              f"valid_loss: {valid_loss.item()}")

    # 绘图
    fig, ax = plt.subplots()

    ax.plot(x_list, train_loss_list, label='train_loss')
    ax.plot(x_list, valid_loss_list, label='valid_loss')
    ax.legend()

    fig.suptitle('Loss')
    plt.savefig('../result/loss.png')
    plt.show()

    # 保存配置日志，并记录最后一次训练集和测试集的 loss
    config.save(train_loss_list[-1], valid_loss_list[-1])


if __name__ == '__main__':
    print(f'device: {device}')
    model = Net()

    dataSet = DataSet()

    # 训练
    train(model, dataSet)

    # 保存模型
    torch.save(model.state_dict(), 'model.pl')

    # 测试模型
    Test()
