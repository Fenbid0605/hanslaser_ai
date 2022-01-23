# coding-utf8
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from dataset import DataSet
from net import Net
from rich.progress import track

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
    optimizer = torch.optim.SGD(net.parameters(), lr=0.000004)
    x_list = []
    loss_list = []
    for i in track(range(50000)):
        # for step, (b_x, b_y) in enumerate(loader):  # step-批次
        prediction = net(x).to(device)
        loss = loss_func(prediction, y).to(device)
        loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
    try:
        model.load_state_dict(torch.load('model.pl', map_location=device))
    except:
        pass

    dataSet = DataSet()

    train(model, dataSet)

    torch.save(model.state_dict(), 'model.pl')
