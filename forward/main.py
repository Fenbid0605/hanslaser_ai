# coding-utf8
import torch
import torch.utils.data as Data  # ff
import torch.nn.functional as F
from dataset import DataSet
from net import Net
from rich.progress import track


def train(_model, dataSet):
    x = dataSet.x_matrix
    y = dataSet.y_matrix

    # ff  批数据处理
    # torch_dataset = Data.TensorDataset(x, y)
    # loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    net = _model
    print(net)
    loss_func = F.mse_loss
    optimizer = torch.optim.SGD(net.parameters(), lr=0.0001)

    for _ in track(range(10000)):
        # for step, (b_x, b_y) in enumerate(loader):  # step-批次
        prediction = net(x)
        loss = loss_func(prediction, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    model = Net()

    dataSet = DataSet()

    train(model, dataSet)

    torch.save(model.state_dict(), 'model.pl')
