import torch
import matplotlib.pyplot as plt

from dataset import DataSet
from net import Net

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test(_model, to_predict, actual, name):
    l_predict_list = []
    a_predict_list = []
    b_predict_list = []
    l_actual_list = []
    a_actual_list = []
    b_actual_list = []
    X_list = []
    cnt = 0

    predicts = _model(to_predict).to(device)
    actual = actual.to(device)
    for row in actual:
        predict = predicts[cnt]

        predict[0] *= 100
        predict[2] *= 10

        row[0] *= 100
        row[2] *= 10

        l_predict_list.append(predict[0].item())
        a_predict_list.append(predict[1].item())
        b_predict_list.append(predict[2].item())

        l_actual_list.append(row[0].item())
        a_actual_list.append(row[1].item())
        b_actual_list.append(row[2].item())

        X_list.append(cnt)
        cnt += 1

    # 绘图
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))

    axs[0].plot(X_list, l_actual_list, label='L_actual')
    axs[0].plot(X_list, l_predict_list, label='L_predict')
    axs[0].legend()

    axs[1].plot(X_list, a_actual_list, label='A_actual')
    axs[1].plot(X_list, a_predict_list, label='A_predict')
    axs[1].legend()

    axs[2].plot(X_list, b_actual_list, label='B_actual')
    axs[2].plot(X_list, b_predict_list, label='B_predict')
    axs[2].legend()

    fig.suptitle('%s-LAB' % name)
    plt.savefig('../result/%s-lab.png' % name)
    plt.show()


def Test():
    # 实例化神经网络
    model = Net()
    # 导入训练好的模型
    model.load_state_dict(torch.load('model.pl', map_location=device))
    # 冻结，进入预测模式
    model.eval()
    model.to(device)
    # 实例化数据集
    dataSet = DataSet()

    # 测试训练集
    test(model, dataSet.train.X.to(device), dataSet.train.Y.to(device), 'Train')
    # 测试验证集
    test(model, dataSet.valid.X.to(device), dataSet.valid.Y.to(device), 'Valid')


if __name__ == '__main__':
    Test()
