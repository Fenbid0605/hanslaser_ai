import openpyxl
import torch
import matplotlib.pyplot as plt

from dataset import DataSet
from net import Net

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')


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
        print('Predict')
        predict = predicts[cnt]
        print(predict)
        print('Actual')
        print(row)

        l_predict_list.append(predict[0].item())
        a_predict_list.append(predict[1].item())
        b_predict_list.append(predict[2].item())

        l_actual_list.append(row[0].item())
        a_actual_list.append(row[1].item())
        b_actual_list.append(row[2].item())

        X_list.append(cnt)
        cnt += 1
        print('=========')

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
    plt.savefig('./%s-lab.png' % name)
    plt.show()


if __name__ == '__main__':
    model = Net()
    model.load_state_dict(torch.load('model.pl', map_location=device))

    model.eval()
    model.to(device)

    dataSet = DataSet()
    test(model, dataSet.x_matrix.to(device), dataSet.y_matrix.to(device), 'Train')
    test(model, dataSet.vx_matrix.to(device), dataSet.vy_matrix.to(device), 'Valid')
