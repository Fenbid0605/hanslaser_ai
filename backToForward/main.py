import torch
from matplotlib import pyplot as plt

from backward.net import Net as back_net
from backward.dataset import DataSet
from forward.net import Net as front_net
import openpyxl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')


def test(_back_model, _front_model, dataSet):
    workbook = openpyxl.load_workbook('../data/data.xlsx')
    worksheet = workbook.worksheets[0]
    l_predict_list = []
    a_predict_list = []
    b_predict_list = []
    l_actual_list = []
    a_actual_list = []
    b_actual_list = []
    X_list = []
    cnt = 0

    # 反向得出预测的四个值
    back_net = _back_model.to(device)
    input_lab_x = dataSet.x_matrix.to(device)  # 初始输入lab
    prediction_x = back_net(input_lab_x).to(device)  # 反向预测得出的四个值，作为前向的输入
    print(f"prediction_x: {prediction_x}")

    # 正向
    front_net = _front_model.to(device)
    for i, row in enumerate(list(worksheet.rows)[1:]):
        to_predict = prediction_x[i].cpu().detach().numpy()
        to_predict = torch.tensor([to_predict]).to(device)
        predict = front_net(to_predict)[0]

        l_predict_list.append(predict[0].item())
        a_predict_list.append(predict[1].item())
        b_predict_list.append(predict[2].item())

        l_actual_list.append(row[4].value)
        a_actual_list.append(row[5].value)
        b_actual_list.append(row[6].value)

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

    fig.suptitle('LAB')
    plt.show()


if __name__ == "__main__":
    back_model = back_net()
    back_model.load_state_dict(torch.load('../backward/model.pl'))
    back_model.eval()

    front_model = front_net()
    front_model.load_state_dict(torch.load('../forward/model.pl'))
    front_model.eval()

    dataSet = DataSet()
    test(back_model, front_model, dataSet)
