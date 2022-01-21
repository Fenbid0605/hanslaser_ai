import openpyxl
import torch
from matplotlib import pyplot as plt

from net import Net

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'device: {device}')
    model = Net()
    model.load_state_dict(torch.load('model.pl', map_location=device))

    model.eval()

    workbook = openpyxl.load_workbook('../data/data.xlsx')
    worksheet = workbook.worksheets[0]

    i_predict_list = []  # 电流
    speed_predict_list = []  # 打标速度
    Qpin_predict_list = []  # Q频
    QshiFang_predict_list = []  # Q释放

    i_actual_list = []  # 电流
    speed_actual_list = []  # 打标速度
    Qpin_actual_list = []  # Q频
    QshiFang_actual_list = []  # Q释放

    X_list = []
    cnt = 0

    for row in list(worksheet.rows)[1:]:
        to_predict = torch.Tensor([[float(c.value) for c in row[4:7]]])
        # print(to_predict)
        print('Predict')
        predict = model(to_predict)[0]
        print(predict)
        print('Actual')
        print(torch.Tensor([float(c.value) for c in row[0:4]]))

        i_predict_list.append(predict[0].item())
        speed_predict_list.append(predict[1].item())
        Qpin_predict_list.append(predict[2].item())
        QshiFang_predict_list.append(predict[3].item())

        i_actual_list.append(row[0].value)
        speed_actual_list.append(row[1].value)
        Qpin_actual_list.append(row[2].value)
        QshiFang_actual_list.append(row[3].value)

        X_list.append(cnt)
        cnt += 1
        print('=========')

    # 绘图
    fig, axs = plt.subplots(1, 4, figsize=(12, 3))

    axs[0].plot(X_list, i_actual_list, label='i_actual')
    axs[0].plot(X_list, i_predict_list, label='i_predict')
    axs[0].legend()

    axs[1].plot(X_list, speed_actual_list, label='speed_actual')
    axs[1].plot(X_list, speed_predict_list, label='speed_predict')
    axs[1].legend()

    axs[2].plot(X_list, Qpin_actual_list, label='Qpin_actual')
    axs[2].plot(X_list, Qpin_predict_list, label='Qpin_predict')
    axs[2].legend()

    axs[3].plot(X_list, QshiFang_actual_list, label='QshiFang_actual')
    axs[3].plot(X_list, QshiFang_predict_list, label='QshiFang_predict')
    axs[3].legend()

    fig.suptitle('A_speed_Qpin_QshiFang')
    plt.show()
