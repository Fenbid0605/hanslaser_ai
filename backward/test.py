import openpyxl
import torch
from matplotlib import pyplot as plt
from dataset import DataSet
from net import Net

POINT_SIZE = 4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(f'device: {device}')


def test(_model,  to_predict, actual, name):
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

    # to_predict = torch.Tensor([[float(c.value) for c in row[4:7]] for row in list(worksheet.rows)[1:]]).to(device)
    # predicts = _model(to_predict).to(device)
    predicts = _model(to_predict).to(device)
    actual = actual.to(device)

    for row in actual:
        print('Predict')
        predict = predicts[cnt]
        print(predict)
        print('Actual')
        print(row)

        i_predict_list.append(predict[0].item())
        speed_predict_list.append(predict[1].item())
        Qpin_predict_list.append(predict[2].item())
        QshiFang_predict_list.append(predict[3].item())

        i_actual_list.append(row[0].item())
        speed_actual_list.append(row[1].item())
        Qpin_actual_list.append(row[2].item())
        QshiFang_actual_list.append(row[3].item())

        X_list.append(cnt)
        cnt += 1
        print('=========')

    # 绘图
    fig, axs = plt.subplots(1, 4, figsize=(24, 5))

    axs[0].plot(X_list, i_actual_list, c='royalblue', label='i_actual')
    axs[0].plot(X_list, i_predict_list, c='darkorange', label='i_predict')
    # axs[0].scatter(X_list, i_actual_list, c='dodgerblue', s=POINT_SIZE, label='i_actual')
    # axs[0].scatter(X_list, i_predict_list, c='darkorange', s=POINT_SIZE, label='i_predict')
    axs[0].legend()

    axs[1].plot(X_list, speed_actual_list, c='royalblue', label='speed_actual')
    axs[1].plot(X_list, speed_predict_list, c='darkorange', label='speed_predict')
    # axs[1].scatter(X_list, speed_actual_list, c='royalblue', s=POINT_SIZE, label='speed_actual')
    # axs[1].scatter(X_list, speed_predict_list, c='darkorange', s=POINT_SIZE, label='speed_predict')
    axs[1].legend()

    axs[2].plot(X_list, Qpin_actual_list, c='royalblue', label='Qpin_actual')
    axs[2].plot(X_list, Qpin_predict_list, c='darkorange', label='Qpin_predict')
    # axs[2].scatter(X_list, Qpin_actual_list, c='royalblue', s=POINT_SIZE, label='Qpin_actual')
    # axs[2].scatter(X_list, Qpin_predict_list, c='darkorange', s=POINT_SIZE, label='Qpin_predict')
    axs[2].legend()

    axs[3].plot(X_list, QshiFang_actual_list, c='royalblue', label='QshiFang_actual')
    axs[3].plot(X_list, QshiFang_predict_list, c='darkorange', label='QshiFang_predict')
    # axs[3].scatter(X_list, QshiFang_actual_list, c='royalblue', s=POINT_SIZE, label='QshiFang_actual')
    # axs[3].scatter(X_list, QshiFang_predict_list, c='darkorange', s=POINT_SIZE, label='QshiFang_predict')
    axs[3].legend()

    fig.suptitle(f'./result/{name} - I_speed_Qpin_QshiFang')
    plt.savefig('./result/%s.png' % name)
    plt.show()


if __name__ == '__main__':
    model = Net()
    model.load_state_dict(torch.load('model.pl', map_location=device))

    model.eval()
    model.to(device)

    # workbook = openpyxl.load_workbook('../data/data.xlsx')
    # test(model, workbook.worksheets[0], 'Train')
    # test(model, workbook.worksheets[1], 'Valid')
    dataSet = DataSet()
    test(model, dataSet.x_matrix.to(device), dataSet.y_matrix.to(device), 'Train')
    test(model, dataSet.vx_matrix.to(device), dataSet.vy_matrix.to(device), 'Valid')