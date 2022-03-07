import openpyxl
import torch
from matplotlib import pyplot as plt
from dataset import DataSet
from net import Net
from net_backward import Net as Net_bcakward
import config
import sys

POINT_SIZE = 4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test(_model, to_predict, actual, name):
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


# 两个模型的test
def test_two_net(_model_1, _model_2, to_predict, actual, name):
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

    predicts_1 = _model_1(to_predict).to(device)
    predicts_2 = _model_2(to_predict).to(device)

    actual = actual.to(device)

    for row in actual:
        print('Predict_1')
        predict_1 = predicts_1[cnt]
        predict_2 = predicts_2[cnt]
        print(predict_1, predict_2)
        print('Actual')
        print(row)

        speed_predict_list.append(predict_1[0].item())
        i_predict_list.append(predict_2[0].item())
        Qpin_predict_list.append(predict_2[1].item())
        QshiFang_predict_list.append(predict_2[2].item())

        speed_actual_list.append(row[0].item())
        i_actual_list.append(row[1].item())
        Qpin_actual_list.append(row[2].item())
        QshiFang_actual_list.append(row[3].item())

        X_list.append(cnt)
        cnt += 1
        print('=========')

    # 绘图
    fig, axs = plt.subplots(1, 4, figsize=(24, 5))

    axs[0].plot(X_list, i_actual_list, c='royalblue', label='i_actual')
    axs[0].plot(X_list, i_predict_list, c='darkorange', label='i_predict')
    axs[0].legend()

    axs[1].plot(X_list, speed_actual_list, c='royalblue', label='speed_actual')
    axs[1].plot(X_list, speed_predict_list, c='darkorange', label='speed_predict')
    axs[1].legend()

    axs[2].plot(X_list, Qpin_actual_list, c='royalblue', label='Qpin_actual')
    axs[2].plot(X_list, Qpin_predict_list, c='darkorange', label='Qpin_predict')
    axs[2].legend()

    axs[3].plot(X_list, QshiFang_actual_list, c='royalblue', label='QshiFang_actual')
    axs[3].plot(X_list, QshiFang_predict_list, c='darkorange', label='QshiFang_predict')
    axs[3].legend()

    fig.suptitle(f'{name} - I_speed_Qpin_QshiFang - TWO_NET')
    plt.savefig('./result/%s_twoNet.png' % name)
    plt.show()


if __name__ == '__main__':

    dataSet = DataSet()

    if len(sys.argv) == 2 and sys.argv[1] == 'two_model':
        print(f"test two model~")
        model_1 = Net_bcakward(config.NET1())  # 打标速度
        model_2 = Net_bcakward(config.NET2())  # a,q频，q释放
        try:
            model_1.load_state_dict(torch.load('model_1.pl', map_location=device))
            model_2.load_state_dict(torch.load('model_2.pl', map_location=device))
            model_1.eval()
            model_2.eval()
            model_1.to(device)
            model_2.to(device)
            test_two_net(model_1, model_2, dataSet.x_matrix.to(device), dataSet.y_matrix.to(device), 'Train')
            test_two_net(model_1, model_2, dataSet.vx_matrix.to(device), dataSet.vy_matrix.to(device), 'Valid')
        except:
            print('load model fail!')
            pass
    else:
        print("test model~")
        model = Net()
        model.load_state_dict(torch.load('model.pl', map_location=device))

        model.eval()
        model.to(device)

        test(model, dataSet.x_matrix.to(device), dataSet.y_matrix.to(device), 'Train')
        test(model, dataSet.vx_matrix.to(device), dataSet.vy_matrix.to(device), 'Valid')
