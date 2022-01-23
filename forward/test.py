import openpyxl
import torch
import matplotlib.pyplot as plt
from net import Net

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')


def test(worksheet, name):
    l_predict_list = []
    a_predict_list = []
    b_predict_list = []
    l_actual_list = []
    a_actual_list = []
    b_actual_list = []
    X_list = []
    cnt = 0
    for row in list(worksheet.rows)[1:]:
        to_predict = torch.Tensor([[float(c.value) for c in row[0:4]]])
        # print(to_predict)
        print('Predict')
        predict = model(to_predict)[0]
        print(predict)
        print('Actual')
        print(torch.Tensor([float(c.value) for c in row[4:7]]))

        l_predict_list.append(predict[0].item())
        a_predict_list.append(predict[1].item())
        b_predict_list.append(predict[2].item())

        l_actual_list.append(row[4].value)
        a_actual_list.append(row[5].value)
        b_actual_list.append(row[6].value)

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

    fig.suptitle('LAB')
    plt.savefig('./%s-lab.png' % name)
    plt.show()


if __name__ == '__main__':
    model = Net()
    model.load_state_dict(torch.load('model.pl', map_location=device))

    model.eval()

    workbook = openpyxl.load_workbook('../data/data.xlsx')
    test(workbook.worksheets[0], 'Train')
    test(workbook.worksheets[1], 'Vaild')
