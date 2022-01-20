import openpyxl
import torch
from net import Net

if __name__ == '__main__':
    model = Net()
    model.load_state_dict(torch.load('model.pl'))

    model.eval()

    workbook = openpyxl.load_workbook('../data/data.xlsx')
    worksheet = workbook.worksheets[0]
    x_matrix = []
    y_matrix = []
    for row in list(worksheet.rows)[1:]:
        to_predict = torch.Tensor([[float(c.value) for c in row[4:7]]])
        print(to_predict)
        print('Predict')
        print(model(to_predict))
        print('Actual')
        print(torch.Tensor([float(c.value) for c in row[0:4]]))
        print('=========')
