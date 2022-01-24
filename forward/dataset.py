import torch
import openpyxl
import torch.nn.functional as F


class DataSet:
    def __init__(self):
        workbook = openpyxl.load_workbook('../data/data.xlsx')
        # 训练集
        worksheet = workbook.worksheets[0]
        x_matrix = []
        y_matrix = []
        for row in list(worksheet.rows)[1:]:
            x_matrix.append([float(c.value) for c in row[0:4]])
            y_matrix.append([float(c.value) for c in row[4:7]])

        # 测试集
        worksheet = workbook.worksheets[1]
        vx_matrix = []
        vy_matrix = []
        for row in list(worksheet.rows)[1:]:
            vx_matrix.append([float(c.value) for c in row[0:4]])
            vy_matrix.append([float(c.value) for c in row[4:7]])

        self.x_matrix = torch.Tensor(x_matrix)
        self.y_matrix = torch.Tensor(y_matrix)
        self.vx_matrix = torch.Tensor(vx_matrix)
        self.vy_matrix = torch.Tensor(vy_matrix)

