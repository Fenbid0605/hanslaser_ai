import torch
import openpyxl
import torch.nn.functional as F
import random


class DataSet:
    def __init__(self):
        workbook = openpyxl.load_workbook('../data/data.xlsx')
        # 训练集
        worksheet = workbook.worksheets[0]
        x_matrix = []
        y_matrix = []

        y_matrix_1 = []  # 电流，打标速度
        y_matrix_2 = []  # Q频、Q释放

        # 测试集
        # worksheet = workbook.worksheets[1]
        vx_matrix = []
        vy_matrix = []

        vy_matrix_1 = []  # 电流，打标速度
        vy_matrix_2 = []  # Q频、Q释放

        random.seed(10)
        for row in list(worksheet.rows)[1:]:
            # 随机生成测试集
            random_number = random.randint(1, 100)
            if random_number > 10:
                x_matrix.append([float(c.value) for c in row[4:7]])
                y_matrix.append([float(c.value) for c in row[0:4]])
                y_matrix_1.append([float(c.value) for c in row[0:2]])
                y_matrix_2.append([float(c.value) for c in row[2:4]])
            else:
                vx_matrix.append([float(c.value) for c in row[4:7]])
                vy_matrix.append([float(c.value) for c in row[0:4]])
                vy_matrix_1.append([float(c.value) for c in row[0:2]])
                vy_matrix_2.append([float(c.value) for c in row[2:4]])

        self.x_matrix = torch.Tensor(x_matrix)
        self.y_matrix = torch.Tensor(y_matrix)
        self.vx_matrix = torch.Tensor(vx_matrix)
        self.vy_matrix = torch.Tensor(vy_matrix)

        self.y_matrix_1 = torch.Tensor(y_matrix_1)
        self.y_matrix_2 = torch.Tensor(y_matrix_2)

        self.vy_matrix_1 = torch.Tensor(vy_matrix_1)
        self.vy_matrix_2 = torch.Tensor(vy_matrix_2)


