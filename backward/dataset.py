import torch
import openpyxl
import random


class DataSet:
    def __init__(self):
        workbook = openpyxl.load_workbook('../data/data.xlsx')
        # 训练集
        worksheet = workbook.worksheets[0]
        x_matrix = []
        y_matrix = []
        # 测试集
        vx_matrix = []
        vy_matrix = []
        random.seed(10)
        rows = list(worksheet.rows)[1:]
        # 打乱数据
        random.shuffle(rows)
        for row in rows:
            # 随机生成测试集
            random_number = random.randint(1, 100)
            x = [float(c.value) for c in row[4:7]]
            y = [float(c.value) for c in row[0:4]]
            y[1] /= 1000
            if random_number > 10:
                x_matrix.append(x)
                y_matrix.append(y)
            else:
                vx_matrix.append(x)
                vy_matrix.append(y)

        self.x_matrix = torch.Tensor(x_matrix)
        self.y_matrix = torch.Tensor(y_matrix)
        self.vx_matrix = torch.Tensor(vx_matrix)
        self.vy_matrix = torch.Tensor(vy_matrix)


# 仅适用backward
class DataSet_backward:
    def __init__(self):
        workbook = openpyxl.load_workbook('../data/data_for_two_net.xlsx')  # 修改了打标速度的列位置，为0
        # 训练集
        worksheet = workbook.worksheets[0]
        x_matrix = []
        y_matrix = []

        y_matrix_1 = []  # 打标速度
        y_matrix_2 = []  # 电流、Q频、Q释放

        # 测试集
        # worksheet = workbook.worksheets[1]
        vx_matrix = []
        vy_matrix = []

        vy_matrix_1 = []  # 打标速度
        vy_matrix_2 = []  # 电流、Q频、Q释放

        random.seed(10)
        for row in list(worksheet.rows)[1:]:
            # 随机生成测试集
            random_number = random.randint(1, 100)
            if random_number > 10:
                x_matrix.append([float(c.value) for c in row[4:7]])
                y_matrix.append([float(c.value) for c in row[0:4]])

                y_matrix_1.append([float(c.value) for c in row[0:1]])
                y_matrix_2.append([float(c.value) for c in row[1:4]])
            else:
                vx_matrix.append([float(c.value) for c in row[4:7]])
                vy_matrix.append([float(c.value) for c in row[0:4]])

                vy_matrix_1.append([float(c.value) for c in row[0:1]])
                vy_matrix_2.append([float(c.value) for c in row[1:4]])

        self.x_matrix = torch.Tensor(x_matrix)
        self.y_matrix = torch.Tensor(y_matrix)
        self.vx_matrix = torch.Tensor(vx_matrix)
        self.vy_matrix = torch.Tensor(vy_matrix)

        self.y_matrix_1 = torch.Tensor(y_matrix_1)
        self.y_matrix_2 = torch.Tensor(y_matrix_2)
        self.vy_matrix_1 = torch.Tensor(vy_matrix_1)
        self.vy_matrix_2 = torch.Tensor(vy_matrix_2)
