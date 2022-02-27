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
        # worksheet = workbook.worksheets[1]
        vx_matrix = []
        vy_matrix = []
        random.seed(10)
        for row in list(worksheet.rows)[1:]:
            # 随机生成测试集
            random_number = random.randint(1, 100)
            if random_number > 10:
                x_matrix.append([float(c.value) for c in row[0:4]])
                y_matrix.append([float(c.value) for c in row[4:7]])
            else:
                vx_matrix.append([float(c.value) for c in row[0:4]])
                vy_matrix.append([float(c.value) for c in row[4:7]])

        self.x_matrix = torch.Tensor(x_matrix)
        self.y_matrix = torch.Tensor(y_matrix)
        self.vx_matrix = torch.Tensor(vx_matrix)
        self.vy_matrix = torch.Tensor(vy_matrix)



if __name__ == "__main__":
    print(DataSet().x_matrix)
    print(DataSet().y_matrix)