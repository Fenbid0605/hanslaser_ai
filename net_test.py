import torch
import matplotlib.pyplot as plt

from dataset import DataSet
from model import Model
from result import Result
from predicted import Predicted

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test(_model, to_predict, actual, name):
    cnt = 0

    predicts = _model(to_predict).to(device)
    actual = actual.to(device)

    result = Result(name=name)

    for row in actual:
        predict = predicts[cnt]
        cnt += 1
        result.add_plot(predict=Predicted(L=predict[0].item() * 100,
                                          A=predict[1].item(),
                                          B=predict[2].item() * 10),
                        actual=row)

    # 绘图
    result.save()


def Test():
    # 实例化数据集
    dataSet = DataSet()
    # 实例化模型
    model = Model()
    # 导入训练好的模型
    model.load()
    # 冻结，进入预测模式
    model.eval()

    # 测试训练集
    test(model.net, dataSet.train.X.to(device), dataSet.train.Y.to(device), 'Train')
    # 测试验证集
    test(model.net, dataSet.valid.X.to(device), dataSet.valid.Y.to(device), 'Valid')


if __name__ == '__main__':
    Test()
