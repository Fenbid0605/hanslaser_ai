import torch
import torch.nn.functional as F

from dataset import DataSet
from model import Model
from result import Result
from predicted import Predicted

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test(_model, to_predict, actual, name):
    cnt = 0

    predicts = _model(to_predict).to(device)
    actual = actual.to(device).cpu()

    result = Result(name=name)

    for row in actual:
        predict = predicts[cnt]
        loss = F.mse_loss(predict, row.to(device), reduction='sum')
        result.add_plot(predict=Predicted(
            speed=to_predict[cnt][0].item(),
            current=to_predict[cnt][1].item(),
            frequency=to_predict[cnt][2].item(),
            release=to_predict[cnt][3].item(),
            loss=loss.item(),
            L=predict[0].item(),
            A=predict[1].item(),
            B=predict[2].item()),
            actual=row)
        cnt += 1

    # 绘图
    result.save()
    # 保存 Execl
    result.save_excel()


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
    test(model.net, dataSet.standby.X.to(device), dataSet.standby.Y.to(device), 'Standby')
    test(model.net, dataSet.universal.X.to(device), dataSet.universal.Y.to(device), 'Universal')


if __name__ == '__main__':
    Test()
