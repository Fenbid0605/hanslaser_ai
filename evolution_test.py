import platform

import openpyxl
import os

import torch
from rich.progress import Progress

import config
from dataset import DataSet
from multiprocessing import Pool, Manager
from evolution import GA
from result import Result
from share_data import ShareData
import random


def gen_data():
    random.seed(1003)
    y = []
    for _ in range(243):
        y.append([random.uniform(68.0, 90.0), random.uniform(-1, 0), random.uniform(-1, 1)])

    return torch.Tensor(y)


def use_excel_data():
    workbook = openpyxl.load_workbook(os.path.join(config.ABSPATH, 'data_test.xlsx'))
    # 数据集
    worksheet = workbook.worksheets[0]
    rows = list(worksheet.rows)[1:]
    y = []
    for row in rows:
        y.append([row[1].value, row[2].value, row[3].value])
    return torch.Tensor(y)


def test_each(inputY, __result: Result):
    target_lab = torch.Tensor(inputY)  # 目标LAB值
    ga = GA()
    predict = ga.predict(target_lab)
    __result.add_plot(predict, inputY)


def test(name, dataSetY: torch.Tensor):
    c = config.Config()
    with Manager() as manager:
        share = ShareData(manager)
        pool = Pool(c.EVOLUTION_MAX_PROC)
        result = Result(name=name, share=share)
        with Progress() as progress:
            task = progress.add_task(name, total=dataSetY.shape[0])

            for y in dataSetY:
                pool.apply_async(test_each, args=(y, result,), callback=lambda x: progress.advance(task))
            pool.close()
            pool.join()

        result.save()
        result.save_excel()


if __name__ == '__main__':

    if platform.system().lower() != 'windows':
        torch.multiprocessing.set_start_method('forkserver', force=True)
    dataset = DataSet()
    # test('evolution-random', gen_data())
    test('evolution-excel', use_excel_data())
    # test('evolution-universal', dataset.universal.Y)
    # test('evolution-standby', dataset.standby.Y)
    # test('evolution-train', dataset.train.Y)
    # test('evolution-valid', dataset.valid.Y)
