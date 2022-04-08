import logging

import torch
from rich.progress import Progress

from config import Config
from dataset import DataSet
from multiprocessing import Pool, Manager
from evolution import GA
from result import Result
from share_data import ShareData


def test_each(inputY, __result: Result):
    target_lab = torch.Tensor(inputY)  # 目标LAB值
    ga = GA()
    predict = ga.predict(target_lab)
    __result.add_plot(predict, inputY)


def test(name, dataSetY: torch.Tensor):
    config = Config()
    with Manager() as manager:
        share = ShareData(manager)
        pool = Pool(config.EVOLUTION_MAX_PROC)
        result = Result(name=name, share=share)
        with Progress() as progress:
            task = progress.add_task(name, total=dataSetY.shape[0])

            for y in DataSet().standby.Y:
                pool.apply_async(test_each, args=(y, result,), callback=lambda x: progress.advance(task))
            pool.close()
            pool.join()

        result.save()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('forkserver', force=True)
    dataset = DataSet()
    test('evolution-standby', dataset.standby.Y)
    test('evolution-train', dataset.train.Y)
    test('evolution-valid', dataset.valid.Y)
