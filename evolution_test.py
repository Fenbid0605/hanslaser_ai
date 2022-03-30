import logging

import torch
from rich.progress import Progress

from config import Config
from dataset import DataSet
from multiprocessing import Pool, Manager
from evolution import GA
from result import Result
from share_data import ShareData


def test(inputY, __result: Result):
    target_lab = torch.Tensor(inputY)  # 目标LAB值
    ga = GA()
    predict = ga.predict(target_lab)
    __result.add_plot(predict, inputY)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('forkserver', force=True)
    config = Config()
    # 多线程执行
    with Manager() as manager:
        share = ShareData(manager)
        pool = Pool(config.EVOLUTION_MAX_PROC)
        result = Result(name='evolution', share=share)
        with Progress() as progress:
            task = progress.add_task('Evolution', total=DataSet().standby.Y.shape[0])

            for y in DataSet().standby.Y:
                pool.apply_async(test, args=(y, result,), callback=lambda x: progress.advance(task))
            pool.close()
            pool.join()

        result.save()
