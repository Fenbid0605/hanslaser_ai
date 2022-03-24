import logging

import torch
from rich.progress import track

from config import Config
from dataset import DataSet
from multiprocessing import Pool, Manager
from evolution import GA
from result import Result
from share_data import ShareData


def test(inputY, __result: Result):
    print('Actual %s,%s,%s' % (inputY[0] * 100, inputY[1], inputY[2] * 10))
    target_lab = torch.Tensor(inputY)  # 目标LAB值
    ga = GA()
    predict = ga.predict(target_lab)
    print('Predict %s,%s,%s' % (predict.L, predict.A, predict.B))
    __result.add_plot(predict, inputY)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    config = Config()
    # 多线程执行
    with Manager() as manager:
        share = ShareData(manager)
        pool = Pool(config.EVOLUTION_MAX_PROC)
        result = Result(name='evolution', share=share)

        for y in track(DataSet().standby.Y):
            pool.apply_async(test, args=(y, result,))
        pool.close()
        pool.join()

        result.save()
