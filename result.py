import os.path
from multiprocessing import Lock

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import config
from share_data import ShareData
from predicted import Predicted

lock = Lock()


class Result:
    def __init__(self, name, share: ShareData = None):
        self.name = name
        self.l_predict_list = []
        self.a_predict_list = []
        self.b_predict_list = []
        self.l_actual_list = []
        self.a_actual_list = []
        self.b_actual_list = []
        self.x_list = []
        self.cnt = 0
        self.predicts = []
        if share:
            self.l_predict_list = share.l_predict_list
            self.a_predict_list = share.a_predict_list
            self.b_predict_list = share.b_predict_list
            self.l_actual_list = share.l_actual_list
            self.a_actual_list = share.a_actual_list
            self.b_actual_list = share.b_actual_list
            self.x_list = share.x_list
            self.predicts = share.predicts

    def add_plot(self, predict: Predicted, actual):
        lock.acquire()
        self.l_predict_list.append(predict.L)
        self.a_predict_list.append(predict.A)
        self.b_predict_list.append(predict.B)

        self.l_actual_list.append(actual[0])
        self.a_actual_list.append(actual[1])
        self.b_actual_list.append(actual[2])

        cnt = len(self.x_list) + 1
        self.x_list.append(cnt)

        self.predicts.append([
            predict.speed, predict.current, predict.frequency, predict.release, predict.loss,
            predict.L, predict.A, predict.B,
            actual[0], actual[1], actual[2]
        ])

        lock.release()

    def get_cnt(self):
        return len(self.x_list) + 1

    def __save_figure(self, name, flag):
        # 绘图
        fig, axs = plt.subplots(1, 3, figsize=(9, 3))

        axs[0].plot(self.x_list, self.l_actual_list, label='L_actual')
        axs[1].plot(self.x_list, self.a_actual_list, label='A_actual')
        axs[2].plot(self.x_list, self.b_actual_list, label='B_actual')

        if flag:
            axs[1].plot(self.x_list, self.a_predict_list, label='A_predict')
            axs[0].plot(self.x_list, self.l_predict_list, label='L_predict')
            axs[2].plot(self.x_list, self.b_predict_list, label='B_predict')

        axs[0].legend()
        axs[1].legend()
        axs[2].legend()

        fig.suptitle('%s-LAB' % name)
        plt.savefig(os.path.join(config.ABSPATH, 'result/%s-lab.png' % name))
        plt.show()

    def save(self):
        # 预测和实际重叠
        self.__save_figure(self.name, True)
        # 仅实际
        self.__save_figure(self.name + '-actual', False)

    def save_excel(self):
        frame = pd.DataFrame(np.array(self.predicts),
                             index=self.x_list,
                             columns=['速度', '电流', 'Q频', 'Q释放', 'Loss',
                                      '预测 L', 'A', 'B', '目标 L', 'A', 'B'])
        print(frame)
        frame.to_excel(self.name + '.xlsx')
