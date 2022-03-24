import os.path
from multiprocessing import Lock

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
        if share:
            self.l_predict_list = share.l_predict_list
            self.a_predict_list = share.a_predict_list
            self.b_predict_list = share.b_predict_list
            self.l_actual_list = share.l_actual_list
            self.a_actual_list = share.a_actual_list
            self.b_actual_list = share.b_actual_list
            self.x_list = share.x_list

    def add_plot(self, predict: Predicted, actual):
        lock.acquire()
        self.l_predict_list.append(predict.L)
        self.a_predict_list.append(predict.A)
        self.b_predict_list.append(predict.B)

        self.l_actual_list.append(actual[0] * 100)
        self.a_actual_list.append(actual[1])
        self.b_actual_list.append(actual[2] * 10)

        cnt = len(self.x_list) + 1
        self.x_list.append(cnt)
        lock.release()

    def save(self):
        # 绘图
        fig, axs = plt.subplots(1, 3, figsize=(9, 3))

        axs[0].plot(self.x_list, self.l_actual_list, label='L_actual')
        axs[0].plot(self.x_list, self.l_predict_list, label='L_predict')
        axs[0].legend()

        axs[1].plot(self.x_list, self.a_actual_list, label='A_actual')
        axs[1].plot(self.x_list, self.a_predict_list, label='A_predict')
        axs[1].legend()

        axs[2].plot(self.x_list, self.b_actual_list, label='B_actual')
        axs[2].plot(self.x_list, self.b_predict_list, label='B_predict')
        axs[2].legend()

        fig.suptitle('%s-LAB' % self.name)
        plt.savefig(os.path.join(config.ABSPATH, 'result/%s-lab.png' % self.name))
        plt.show()
