# 次要，仅适用于两个模型

import torch
from config import B_INIT


# 参数以传入CONFIG
class Net(torch.nn.Module):
    def __init__(self, CONFIG):
        self.n_hidden_layer = CONFIG.N_HIDDEN_LAYER
        self.ACTIVATION = CONFIG.ACTIVATION
        # 初始网络的内部结构
        super(Net, self).__init__()
        self.fcs = []
        self.bns = []
        self.drops = []
        self.bn_input = torch.nn.BatchNorm1d(CONFIG.N_INPUT, momentum=0.5)
        for i in range(0, CONFIG.N_HIDDEN_LAYER):
            input_size = CONFIG.N_INPUT if i == 0 else CONFIG.N_HIDDEN
            fc = torch.nn.Linear(input_size, CONFIG.N_HIDDEN)
            setattr(self, 'fc%i' % i, fc)
            self._set_init(fc)
            self.fcs.append(fc)

            bn = torch.nn.BatchNorm1d(CONFIG.N_HIDDEN, momentum=0.5)
            setattr(self, 'bn%i' % i, bn)
            self.bns.append(bn)

            dropout = torch.nn.Dropout(0.01)
            setattr(self, 'dropout%i' % i, dropout)
            self.drops.append(dropout)

        self.predict = torch.nn.Linear(CONFIG.N_HIDDEN, CONFIG.N_OUTPUT)
        self._set_init(self.predict)

    @staticmethod
    def _set_init(layer):
        torch.nn.init.normal_(layer.weight, mean=0, std=.1)
        torch.nn.init.constant_(layer.bias, B_INIT)

    def forward(self, x):
        # 一次正向行走过程
        x = self.bn_input(x)

        for i in range(self.n_hidden_layer):
            x = self.fcs[i](x)
            x = self.drops[i](x)
            x = self.bns[i](x)
            x = self.ACTIVATION(x)
        # output
        x = self.predict(x)
        return x
