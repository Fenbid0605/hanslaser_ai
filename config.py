import datetime
import os.path

import torch

ABSPATH = os.path.split(os.path.realpath(__file__))[0]


class Config:
    # Hyper parameters
    N_INPUT = 4
    N_HIDDEN_LAYER = 16
    N_HIDDEN = 82
    N_OUTPUT = 3
    ACTIVATION = torch.tanh
    B_INIT = -0.2  # use a bad bias constant initializer
    EPOCH = 1e4
    LR = 1e-1
    BATCH_SIZE = 128
    STEP = 50
    GAMMA = 0.1
    DROPOUT = 0.2
    EVOLUTION_MAX_PROC = 10

    def __init__(self):
        self.EPOCH = int(self.EPOCH)

    def save(self, train_loss, valid_loss):
        with open(os.path.join(ABSPATH, 'result/config_log.txt'), 'a+') as f:
            f.write(f'Time: {datetime.datetime.now()} \n')
            f.write(f"N_HIDDEN_LAYER: {self.N_HIDDEN_LAYER} \n")
            f.write(f"N_HIDDEN: {self.N_HIDDEN} \n")
            f.write(f"EPOCH: {self.EPOCH} \n")
            f.write(f"LR: {self.LR} \n")
            f.write(f"BATCH_SIZE: {self.BATCH_SIZE}\n")
            f.write(f"STEP:{self.STEP}\n")
            f.write(f"GAMMA:{self.GAMMA}\n")
            f.write(f"DROPOUT:{self.DROPOUT}\n")
            f.write(f"First Train LOSS: {train_loss[0]} \n")
            f.write(f"First Valid LOSS: {valid_loss[0]} \n")
            f.write(f"Last Train LOSS: {train_loss[-1]} \n")
            f.write(f"Last Valid LOSS: {valid_loss[-1]} \n\n")
