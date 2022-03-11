import datetime
import torch


class Config:
    # Hyper parameters
    N_INPUT = 4
    N_HIDDEN_LAYER = 24
    N_HIDDEN = 120
    N_OUTPUT = 3
    ACTIVATION = torch.tanh
    B_INIT = -0.2  # use a bad bias constant initializer
    EPOCH = 1e4
    LR = 1e-3
    BATCH_SIZE = 64
    STEP = 300
    GAMMA = 0.1
    DROPOUT = 0.02

    def __init__(self):
        self.EPOCH = int(self.EPOCH)

    def save(self, train_loss, valid_loss):
        with open('../result/config_log.txt', "a+") as f:
            f.write(f"Time: {datetime.datetime.now()} \n")
            f.write(f"N_HIDDEN_LAYER: {self.N_HIDDEN_LAYER} \n")
            f.write(f"N_HIDDEN: {self.N_HIDDEN} \n")
            f.write(f"EPOCH: {self.EPOCH} \n")
            f.write(f"LR: {self.LR} \n")
            f.write(f"Train LOSS: {train_loss} \n")
            f.write(f"Valid LOSS: {valid_loss} \n\n")
