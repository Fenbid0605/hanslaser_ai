# Hyper parameters
import datetime

import torch

N_INPUT = 4
N_HIDDEN_LAYER = 42
N_HIDDEN = 130
N_OUTPUT = 3
ACTIVATION = torch.tanh
B_INIT = -0.2  # use a bad bias constant initializer
EPOCH = 5e5
LR = 1e-5

EPOCH = int(EPOCH)


def save_config(loss):
    with open('./config_log.txt', "a+") as f:
        f.write(f"N_HIDDEN_LAYER: {N_HIDDEN_LAYER} \n")
        f.write(f"N_HIDDEN: {N_HIDDEN} \n")
        f.write(f"EPOCH: {EPOCH} \n")
        f.write(f"LR: {LR} \n")
        f.write(f"LOSS: {loss} \n")
        f.write(f"time: {datetime.datetime.now()} \n\n")
