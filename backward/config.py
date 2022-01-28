# Hyper parameters
import torch
import datetime
N_INPUT = 3
N_HIDDEN_LAYER = 12
N_HIDDEN = 20
N_OUTPUT = 4
ACTIVATION = torch.relu
B_INIT = -0.2  # use a bad bias constant initializer
EPOCH = 1e5
LR = 1e-4

EPOCH = int(EPOCH)


def save_config(loss):
    with open('./result/config_log.txt', "a+") as f:  # 只需要将之前的”w"改为“a"即可，代表追加内容
        f.write(f"N_HIDDEN: {N_HIDDEN} \n")
        f.write(f"N_HIDDEN_LAYER: {N_HIDDEN_LAYER} \n")
        f.write(f"EPOCH: {EPOCH} \n")
        f.write(f"LR: {LR} \n")
        f.write(f"LOSS: {loss} \n")
        f.write(f"time: {datetime.datetime.now()} \n\n")

