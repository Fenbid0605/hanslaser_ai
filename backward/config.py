# Hyper parameters
import torch
import datetime

N_INPUT = 3
N_HIDDEN_LAYER = 40
N_HIDDEN = 300
N_OUTPUT = 4
ACTIVATION = torch.sigmoid
B_INIT = -0.2  # use a bad bias constant initializer
EPOCH = 1e5
LR = 1e-6

EPOCH = int(EPOCH)


class NET1:
    def __init__(self):
        self.N_INPUT = 3
        self.N_HIDDEN_LAYER = 30
        self.N_HIDDEN = 120
        self.N_OUTPUT = 1


class NET2:
    def __init__(self):
        self.N_INPUT = 3
        self.N_HIDDEN_LAYER = 30
        self.N_HIDDEN = 120
        self.N_OUTPUT = 3


def print_config():
    print(f"N_HIDDEN_LAYER: {N_HIDDEN_LAYER}")
    print(f"N_HIDDEN: {N_HIDDEN}")
    print(f"ACTIVATION: {ACTIVATION}")
    print(f"EPOCH: {EPOCH} , LR:{LR}")


def save_config(loss, is_test=False, is_two_net=False):
    with open('./result/config_log.txt', "a+") as f:  # 只需要将之前的”w"改为“a"即可，代表追加内容
        if is_test:
            f.write("[ is_test , EPOCH TEST = 5000]\n")

        if is_test:
            f.write(f"TWO_NET")
        f.write(f"N_HIDDEN_LAYER: {N_HIDDEN_LAYER} \n")
        f.write(f"N_HIDDEN: {N_HIDDEN} \n")
        f.write(f"EPOCH: {EPOCH} \n")
        f.write(f"LR: {LR} \n")
        f.write(f"LOSS: {loss} \n")
        f.write(f"time: {datetime.datetime.now()} \n\n")


def print_config_(net1, net2):
    print(f"net1:")
    print(f"N_HIDDEN_LAYER: {net1.N_HIDDEN_LAYER}")
    print(f"N_HIDDEN: {net1.N_HIDDEN}")
    print("========")
    print(f"net2:")
    print(f"N_HIDDEN_LAYER: {net2.N_HIDDEN_LAYER}")
    print(f"N_HIDDEN: {net2.N_HIDDEN}")
    print("========")
    print(f"ACTIVATION: {ACTIVATION}")
    print(f"EPOCH: {EPOCH} , LR:{LR}")


def save_config_two_net(loss_1, loss_2, net1, net2, is_test=False):
    with open('./result/config_log.txt', "a+") as f:  # 只需要将之前的”w"改为“a"即可，代表追加内容
        if is_test:
            f.write("[ is_test , EPOCH TEST = 5000]\n")
        f.write(f"TWO NET \n")
        f.write(f"net1 \n")
        f.write(f"N_HIDDEN_LAYER: {net1.N_HIDDEN_LAYER} \n")
        f.write(f"N_HIDDEN: {net2.N_HIDDEN} \n")
        f.write(f"---\n")
        f.write(f"net2 \n")
        f.write(f"N_HIDDEN_LAYER: {net2.N_HIDDEN_LAYER} \n")
        f.write(f"N_HIDDEN: {net2.N_HIDDEN} \n")

        f.write(f"EPOCH: {EPOCH} \n")
        f.write(f"LR: {LR} \n")
        f.write(f"LOSS_1: {loss_1} \n")
        f.write(f"LOSS_2: {loss_2} \n")
        f.write(f"time: {datetime.datetime.now()} \n\n")
