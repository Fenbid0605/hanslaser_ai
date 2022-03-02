# Hyper parameters
import torch
import datetime

N_INPUT = 3
N_HIDDEN_LAYER = 40
N_HIDDEN = 300
N_OUTPUT = 4
ACTIVATION = torch.sigmoid
B_INIT = -0.2  # use a bad bias constant initializer
EPOCH = 0.02e5
LR = 1e-5

EPOCH = int(EPOCH)


class NET1:  # 打标速度
    def __init__(self):
        self.N_INPUT = 3
        self.N_HIDDEN_LAYER = 5
        self.N_HIDDEN = 6
        self.N_OUTPUT = 1
        self.EPOCH = 2e6


class NET2:  # a,q频，q释放
    def __init__(self):
        self.N_INPUT = 3
        self.N_HIDDEN_LAYER = 32
        self.N_HIDDEN = 94
        self.N_OUTPUT = 3
        self.EPOCH = 4e5


def print_config():
    print(f"N_HIDDEN_LAYER: {N_HIDDEN_LAYER}")
    print(f"N_HIDDEN: {N_HIDDEN}")
    print(f"ACTIVATION: {ACTIVATION}")
    print(f"EPOCH: {EPOCH} , LR:{LR}")


def save_config(loss, is_test=False, is_two_net=False):
    with open('./result/config_log.txt', "a+") as f:  # 只需要将之前的”w"改为“a"即可，代表追加内容
        if is_test:
            f.write("[ is_test , EPOCH TEST = 5000]\n")
        if is_two_net:
            f.write(f"TWO_NET")
        f.write(f"N_HIDDEN_LAYER: {N_HIDDEN_LAYER} \n")
        f.write(f"N_HIDDEN: {N_HIDDEN} \n")
        f.write(f"EPOCH: {EPOCH} \n")
        f.write(f"LR: {LR} \n")
        f.write(f"LOSS: {loss} \n")
        f.write(f"time: {datetime.datetime.now()} \n\n")


# 有空改写成单一的，这样太sb了

def print_config_net(net):
    print(f"N_HIDDEN_LAYER: {net.N_HIDDEN_LAYER}")
    print(f"N_HIDDEN: {net.N_HIDDEN}")
    print(f"EPOCH: {net.EPOCH}")
    print(f"ACTIVATION: {ACTIVATION}")
    print(f"LR:{LR}")
    print("===========")


def print_config_two_net(net1, net2):
    print(f"net1:")
    print(f"N_HIDDEN_LAYER: {net1.N_HIDDEN_LAYER}")
    print(f"N_HIDDEN: {net1.N_HIDDEN}")
    print(f"EPOCH: {net1.EPOCH}")
    print("========")
    print(f"net2:")
    print(f"N_HIDDEN_LAYER: {net2.N_HIDDEN_LAYER}")
    print(f"N_HIDDEN: {net2.N_HIDDEN}")
    print(f"EPOCH: {net2.EPOCH}")
    print("========")
    print(f"ACTIVATION: {ACTIVATION}")
    print(f" LR:{LR}")


def save_config_net(loss, net, net_name):
    with open('./result/config_log.txt', "a+") as f:  # 只需要将之前的”w"改为“a"即可，代表追加内容
        f.write(f"[ {net_name} ] \n")
        f.write(f"N_HIDDEN_LAYER: {net.N_HIDDEN_LAYER} \n")
        f.write(f"N_HIDDEN: {net.N_HIDDEN} \n")
        f.write(f"EPOCH: {net.EPOCH} \n")
        f.write(f"LR: {LR} \n")
        f.write(f"LOSS: {loss} \n")
        f.write(f"time: {datetime.datetime.now()} \n\n")


def save_config_two_net(loss_1, loss_2, net1, net2, is_test=False):
    with open('./result/config_log.txt', "a+") as f:  # 只需要将之前的”w"改为“a"即可，代表追加内容
        if is_test:
            f.write("[ is_test , EPOCH TEST = 5000]\n")
        f.write(f"【 TWO NET 】 \n")
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
