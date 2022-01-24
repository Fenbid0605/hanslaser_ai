# Hyper parameters
import torch

N_INPUT = 4
N_HIDDEN_LAYER = 20
N_HIDDEN = 130
N_OUTPUT = 3
ACTIVATION = torch.tanh
B_INIT = -0.2  # use a bad bias constant initializer
EPOCH = 5e2
LR = 1e-2


EPOCH = int(EPOCH)
