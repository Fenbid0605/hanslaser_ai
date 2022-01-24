# Hyper parameters
import torch

N_INPUT = 4
N_HIDDEN_LAYER = 18
N_HIDDEN = 50
N_OUTPUT = 3
ACTIVATION = torch.tanh
B_INIT = -0.2  # use a bad bias constant initializer
EPOCH = int(3e5)
LR = 5e-7
