import torch
import logging
import numpy as np
import torch.nn.functional as F
from torch import Tensor

from model import Model
from dataset import DataSet
from predicted import Predicted
from helper.log import log

device = torch.device("cpu")

data = DataSet()

POP_SIZE = data.valid.X.shape[0]  # population size
CROSS_RATE = 0.5  # mating probability (DNA crossover)
MUTATION_RATE = 0.01  # mutation probability
N_GENERATIONS = 1000
DNA_SIZE = data.valid.X.shape[1]
FREQ_BOUND = [10, 50]  # 频率取值范围
SPEED_BOUND = [5, 37]  # 打标速度取值范围 * 100
I_BOUND = [28, 34]  # 电流取值范围
GAP_BOUND = [2, 4]  # 填充间距取值范围 / 100

logger = log


class GA:
    def __init__(self):
        super().__init__()
        self.LAB = None
        self.DNA_size = DNA_SIZE
        self.DNA_bound_I = I_BOUND
        self.DNA_bound_speed = SPEED_BOUND
        self.DNA_bound_freq = FREQ_BOUND
        self.DNA_bound_gap = GAP_BOUND
        self.cross_rate = CROSS_RATE
        self.mutate_rate = MUTATION_RATE
        self.pop_size = POP_SIZE
        self.pop = data.valid.X

        self.model = Model(device=device)
        self.model.load()
        self.model.eval()

    def F(self, x):
        lab = self.model.net(x)
        return lab

    def get_fitness(self, preds):  # count how many character matches
        match_count = []
        for pred in preds:
            match_count.append(1 / F.mse_loss(pred, self.LAB).item())
        return match_count

    def select(self):
        _fitness = self.get_fitness(self.F(self.pop))  # add a small amount to avoid all zero fitness
        _fitness = np.array(_fitness)
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True, p=_fitness / _fitness.sum())
        return self.pop[idx]

    def crossover(self, parent, pop):
        if np.random.rand() < self.cross_rate:
            i_ = np.random.randint(0, self.pop_size, size=1)  # select another individual from pop
            cross_points = np.random.randint(0, 2, self.DNA_size).astype(bool)  # choose crossover points
            parent[cross_points] = pop[i_, cross_points]  # mating and produce one child
        return parent

    def mutate(self, child):
        for point in range(self.DNA_size):
            if np.random.rand() < self.mutate_rate:
                if point == 0:
                    child[point] = np.random.randint(*self.DNA_bound_freq)
                elif point == 1:
                    child[point] = np.random.randint(*self.DNA_bound_speed)
                elif point == 2:
                    child[point] = np.random.randint(*self.DNA_bound_I)
                else:
                    child[point] = np.random.randint(*self.DNA_bound_gap)
        return child

    def evolve(self):
        pop = self.select()
        pop_copy = pop
        for parent in pop:  # for every parent
            child = self.crossover(parent, pop_copy)
            child = self.mutate(child)
            parent[:] = child
        self.pop = pop

    def predict(self, lab: Tensor) -> Predicted:
        self.LAB = lab
        count = 0
        prev_loss = 0
        for generation in range(N_GENERATIONS):
            fitness = self.get_fitness(self.F(self.pop))
            best_DNA: Tensor = self.pop[np.argmax(fitness)].unsqueeze(0)
            loss_func = F.mse_loss
            predictions = self.F(best_DNA)[0]
            train_loss = loss_func(predictions, self.LAB, reduction='sum').to(device)

            # logger.info("Gen %s: %s. Predict: %.4f, %.4f, %.4f. Loss: %s" % (generation, best_DNA[0],
            #                                                                  predictions[0].item(),
            #                                                                  predictions[1].item(),
            #                                                                  predictions[2].item(), train_loss.item()))

            count = count + 1 if prev_loss == 0 or prev_loss == train_loss.item() else 0
            prev_loss = train_loss.item()
            if count == 20 or generation == N_GENERATIONS - 1:  # 重复20次中断循环
                best_DNA = best_DNA[0]
                # logger.info("Best result: %s, predict:%s, target: %s", best_DNA, predictions, self.LAB)

                return Predicted(
                    frequency=best_DNA[0].item(), speed=best_DNA[1].item() * 100, current=best_DNA[2].item(),
                    gap=best_DNA[3].item() / 100, L=round(predictions[0].item(), 2),
                    A=round(predictions[1].item(), 2), B=round(predictions[2].item(), 2), loss=train_loss.item()
                )

            self.evolve()
