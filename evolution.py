import torch
import logging
import numpy as np
import torch.nn.functional as F
from torch import Tensor

from model import Model
from dataset import DataSet
from predicted import Predicted

from PyQt6.QtCore import QObject, pyqtSignal

device = torch.device("cpu")

POP_SIZE = DataSet().universal.X.shape[0]  # population size
CROSS_RATE = 0.5  # mating probability (DNA crossover)
MUTATION_RATE = 0.01  # mutation probability
N_GENERATIONS = 1000
DNA_SIZE = DataSet().universal.X.shape[1]
I_BOUND = [29, 45]  # 电流取值范围
SPEED_BOUND = [700, 2301]  # 打标速度取值范围
Q_F_BOUND = [10, 23]  # Q频取值范围
Q_S_BOUND = [5, 46]  # Q释放取值范围

logger = logging.getLogger('GA')


class GA(QObject):
    progress_changed = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.LAB = None
        self.DNA_size = DNA_SIZE
        self.DNA_bound_I = I_BOUND
        self.DNA_bound_speed = SPEED_BOUND
        self.DNA_bound_qf = Q_F_BOUND
        self.DNA_bound_qs = Q_S_BOUND
        self.cross_rate = CROSS_RATE
        self.mutate_rate = MUTATION_RATE
        self.pop_size = POP_SIZE
        self.pop = DataSet().universal.X

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
                    child[point] = np.random.randint(*self.DNA_bound_I)
                elif point == 1:
                    child[point] = np.random.randint(*self.DNA_bound_speed)
                elif point == 2:
                    child[point] = np.random.randint(*self.DNA_bound_qf)
                else:
                    child[point] = np.random.randint(*self.DNA_bound_qs)
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
            self.progress_changed.emit(generation * 100 // N_GENERATIONS)
            fitness = self.get_fitness(self.F(self.pop))
            best_DNA: Tensor = self.pop[np.argmax(fitness)].unsqueeze(0)
            loss_func = F.mse_loss
            predictions = self.F(best_DNA)[0]
            train_loss = loss_func(predictions, self.LAB, reduction='sum').to(device)

            logger.debug("Gen %s: %s. Predict: %.4f, %.4f, %.4f. Loss: %s", generation, best_DNA[0],
                         predictions[0].item(), predictions[1].item(), predictions[2].item(), train_loss.item())

            count = count + 1 if prev_loss == 0 or prev_loss == train_loss.item() else 0
            prev_loss = train_loss.item()
            if count == 20 or generation == N_GENERATIONS - 1:  # 重复20次中断循环
                self.progress_changed.emit(100)
                best_DNA = best_DNA[0]
                logger.debug("Best result: %s, predict:%s, target: %s", best_DNA, predictions, self.LAB)

                return Predicted(
                    current=best_DNA[0].item(), speed=best_DNA[1].item(), frequency=best_DNA[2].item(),
                    release=round(best_DNA[3].item()), L=round(predictions[0].item(), 2),
                    A=round(predictions[1].item(), 2), B=round(predictions[2].item(), 2), loss=train_loss.item()
                )

            self.evolve()
