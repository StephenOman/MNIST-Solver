# Loss function

import numpy as np

from abc import ABC

class Base(ABC):

    def __init__(self) -> None:
        super().__init__()

    def one_hot(self, indexes, num_categories) -> np.array:
        y = np.zeros((indexes.shape[0], num_categories))
        for i in range(y.shape[0]):
            y[i][indexes[i]] = 1
        return y

class Cross_Entropy(Base):

    def __init__(self) -> None:
        self.loss = 0

    def calc_loss(self, target, categories, estimate) -> None:
        self.loss = np.sum(-1 * np.transpose(self.one_hot(target, categories)) * np.log(estimate)) / estimate[0].shape[0]