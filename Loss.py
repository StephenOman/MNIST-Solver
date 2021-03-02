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

    def calc_loss(self, labels, categories, estimate) -> None:
        one_hot = self.one_hot(labels, categories)
        losses = np.sum(-1 * np.transpose(one_hot) * np.log(estimate),axis=0)
        #losses = -1 * np.transpose(one_hot) * np.log(estimate)
        self.loss = np.average(losses)