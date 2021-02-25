import numpy as np
from abc import ABC

class Flatten:
    def __init__(self, input_data, instance_count, rows, columns) -> None:
        self.data = input_data.reshape(instance_count, rows * columns)


class Base(ABC):
    def __init__(self, num_inputs, num_nodes, learn_rate = 0.1) -> None:
        self.num_inputs = num_inputs
        self.num_nodes = num_nodes
        self.weights = (2 * (np.random.random([self.num_inputs, self.num_nodes]))) - 1
        self.learn_rate = learn_rate

    def feedforward(self, input_data) -> None:
        self.outputs = np.dot(np.transpose(self.weights), input_data)

    def backprop(self, input_data, error, labels) -> None:
        pass #TODO


class LeakyReLU(Base):
    def __init__(self, num_inputs, num_nodes, epsilon = 0.1, learn_rate = 0.1) -> None:
        super().__init__(num_inputs, num_nodes, learn_rate)
        self.epsilon = epsilon

    def feedforward(self, input_data) -> None:
        super().feedforward(input_data)
        self.outputs[self.outputs<0] *= self.epsilon

    def backprop(self, input_data, error, labels) -> None:
        prime = np.copy(self.outputs)
        prime[prime >= 0] = 1
        prime[prime <0 ] = self.epsilon
        prime = prime * error

        dir_matrix = np.dot(input_data, np.transpose(prime))

        self.delta = dir_matrix / input_data.shape[1]

        self.learn_delta = self.delta * self.learn_rate

        # Error to be propagated to previous layers
        self.bp_error = np.dot(self.weights, prime)


class Softmax(Base):
    def __init__(self, num_inputs, num_nodes, learn_rate = 0.1) -> None:
        super().__init__(num_inputs, num_nodes, learn_rate)

    def feedforward(self, input_data) -> None:
        super().feedforward(input_data)
        ez = np.exp(self.outputs)
        sum_ez = np.sum(ez, axis = 0)
        self.outputs = np.divide(ez, sum_ez)

    def backprop(self, input_data, error, labels) -> None:
        prime = np.copy(self.outputs)
        for i in range(labels.shape[0]):
            prime[labels[i]][i] -= 1
        
        dir_matrix = np.dot(input_data, np.transpose(prime))

        # Change in this layer's weights
        self.delta = dir_matrix / input_data.shape[1]
        self.learn_delta = self.delta * self.learn_rate

        # Error to be propagated to previous layers
        self.bp_error = np.dot(self.weights, prime)