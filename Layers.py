import numpy as np
from abc import ABC

class Flatten:
    def flatten(input_data, instance_count, rows, columns) -> np.array:
        return input_data.reshape(instance_count, rows * columns)


class Base(ABC):
    def __init__(self, num_inputs, num_nodes, learn_rate = 0.1, bias=False) -> None:
        self.bias = bias
        if(self.bias):
            self.num_inputs = num_inputs + 1
        else:
            self.num_inputs = num_inputs
        self.num_nodes = num_nodes
        self.weights = (2 * (np.random.random([self.num_inputs, self.num_nodes]))) - 1
        self.learn_rate = learn_rate

    def feedforward(self, input_data) -> None:
        if(self.bias):
            bias_signals = np.ones((input_data.shape[1]))
            self.inputs = np.vstack((bias_signals, input_data))
        else:
            self.inputs = input_data
        self.outputs = np.dot(np.transpose(self.weights), self.inputs)

    def backprop(self, error) -> None:
        dir_matrix = np.dot(self.inputs, np.transpose(error))

        delta = dir_matrix / self.inputs.shape[1]

        learn_delta = delta * self.learn_rate

        # Error to be propagated to previous layers
        if(self.bias):
            self.bp_error = np.dot(self.weights, error) # TODO
        else:
            self.bp_error = np.dot(self.weights, error)

        self.weights = self.weights - learn_delta


class LeakyReLU(Base):
    def __init__(self, num_inputs, num_nodes, bias=False, epsilon = 0.1, learn_rate = 0.1) -> None:
        super().__init__(num_inputs, num_nodes, learn_rate, bias)
        self.epsilon = epsilon

    def feedforward(self, input_data) -> None:
        super().feedforward(input_data)
        self.outputs[self.outputs<0] *= self.epsilon

    def backprop(self, error, labels) -> None:
        prime = np.copy(self.outputs)
        prime[prime >= 0] = 1
        prime[prime <0 ] = self.epsilon
        prime = prime * error
        super().backprop(prime)


class Softmax(Base):
    def __init__(self, num_inputs, num_nodes, bias=False, learn_rate = 0.1) -> None:
        super().__init__(num_inputs, num_nodes, learn_rate, bias)

    def feedforward(self, input_data) -> None:
        super().feedforward(input_data)
        ez = np.exp(self.outputs)
        sum_ez = np.sum(ez, axis = 0)
        self.outputs = np.divide(ez, sum_ez)

    def backprop(self, error, labels) -> None:
        prime = np.copy(self.outputs)
        for i in range(labels.shape[0]):
            prime[labels[i]][i] -= 1
        
        super().backprop(prime)