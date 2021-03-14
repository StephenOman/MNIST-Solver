import Network
import Layers
import Loss
import MNIST_Data
import Config
import pickle
import numpy as np

from matplotlib import pyplot as plt

class MNIST_Solver:

    def __init__(self, data: MNIST_Data.MNIST_Data, config: Config.Config) -> None:

        self.nn = Network.Neural_Network()

        self.nn.add_layer(Layers.LeakyReLU(784, 512, config.bias, learn_rate=config.learn_rate))
        self.nn.add_layer(Layers.Softmax(512, 10, config.bias, learn_rate=config.learn_rate))
        self.nn.add_loss(Loss.Cross_Entropy())

        self.categories = 10

        self.data = data

        self.flat_train_data = Layers.Flatten.flatten(data.train_images, 60000, 28, 28)
        self.flat_test_data = Layers.Flatten.flatten(data.test_images, 10000, 28, 28)

    def save(self, model_file = "mnist.model"):
        self.nn.clear_data()
        with open(model_file, 'wb') as f:
            pickle.dump(self.nn, f)

    def load(self, model_file = "mnist.model"):
        with open(model_file, 'rb') as f:
            self.nn = pickle.load(f)

    def train(self, config: Config.Config) -> None:
        self.nn.train(self.flat_train_data, self.data.train_labels, self.categories, config)

    def print_img(self, image: int) -> None:
        # Test a particular value from the test set

        test_img = np.array([self.flat_test_data[image]])
        test_label = np.array([self.data.test_labels[image]])

        result = self.nn.feedforward(test_img)

        pixels = test_img.reshape((28, 28))
        plt.imshow(pixels, cmap='gray_r')
        heading = "Image label is " + str(test_label[0]) + \
            " \nand\n the predicted label is " + str(np.argmax(result)) + "\n"
        plt.title(heading)
        plt.show()
        