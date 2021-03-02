import Network
import Layers
import Loss
import MNIST_Data

import numpy as np

from matplotlib import pyplot as plt

class MNIST_Solver:

    def __init__(self, data: MNIST_Data.MNIST_Data) -> None:

        self.nn = Network.Neural_Network()

        self.nn.add_layer(Layers.LeakyReLU(784, 512))
        self.nn.add_layer(Layers.Softmax(512, 10))
        self.nn.add_loss(Loss.Cross_Entropy())

        self.categories = 10

        self.data = data

        self.flat_train_data = Layers.Flatten.flatten(data.train_images, 60000, 28, 28)
        self.flat_test_data = Layers.Flatten.flatten(data.test_images, 10000, 28, 28)

    def train(self, batch_size: int, epochs: int) -> None:
        
        self.nn.train(self.flat_train_data, data.train_labels, self.categories, batch_size, epochs)

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
        

data = MNIST_Data.MNIST_Data(path="./mnist_data/")
data.read_all_data()

network = MNIST_Solver(data)
network.train(50, 5)

network.print_img(1234)