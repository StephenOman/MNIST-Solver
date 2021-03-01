import Network
import Layers
import Loss
import MNIST_Data

class MNIST_Solver:

    def __init__(self) -> None:

        self.nn = Network.Neural_Network()

        self.nn.add_layer(Layers.LeakyReLU(784, 512))
        self.nn.add_layer(Layers.Softmax(512, 10))
        self.nn.add_loss(Loss.Cross_Entropy())

        self.categories = 10

    def train(self, data: MNIST_Data.MNIST_Data, batch_size: int, epochs: int) -> None:

        flat_data = Layers.Flatten.flatten(data.train_images, 60000, 28, 28)
        self.nn.train(flat_data, data.train_labels, self.categories, batch_size, epochs)
        

data = MNIST_Data.MNIST_Data(path="./mnist_data/")
data.read_all_data()

network = MNIST_Solver()
network.train(data, 50, 5)