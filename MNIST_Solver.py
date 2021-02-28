import Network
import Layers
import MNIST_Data

class MNIST_Solver:

    def __init__(self, data: MNIST_Data.MNIST_Data) -> None:

        nn = Network.Neural_Network()

        nn.add_layer(Layers.LeakyReLU(data.train_images[0].shape), number_nodes)

