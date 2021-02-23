#import MNIST_Data as md
#data = md.MNIST_Data(path="./mnist_data/")
#data.read_train_labels()

import Layers

l0 = Layers.LeakyReLU(3, 5)

print(l0.weights)

