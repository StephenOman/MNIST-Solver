import MNIST_Data
import MNIST_Solver


data = MNIST_Data.MNIST_Data(path="./mnist_data/")
data.read_all_data()

model = MNIST_Solver.MNIST_Solver(data, bias=False)
model.train(50, 5)

model.nn.metrics(model.flat_test_data, model.data.test_labels)

model.save()

model.nn.graph_training_loss()