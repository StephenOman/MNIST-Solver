import MNIST_Data
import MNIST_Solver


data = MNIST_Data.MNIST_Data(path="./mnist_data/")
data.read_all_data()

model = MNIST_Solver.MNIST_Solver(data)
model.load(model_file="mnist_bias.model")

model.nn.metrics(model.flat_test_data, model.data.test_labels)

model.print_img(1307)