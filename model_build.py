import sys
import getopt

import Config
import MNIST_Data
import MNIST_Solver

# this builder takes 1 parameter, which is the name of the config file
# if none is submitted available, then it will use mnist_solver_default.config

conf_filename = "mnist_solver_default.config"

opts, args = getopt.getopt(sys.argv, 'c:', ['config'])
if len(opts) == 0:
    print("Missing configuration file name. Defaulting to ", conf_filename)
else:
    for opt, arg in opts:
        if opt in ('-c', '--config'):
            conf_filename = arg
    

config = Config.Config()
config.read_config(conf_filename)

data = MNIST_Data.MNIST_Data(path=config.path_to_mnist_data)

data.read_all_data()

model = MNIST_Solver.MNIST_Solver(data, config)
model.train(config)

model.nn.metrics(model.flat_test_data, model.data.test_labels)

model.save(config.model_name)

model.nn.graph_training_loss()