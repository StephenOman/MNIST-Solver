import configparser
import os

class Config():
    def __init__(self):
        # Defaults for the model - can be overridden in the config file
        self.path_to_mnist_data = "./mnist_data/"
        self.bias = False
        self.model_name = "default.model"
        self.batch_size = 50
        self.epochs = 5
        self.learn_rate = 0.1

    def read_config(self, config_filename):
        if not os.path.exists(config_filename):
            print("Configuration file ", config_filename, " not found. Using defaults.")
        else:
            configs = configparser.ConfigParser()
            configs.read(config_filename)

            for option in configs.options("General"):
                if(option == "path_to_mnist_data"):
                    self.path_to_mnist_data = configs.get("General", option)
                elif(option == "model_name"):
                    self.model_name = configs.get("General", option)
                elif(option == "bias"):
                    self.bias = configs.getboolean("General", option)

            for option in configs.options("Network"):
                if(option == "learn_rate"):
                    self.batch_size = configs.getfloat("Network", option)

            for option in configs.options("Training"):
                if(option == "batch_size"):
                    self.batch_size = configs.getint("Training", option)
                elif(option == "epochs"):
                    self.epochs = configs.getint("Training", option)
