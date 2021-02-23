import numpy as np

class Neural_Network:
    def __init__(self, learn_rate = 0.1) -> None:
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def add_loss(self, loss):
        self.loss = loss

    def train(self, input_data, input_labels, categories, batch_size, epochs = 5):
        for epoch in range(epochs):
            batch_start = 0
            batch_end = batch_start + batch_size
            batch_num = 1

            # In the case where batch size is set to be
            # greater than the number of training images
            if(batch_end > input_data.shape[0]):
                batch_end = input_data.shape[0]
            
            while(batch_start < input_data.shape[0]):
                inputs = input_data[batch_start:batch_end]
                labels = input_labels[batch_start:batch_end]
                
                for layer in self.layers:
                    layer.feedforward(inputs)

                loss.calc_loss(labels, categories, self.layers[-1].outputs)

                for layer in self.layers:
                    layer.backprop() #TODO
            