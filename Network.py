import numpy as np

class Neural_Network:
    def __init__(self, learn_rate = 0.1) -> None:
        self.layers = []

    def add_layer(self, layer) -> None:
        self.layers.append(layer)

    def add_loss(self, loss) -> None:
        self.loss = loss

    def feedforward(self, input_data: np.array) -> np.array:
        inputs = np.transpose(input_data)
        for layer in self.layers:
            layer.feedforward(inputs)
            inputs = layer.outputs
        return inputs

    def backprop(self, error, labels) -> None:
        for layer in reversed(self.layers):
            layer.backprop(error, labels)
            error = layer.bp_error

    def train(self, input_data, input_labels, categories, batch_size, epochs = 5) -> None:

        # TODO - check for well-formed network before running training

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
                
                output = self.feedforward(inputs)

                self.loss.calc_loss(labels, categories, output)
                #if(batch_num % 100 == 0):
                #    print("Current loss (batch " + str(batch_num) + ") " + str(self.loss.loss))

                error = None
                self.backprop(error, labels)

                batch_start = batch_start + batch_size
                batch_end = batch_end + batch_size

                # check for last batch size being greater
                # than the number of remaining instances
                
                if(batch_end > input_data.shape[0]):
                    batch_end = input_data.shape[0]
                batch_num = batch_num + 1

            print("End of epoch with loss " + str(self.loss.loss))

        print("End of training with loss " + str(self.loss.loss))
            