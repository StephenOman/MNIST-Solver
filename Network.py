import time
import numpy as np
from matplotlib import pyplot as plt

class Neural_Network:
    def __init__(self, learn_rate = 0.1) -> None:
        self.layers = []
        self.loss_track = []

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
        t_start = time.perf_counter()
        
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
                self.loss_track.append(self.loss.loss)
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

        t_end = time.perf_counter()
        print('Elapsed time ', t_end - t_start)     
        print("End of training with loss " + str(self.loss.loss))


    def metrics(self, test_data: np.array, test_labels: np.array):   

        test_results = self.feedforward(test_data)

        correct_predictions_per_digit = np.zeros((10,2))

        for prediction in range(test_results.shape[1]):
            correct_predictions_per_digit[test_labels[prediction]][0] += 1
            if(np.argmax(np.transpose(test_results)[prediction]) == test_labels[prediction]):
                correct_predictions_per_digit[test_labels[prediction]][1] += 1
                
        total_predictions = prediction + 1
        total_correct_predictions = int(np.sum(correct_predictions_per_digit, axis = 0)[1])

        stats = [{"tp":0, "fp":0, "tn":0, "fn":0} for i in range(10)]

        for stat in range(10):
            for prediction in range(test_results.shape[1]):
                if(stat == test_labels[prediction]):
                    # True Positives & False Negative
                    if(np.argmax(np.transpose(test_results)[prediction]) == stat):
                        stats[stat]["tp"] += 1
                    else:
                        stats[stat]["fn"] += 1
                else:
                    # False Positives and True Negatives
                    if(np.argmax(np.transpose(test_results)[prediction]) == stat):
                        stats[stat]["fp"] += 1
                    else:
                        stats[stat]["tn"] += 1

        print("Total predictions ", total_predictions)
        print("Total correct predictions ", total_correct_predictions)
        print("Overall model accuracy", total_correct_predictions / total_predictions, "\n")

        print("{0:5} {1:6} {2:8} {3:8} {4:>6} {5:>6} {6:>6} {7:>6} {8:>10} {9:>8} {10:>12} {11:>8}".format(
            "Digit", "Total", "Correct", "Accuracy", "TP", "FN", "FP", "TN", 
            "Precision", "Recall", "Specificity", "F1score"))

        for digit in range(10):
            total = int(correct_predictions_per_digit[digit][0])
            correct = int(correct_predictions_per_digit[digit][1])
            accuracy = correct_predictions_per_digit[digit][1] / correct_predictions_per_digit[digit][0]
            tp = stats[digit]["tp"]
            fp = stats[digit]["fp"]
            tn = stats[digit]["tn"]
            fn = stats[digit]["fn"]
            precision =  tp / (tp + fp)
            recall = tp / (tp + fn)
            specificity = tn / (tn + fp)
            f1score = (tp * 2) / ((tp * 2) + fp + fn)
            
            print("{0:^5} {1:5} {2:8} {3:9.5f} {4:6} {5:6} {6:6} {7:6} {8:10.5f} {9:8.5f} {10:12.5f} {11:8.5f}"
                .format(digit, total, correct, accuracy, tp, fp, tn, fn, 
                        precision, recall, specificity, f1score))

    def clear_data(self):
       for layer in self.layers:
           layer.inputs = None
           layer.outputs = None
           layer.bp_error = None

    def graph_training_loss(self):
        fig, ax = plt.subplots()

        plt.title("Training Loss")
        ax.plot(self.loss_track)
        ax.set_xlabel("Batch")
        ax.set_ylabel("Loss")
        plt.show()