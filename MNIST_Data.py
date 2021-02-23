####################
#
# MNIST data class
#
####################

import struct
import traceback
import numpy as np

from typing import Tuple


class MNIST_Data:

    normalise_value = 255.0

    def __init__(self, path = "./", train_data_filename = "train-images-idx3-ubyte",
            train_labels_filename = "train-labels-idx1-ubyte",
            test_data_filename = "t10k-images-idx3-ubyte",
            test_labels_filename = "t10k-labels-idx1-ubyte") -> None:
        if(path[-1:0] != '/'):
            self.train_filename = path + '/' + train_data_filename
            self.train_label_filename = path + '/' + train_labels_filename
            self.test_filename = path + '/' + test_data_filename
            self.test_label_filename = path + '/' + test_labels_filename
        else:
            self.train_filename = path + train_data_filename
            self.train_label_filename = path + train_labels_filename
            self.test_filename = path + test_data_filename
            self.test_label_filename = path + test_labels_filename 

        self.train_images_count = 0
        self.train_labels_count = 0
        self.test_images_count = 0
        self.test_labels_count = 0

        self.train_images = None
        self.train_labels = None
        self.test_images = None
        self.test_labels = None

    def __one_hot(self, indexes):
        y = np.zeros((indexes.shape[0],10))
        for i in range(y.shape[0]):
            y[i][indexes[i]] = 1
        return y

    def __read_labels(self, filename) -> Tuple[int, np.ndarray]:
        try:
            with open(filename, 'rb') as f:
                label_magic, label_count = struct.unpack('>LL', f.read(8))
                label_data = np.fromfile(f, dtype=np.uint8)
        except Exception as e:
            print("Exception reading from " + filename)
            print(e.message)
            return(None, None)
        finally:
            return(label_count, label_data)

    def read_train_labels(self):
        self.train_labels_count, self.train_labels = self.__read_labels(self.train_label_filename)
        self.train_labels_1h = self.__one_hot(self.train_labels)
    
    def read_test_labels(self):
        self.test_labels_count, self.test_labels = self.__read_labels(self.test_label_filename)
        self.test_labels_1h = self.__one_hot(self.test_labels)

    def __read_data(self, filename) -> Tuple[int, np.ndarray]:
        try:
            with open(filename, 'rb') as f:
                data_magic, data_images_count, self.rows, self.columns = struct.unpack('>LLLL', f.read(16))
                image_data = np.fromfile(f, dtype=np.uint8)
        except Exception as e:
            print("Exception reading from " + filename)
            print(e.message)
            return(None, None)
        finally:
            return(data_images_count, image_data)

    def read_train_data(self):
        self.train_images_count, self.train_images = self.__read_data(self.train_filename)
        self.train_images = self.train_images / self.normalise_value
    
    def read_test_data(self):
        self.test_images_count, self.test_images = self.__read_data(self.test_filename)
        self.test_images = self.test_images / self.normalise_value