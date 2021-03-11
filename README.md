# MNIST-Solver

## Overview

MNIST Solver is an artificial neural network that can predict a digit from a handwritten image. MNIST is a dataset of 60,000 black & white images that can be used to train the network, and a further 10,000 images to test it.

There are two approaches to building a model. The first is coded in Python using Jupyter notebooks so you can see the detail behind the various blocks of code. The second approach uses pure Python so that the components can be reused.

Note that there are two notebooks, one includes bias signals on every node in the network, while the second omits them.

## Dependencies

These are the various libraries that I've used to create this solution.

### For the Python solution

* Python 3.9.1
* Numpy 1.19.5
* Matplotlib 3.3.3

### For the Jupyter Notebook solution (in addition to the libraries above)

Jupyter 1.0.0

## Running the Python Builder

To build a model using the training set:

```
$ python model_build.py
```

By default, this will create the "mnist.model" file. It will also produce a graph of the loss of each batch through the training run.

![Sample Run](/MNIST_Loss_No_Bias.png)

To test the model using the test dataset:

```
$ python model_check.py
```

This will produce a table of statistics similar to this one:
```
Total predictions  10000
Total correct predictions  9436
Overall model accuracy 0.9436 

Digit Total  Correct  Accuracy     TP     FN     FP     TN  Precision   Recall  Specificity  F1score
  0     980      959   0.97857    959     28   8992     21    0.97163  0.97857      0.99690  0.97509
  1    1135     1122   0.98855   1122     44   8821     13    0.96226  0.98855      0.99504  0.97523
  2    1032      963   0.93314    963     41   8927     69    0.95916  0.93314      0.99543  0.94597
  3    1010      922   0.91287    922     51   8939     88    0.94758  0.91287      0.99433  0.92990
  4     982      949   0.96640    949     56   8962     33    0.94428  0.96640      0.99379  0.95521
  5     892      869   0.97422    869    174   8934     23    0.83317  0.97422      0.98090  0.89819
  6     958      879   0.91754    879     17   9025     79    0.98103  0.91754      0.99812  0.94822
  7    1028      940   0.91440    940     21   8951     88    0.97815  0.91440      0.99766  0.94520
  8     974      892   0.91581    892     57   8969     82    0.93994  0.91581      0.99368  0.92772
  9    1009      941   0.93261    941     75   8916     68    0.92618  0.93261      0.99166  0.92938
```

TP, FN, FP and TN are True Positive, False Negative, False Positive and True Negative respectively.

The model checker will also produce a sample prediction.

![Sample Prediction](/test_image_1307.png)

## Tests

There are some tests for the Loss function, the Layers classes and the Network class. They are run with pytest.

