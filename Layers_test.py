import Layers
import numpy as np


#################################################
#
#  Base Class Tests
#
#################################################

# Base Layer Test Single Instance No Bias Signals

class Test_11_Base_Single_No_Bias:
    number_inputs = 3
    number_nodes = 5
    known_single_input = np.transpose(np.array([[1,2,3]]))
    expected_single_output = np.transpose(np.array([[-0.77, 0.84, -0.91, 0.98, 0.75]]))
    known_weights = np.array([[0.1, -0.2, 0.3, -0.4, 0.5],
                                [-0.6, 0.7, -0.8, 0.9, -0.1],
                                [0.11, -0.12, 0.13, -0.14, 0.15]])

    # Base is an abstract class, so create a new concrete base class
    class Conc_Layer(Layers.Base):
        pass
    
    def test_init(self):
        base_layer = self.Conc_Layer(self.number_inputs, self.number_nodes)
        assert base_layer.weights.shape[0] == self.number_inputs
        assert base_layer.weights.shape[1] == self.number_nodes

    def test_feedforward(self):
        base_layer = self.Conc_Layer(self.number_inputs, self.number_nodes)
        # Override the random weights with the known weights
        base_layer.weights = self.known_weights

        base_layer.feedforward(self.known_single_input)
        assert base_layer.outputs.shape == self.expected_single_output.shape
        assert np.array_equal(np.round(base_layer.outputs,2), self.expected_single_output)

    def test_backprop(self):
        assert True     #TODO

# Base Layer Test Single Instance With Bias Signals
class Test_12_Base_Single_Bias:
    number_inputs = 3
    number_nodes = 5
    known_single_input = np.transpose(np.array([[1,2,3]]))
    expected_single_output_bias = np.transpose(np.array([[-0.57, 0.04, -0.61, 0.28, 1.15]]))
    known_weights_bias = np.array([[0.2, -0.8, 0.3, -0.7, 0.4],
                                [0.1, -0.2, 0.3, -0.4, 0.5],
                                [-0.6, 0.7, -0.8, 0.9, -0.1],
                                [0.11, -0.12, 0.13, -0.14, 0.15]])
    
    # Base is an abstract class, so create a new concrete base class
    class Conc_Layer(Layers.Base):
        pass

    def test_init_bias(self):
        base_layer = self.Conc_Layer(self.number_inputs, self.number_nodes, bias=True)
        assert base_layer.weights.shape[0] == self.number_inputs + 1
        assert base_layer.weights.shape[1] == self.number_nodes

    def test_feedforward_bias(self):
        base_layer = self.Conc_Layer(self.number_inputs, self.number_nodes, bias=True)
        # Override the random weights with the known weights
        base_layer.weights = self.known_weights_bias

        base_layer.feedforward(self.known_single_input)
        assert base_layer.outputs.shape == self.expected_single_output_bias.shape
        assert np.array_equal(np.round(base_layer.outputs,2), self.expected_single_output_bias)

    def test_backprop_bias(self):
        assert True     #TODO

# Base Layer Test Batch No Bias Signals
class Test_13_Base_Batch_No_Bias:
    # data for testing
    number_inputs = 3
    number_nodes = 5
    known_batch_input = np.transpose(np.array([[1, 2, 3],
                                                [3, 2, 1],
                                                [2, 1, 3],
                                                [3, 1, 2]]))
    expected_batch_output = np.transpose(np.array([[-0.77, 0.84, -0.91, 0.98, 0.75], 
                                            [-0.79, 0.68, -0.57, 0.46, 1.45],
                                            [-0.07, -0.06, 0.19, -0.32, 1.35],
                                            [-0.08, -0.14, 0.36, -0.58, 1.7]]))
    known_weights = np.array([[0.1, -0.2, 0.3, -0.4, 0.5],
                            [-0.6, 0.7, -0.8, 0.9, -0.1],
                            [0.11, -0.12, 0.13, -0.14, 0.15]])
    
    # Base is an abstract class, so create a new concrete base class
    class Conc_Layer(Layers.Base):
        pass

    def test_feedforward_batch(self):
        base_layer = self.Conc_Layer(self.number_inputs, self.number_nodes)
        # Override the random weights with the known weights
        base_layer.weights = self.known_weights

        base_layer.feedforward(self.known_batch_input)
        assert base_layer.outputs.shape == self.expected_batch_output.shape
        assert np.array_equal(np.round(base_layer.outputs,2), self.expected_batch_output)

    def test_backprop_batch(self):
        assert True     #TODO

# Base Layer Test Batch With Bias Signals
class Test_14_Base_Batch_With_Bias:
    # data for testing
    number_inputs = 3
    number_nodes = 5
    known_batch_input = np.transpose(np.array([[1, 2, 3],
                                                [3, 2, 1],
                                                [2, 1, 3],
                                                [3, 1, 2]]))
    expected_batch_output_bias = np.transpose(np.array([[-0.57, 0.04, -0.61, 0.28, 1.15],
                                        [-0.59, -0.12, -0.27, -0.24, 1.85], 
                                        [0.13, -0.86, 0.49, -1.02, 1.75],
                                        [0.12, -0.94, 0.66, -1.28, 2.1]]))
    
    known_weights_bias = np.array([[0.2, -0.8, 0.3, -0.7, 0.4],
                                [0.1, -0.2, 0.3, -0.4, 0.5],
                                [-0.6, 0.7, -0.8, 0.9, -0.1],
                                [0.11, -0.12, 0.13, -0.14, 0.15]])

    # Base is an abstract class, so create a new concrete base class
    class Conc_Layer(Layers.Base):
        pass

    def test_feedforward_bias(self):
        base_layer = self.Conc_Layer(self.number_inputs, self.number_nodes, bias=True)
        # Override the random weights with the known weights
        base_layer.weights = self.known_weights_bias

        base_layer.feedforward(self.known_batch_input)
        assert base_layer.outputs.shape == self.expected_batch_output_bias.shape
        assert np.array_equal(np.round(base_layer.outputs,2), self.expected_batch_output_bias)

    def test_backprop_batch_bias(self):
        assert True     #TODO

#################################################
#
# End of Base Class Tests
#
#################################################


#################################################
#
# LeakyReLU Tests
#
#################################################

class Test_21_LeakyRelu_Single_No_Bias:
    # data for testing
    number_inputs = 3
    number_nodes = 5
    epsilon = 0.2
    known_single_input = np.transpose(np.array([[1,2,3]]))
    known_weights = np.array([[0.1, -0.2, 0.3, -0.4, 0.5],
                            [-0.6, 0.7, -0.8, 0.9, -0.1],
                            [0.11, -0.12, 0.13, -0.14, 0.15]])

    expected_single_output = np.transpose(np.array([[-0.077, 0.84, -0.091, 0.98, 0.75]]))
    known_single_error = np.transpose(np.array([[0.18923, -0.23037, 0.27152, -0.31266, 0.11230]]))
    known_single_delta = np.transpose(np.array([[0.01892, 0.03785, 0.05677],
                                        [-0.23037, -0.46074, -0.69111],
                                        [0.02715, 0.05430, 0.08146],
                                        [-0.31266, -0.62532, -0.93798],
                                        [0.11230, 0.2246, 0.33690]]))

    def test_init_weight_shape(self):
        relu_layer = Layers.LeakyReLU(self.number_inputs, self.number_nodes)
        assert relu_layer.weights.shape[0] == self.number_inputs
        assert relu_layer.weights.shape[1] == self.number_nodes

    def test_init_epsilon_override(self):
        relu_layer = Layers.LeakyReLU(self.number_inputs, self.number_nodes, epsilon=self.epsilon)
        assert relu_layer.epsilon == self.epsilon

    def test_feedforward(self):
        relu_layer = Layers.LeakyReLU(self.number_inputs, self.number_nodes)
        relu_layer.weights = self.known_weights
        relu_layer.feedforward(self.known_single_input)
        assert np.array_equal(np.round(relu_layer.outputs,3), self.expected_single_output)

    def test_backprop(self):
        relu_layer = Layers.LeakyReLU(self.number_inputs, self.number_nodes)
        relu_layer.weights = self.known_weights
        relu_layer.feedforward(self.known_single_input)
        relu_layer.backprop(self.known_single_error, None)
        assert np.array_equal(np.round(relu_layer.weights, 5),  
                        np.round((self.known_weights - (self.known_single_delta * relu_layer.learn_rate)),5))

class Test_22_LeakyRelu_Single_Bias:
    # data for testing
    number_inputs = 3
    number_nodes = 5
    epsilon = 0.2
    known_single_input = np.transpose(np.array([[1,2,3]]))
    known_weights_bias = np.array([[0.2, -0.8, 0.3, -0.7, 0.4],
                                [0.1, -0.2, 0.3, -0.4, 0.5],
                                [-0.6, 0.7, -0.8, 0.9, -0.1],
                                [0.11, -0.12, 0.13, -0.14, 0.15]]) 

    expected_single_output_bias = np.transpose(np.array([[-0.057, 0.04, -0.061, 0.28, 1.15]]))
    known_single_error_bias = np.transpose(np.array([[0.20653, -0.25273, 0.29892, -0.34512, 0.12755]]))
    
    expected_revised_weights_bias = np.array([[0.19793, -0.77473,  0.29701, -0.66549,  0.38724],
                                            [0.09793, -0.17473,  0.29701, -0.36549,  0.48724],
                                            [-0.60413,  0.75055, -0.80598,  0.96902, -0.12551],
                                            [0.1038 , -0.04418,  0.12103, -0.03646,  0.11174]])

    def test_feedforward_bias(self):
        relu_layer = Layers.LeakyReLU(self.number_inputs, self.number_nodes, bias=True)
        relu_layer.weights = self.known_weights_bias
        
        relu_layer.feedforward(self.known_single_input)
        assert np.array_equal(np.round(relu_layer.outputs,3), self.expected_single_output_bias)

    def test_backprop_bias(self):
        relu_layer = Layers.LeakyReLU(self.number_inputs, self.number_nodes, bias=True)
        relu_layer.weights = self.known_weights_bias
        relu_layer.feedforward(self.known_single_input)
        relu_layer.backprop(self.known_single_error_bias, None)
        assert np.array_equal(np.round(relu_layer.weights, 5), self.expected_revised_weights_bias)

class Test_23_LeakyRelu_Batch_No_Bias:
    # data for testing
    number_inputs = 3
    number_nodes = 5
    epsilon = 0.2
    known_batch_input = np.transpose(np.array([[1, 2, 3],
                                            [3, 2, 1],
                                            [2, 1, 3],
                                            [3, 1, 2]]))
    known_weights = np.array([[0.1, -0.2, 0.3, -0.4, 0.5],
                                [-0.6, 0.7, -0.8, 0.9, -0.1],
                                [0.11, -0.12, 0.13, -0.14, 0.15]]) 

    expected_batch_output = np.transpose(np.array([[-0.077, 0.84, -0.091, 0.98, 0.75],
                                            [-0.079, 0.68, -0.057, 0.46, 1.45],
                                            [-0.007, -0.006, 0.19, -0.032, 1.35],
                                            [-0.008, -0.014, 0.36, -0.058, 1.7]]))

    known_batch_error = np.transpose(np.array([[0.18923, -0.23037, 0.27152, -0.31266, 0.11230],
                                                [0.36722, -0.45182, 0.53643, -0.62103, 0.23631],
                                                [-0.12151, 0.18480, -0.24810, 0.31138, -0.21449],
                                                [-0.09728, 0.05955, -0.02183, -0.01590, 0.16963]]))           

    known_batch_delta = np.array([[0.01890,	-0.38275, -0.09340,	-0.52956, 0.22528],
                                [0.02235, -0.33499, -0.02708, -0.45946, 0.16309],
                                [0.00940, -0.26890, -0.16322, -0.36719, 0.06725]])

    def test_feedforward(self):
        relu_layer = Layers.LeakyReLU(self.number_inputs, self.number_nodes)
        relu_layer.weights = self.known_weights
        relu_layer.feedforward(self.known_batch_input)
        assert np.array_equal(np.round(relu_layer.outputs,3), self.expected_batch_output)     

    def test_batch_backprop(self):
        relu_layer = Layers.LeakyReLU(self.number_inputs, self.number_nodes)
        relu_layer.weights = self.known_weights
        relu_layer.feedforward(self.known_batch_input)
        relu_layer.backprop(self.known_batch_error, None)
        assert np.array_equal(np.round(relu_layer.weights, 5),
                        np.round((self.known_weights - (self.known_batch_delta * relu_layer.learn_rate)),5))

class Test_24_LeakyRelu_Batch_With_Bias:
    # data for testing
    number_inputs = 3
    number_nodes = 5
    epsilon = 0.2
    known_batch_input = np.transpose(np.array([[1, 2, 3],
                                            [3, 2, 1],
                                            [2, 1, 3],
                                            [3, 1, 2]]))

    known_weights_bias = np.array([[0.2, -0.8, 0.3, -0.7, 0.4],
                                [0.1, -0.2, 0.3, -0.4, 0.5],
                                [-0.6, 0.7, -0.8, 0.9, -0.1],
                                [0.11, -0.12, 0.13, -0.14, 0.15]])   
    
    expected_batch_output_bias = np.array([[-0.057, -0.059, 0.13, 0.12],  
                                            [0.04, -0.012, -0.086, -0.094],
                                            [-0.061, -0.027, 0.49, 0.66],
                                            [0.28, -0.024, -0.102, -0.128],
                                            [1.15, 1.85, 1.75, 2.1]])

    known_batch_error_bias = np.transpose(np.array([[0.18923, -0.23037, 0.27152, -0.31266, 0.11230],
                                                [0.36722, -0.45182, 0.53643, -0.62103, 0.23631],
                                                [-0.12151, 0.18480, -0.24810, 0.31138, -0.21449],
                                                [-0.09728, 0.05955, -0.02183, -0.01590, 0.16963]]))

    known_batch_delta_bias = np.array([[-0.09383, -0.06075, -0.12529, -0.08395, 0.03995],
                                [-0.23256, -0.06408, -0.28635, -0.09201, 0.12876],
                                [-0.08042, -0.13411, -0.10571, -0.18427, 0.12616],
                                [-0.25002, -0.16632, -0.35969, -0.22743, -0.01522]])

    expected_revised_weights_bias = np.array([[ 0.20408, -0.79372,  0.30473, -0.69137,  0.39241],
                                            [0.11014, -0.19222,  0.30934, -0.38896,  0.47747],
                                            [-0.59731,  0.71317, -0.79729,  0.918  , -0.11631],
                                            [0.12164, -0.10328,  0.14632, -0.11725,  0.14327]])
    def test_feedforward_bias(self):
        relu_layer = Layers.LeakyReLU(self.number_inputs, self.number_nodes, bias=True)
        relu_layer.weights = self.known_weights_bias

        relu_layer.feedforward(self.known_batch_input)
        assert np.array_equal(np.round(relu_layer.outputs,3), self.expected_batch_output_bias)

    def test_batch_backprop_bias(self):
        relu_layer = Layers.LeakyReLU(self.number_inputs, self.number_nodes, bias=True)
        relu_layer.weights = self.known_weights_bias
        relu_layer.feedforward(self.known_batch_input)
        relu_layer.backprop(self.known_batch_error_bias, None)
        assert np.array_equal(np.round(relu_layer.weights, 5), self.expected_revised_weights_bias)


#################################################
#
# End of LeakyReLU Class Tests
#
#################################################


#################################################
#
# Softamx Tests
#
#################################################

class Test_31_Softmax_Single_No_Bias:
    # data for testing
    number_inputs = 5
    number_nodes = 3

    # single instance test case
    known_single_inputs = np.transpose(np.array([[-0.077, 0.84, -0.091, 0.98, 0.75]]))
    expected_single_output = np.transpose(np.array([[0.129, 0.732, 0.139]]))
    known_single_label = np.array([1])
    known_single_delta = np.array([[-0.00995, 0.02066, -0.01071],
                                [0.10852, -0.22540, 0.11688],
                                [-0.01176, 0.02442, -0.01266],
                                [0.12661, -0.26297, 0.13636],
                                [0.09690, -0.20125, 0.10436]])
    known_single_bp_error = np.transpose(np.array([[0.18923, -0.23037, 0.27152, -0.31266, 0.11230]]))

    known_weights = np.array([[0.1, -0.6, 0.11],
                            [-0.2, 0.7, -0.12],
                            [0.3, -0.8, 0.13],
                            [-0.4, 0.9, -0.14],
                            [0.5, -0.1, 0.15]])
   
    expected_revised_weights = np.array([[ 0.10099, -0.60207,  0.11107],
                            [-0.21085,  0.72254, -0.13169],
                            [0.30118, -0.80244,  0.13127],
                            [-0.41266,  0.9263 , -0.15364],
                            [0.49031, -0.07987,  0.13956]])
    def test_init_weight_shape(self):
        sm_layer = Layers.Softmax(self.number_inputs, self.number_nodes)
        assert sm_layer.weights.shape[0] == self.number_inputs
        assert sm_layer.weights.shape[1] == self.number_nodes

    def test_feedforward(self):
        sm_layer = Layers.Softmax(self.number_inputs, self.number_nodes)
        sm_layer.weights = self.known_weights
        sm_layer.feedforward(self.known_single_inputs)
        assert np.array_equal(np.round(sm_layer.outputs,3), self.expected_single_output)

    def test_backprop(self):
        sm_layer = Layers.Softmax(self.number_inputs, self.number_nodes)
        sm_layer.weights = self.known_weights
        sm_layer.feedforward(self.known_single_inputs)
        sm_layer.backprop(None, self.known_single_label)
        assert np.array_equal(np.round(sm_layer.weights, 5), self.expected_revised_weights)
        assert np.array_equal(np.round(sm_layer.bp_error, 5), self.known_single_bp_error)


class Test_32_Softmax_Single_With_Bias:
    # data for testing
    number_inputs = 5
    number_nodes = 3

    # single instance test case
    known_single_inputs_bias = np.transpose(np.array([[-0.057, 0.04, -0.061, 0.28, 1.15]]))
    known_single_label = np.array([1])
    known_weights_bias = np.array([[-0.9, 0.8, -0.7],
                            [0.1, -0.6, 0.11],
                            [-0.2, 0.7, -0.12],
                            [0.3, -0.8, 0.13],
                            [-0.4, 0.9, -0.14],
                            [0.5, -0.1, 0.15]])
    known_single_delta_bias = np.array([[0.15507, -0.29307, 0.13800],
                                [-0.00884, 0.016705, -0.00787],
                                [0.00620, -0.01172, 0.00552],
                                [-0.00946, 0.01788, -0.00841],
                                [0.04342, -0.08206, 0.03864],
                                [0.17834, -0.33703, 0.15870]])
    expected_single_output_bais = np.transpose(np.array([[0.155, 0.707, 0.138]]))
    expected_single_bp_error_bias = np.transpose(np.array([[0.20653, -0.25273, 0.29892, -0.34512, 0.12755]]))
    expected_revised_weights_bias = np.array([[-0.91551, 0.82931, -0.71380],
                            [0.10088, -0.60167, 0.11079],
                            [-0.20062, 0.70117, -0.12055],
                            [0.30095, -0.80179, 0.13084],
                            [-0.40434, 0.90821, -0.14386],
                            [0.48217, -0.06630, 0.13413]])

    def test_feedforward(self):
        sm_layer = Layers.Softmax(self.number_inputs, self.number_nodes, bias=True)
        sm_layer.weights = self.known_weights_bias
        sm_layer.feedforward(self.known_single_inputs_bias)
        assert np.array_equal(np.round(sm_layer.outputs,3), self.expected_single_output_bais)   

    def test_backprop(self):
        sm_layer = Layers.Softmax(self.number_inputs, self.number_nodes, bias=True)
        sm_layer.weights = self.known_weights_bias
        sm_layer.feedforward(self.known_single_inputs_bias)
        sm_layer.backprop(None, self.known_single_label)
        assert np.array_equal(np.round(sm_layer.weights, 5), self.expected_revised_weights_bias)
        assert np.array_equal(np.round(sm_layer.bp_error, 5), self.expected_single_bp_error_bias)


class Test_33_Softmax_Batch_No_Bias:
    # data for testing
    number_inputs = 5
    number_nodes = 3

    known_batch_inputs = np.transpose(np.array([[-0.077, 0.84, -0.091, 0.98, 0.75],
                                            [-0.079, 0.68, -0.057, 0.46, 1.45],
                                            [-0.007, -0.006, 0.19, -0.032, 1.35],
                                            [-0.008, -0.014, 0.36, -0.058, 1.7]]))
    known_batch_labels = np.array([1, 1, 0, 2])        
    known_weights = np.array([[0.1, -0.6, 0.11],
                            [-0.2, 0.7, -0.12],
                            [0.3, -0.8, 0.13],
                            [-0.4, 0.9, -0.14],
                            [0.5, -0.1, 0.15]])
    expected_batch_output = np.transpose(np.array([[0.129, 0.732, 0.139],
                                                [0.303, 0.479, 0.219],
                                                [0.514, 0.178, 0.308],
                                                [0.577, 0.129, 0.294]]))

    expected_revised_weights = np.array([[0.10088, -0.60149, 0.11061],
                            [-0.20773, 0.71457, -0.12684],
                            [0.29784, -0.80336, 0.13552],
                            [-0.40620, 0.91290, -0.14670],
                            [0.47849, -0.08755, 0.15906]])

    expected_batch_bp_error = np.transpose(np.array([[0.18923, -0.23037, 0.27152, -0.31266, 0.11230],
                                                [0.36722, -0.45183, 0.53643, -0.62103, 0.23631],
                                                [-0.12151, 0.18481, -0.24810, 0.31139, -0.21449],
                                                [-0.09728, 0.05955, -0.02183, -0.01590, 0.16963]]))
    def test_batch_feedforward(self):
        sm_layer = Layers.Softmax(self.number_inputs, self.number_nodes)
        sm_layer.weights = self.known_weights

        sm_layer.feedforward(self.known_batch_inputs)
        assert np.array_equal(np.round(sm_layer.outputs,3), self.expected_batch_output)                                

    def test_batch_backprop(self):
        sm_layer = Layers.Softmax(self.number_inputs, self.number_nodes)
        sm_layer.weights = self.known_weights
        sm_layer.feedforward(self.known_batch_inputs)
        sm_layer.backprop(None, self.known_batch_labels)
        assert np.array_equal(np.round(sm_layer.weights, 5), self.expected_revised_weights)
        assert np.array_equal(np.round(sm_layer.bp_error, 5), self.expected_batch_bp_error)

class Test_34_Softmax_Batch_With_Bias:
    # data for testing
    number_inputs = 5
    number_nodes = 3

    known_batch_inputs = np.array([[-0.057, -0.059, 0.13, 0.12],
                                    [0.04, -0.012, -0.086, -0.094],
                                    [-0.061, -0.027, 0.49, 0.66],
                                    [0.28, -0.024, -0.102, -0.128],
                                    [1.15, 1.85, 1.75, 2.1]])
    known_batch_labels = np.array([1, 1, 0, 2])        
    known_weights_bias = np.array([[-0.9, 0.8, -0.7],
                            [0.1, -0.6, 0.11],
                            [-0.2, 0.7, -0.12],
                            [0.3, -0.8, 0.13],
                            [-0.4, 0.9, -0.14],
                            [0.5, -0.1, 0.15]])

    expected_batch_output = np.array([[0.15508, 0.28618, 0.41377, 0.49005],
                                    [0.70693, 0.53146, 0.34217, 0.26336],
                                    [0.13800, 0.18235, 0.24406, 0.24659]])

    expected_revised_weights = np.array([[-0.90863,  0.8039 , -0.69527],
                                        [0.10108, -0.60301,  0.11193],
                                        [-0.20018,  0.70151, -0.12133],
                                        [0.29953, -0.8093 ,  0.13978],
                                        [-0.40084,  0.90349, -0.14265],
                                        [0.48223, -0.0987 ,  0.16648]])

    expected_batch_bp_error = np.transpose(np.array([[0.20653, -0.25273, 0.29892, -0.34512, 0.12755],
                                        [0.32980, -0.40709, 0.48439, -0.56169, 0.21730],
                                        [-0.23708, 0.32748, -0.41788, 0.50828, -0.29072],
                                        [-0.19189, 0.17675, -0.16161, 0.14648, 0.10568]]))

    def test_feedforward(self):
        sm_layer = Layers.Softmax(self.number_inputs, self.number_nodes, bias=True)
        sm_layer.weights = self.known_weights_bias
        sm_layer.feedforward(self.known_batch_inputs)
        assert np.array_equal(np.round(sm_layer.outputs,5), self.expected_batch_output)

    def test_batch_backprop(self):
        sm_layer = Layers.Softmax(self.number_inputs, self.number_nodes, bias=True)
        sm_layer.weights = self.known_weights_bias
        sm_layer.feedforward(self.known_batch_inputs)
        sm_layer.backprop(None, self.known_batch_labels)
        assert np.array_equal(np.round(sm_layer.weights, 5), self.expected_revised_weights)
        assert np.array_equal(np.round(sm_layer.bp_error, 5), self.expected_batch_bp_error)


#################################################
#
# End of Softmax Class Tests
#
#################################################