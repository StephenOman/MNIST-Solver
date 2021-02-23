import Layers
import numpy as np

# Base Layer Tests
class Test_Base:

    # data for testing
    number_inputs = 3
    number_nodes = 5
    known_1D_input = np.transpose(np.array([[1,2,3]]))
    expected_1D_output = np.transpose(np.array([[-0.77, 0.84, -0.91, 0.98, 0.75]]))

    known_2D_input = np.transpose(np.array([[1, 2, 3],
                                            [3, 2, 1],
                                            [2, 1, 3],
                                            [3, 1, 2]]))

    

    known_weights = np.array([[0.1, -0.2, 0.3, -0.4, 0.5],
                                [-0.6, 0.7, -0.8, 0.9, -0.1],
                                [0.11, -0.12, 0.13, -0.14, 0.15]])
    
    expected_2D_output = np.transpose(np.array([[-0.77, 0.84, -0.91, 0.98, 0.75], 
                                            [-0.79, 0.68, -0.57, 0.46, 1.45],
                                            [-0.07, -0.06, 0.19, -0.32, 1.35],
                                            [-0.08, -0.14, 0.36, -0.58, 1.7]]))
    

    # Base is an abstract class, so create a new class
    class Conc_Layer(Layers.Base):
        pass

    def test_init(self):
        base_layer = Test_Base.Conc_Layer(self.number_inputs, self.number_nodes)
        assert base_layer.weights.shape[0] == self.number_inputs
        assert base_layer.weights.shape[1] == self.number_nodes

    def test_feedforward(self):
        base_layer = Test_Base.Conc_Layer(self.number_inputs, self.number_nodes)
        # Override the random weights with the known weights
        base_layer.weights = self.known_weights
        base_layer.feedforward(self.known_1D_input)
        assert base_layer.outputs.shape == self.expected_1D_output.shape
        assert np.array_equal(np.round(base_layer.outputs,2), self.expected_1D_output)

        base_layer.feedforward(self.known_2D_input)
        assert base_layer.outputs.shape == self.expected_2D_output.shape
        assert np.array_equal(np.round(base_layer.outputs,2), self.expected_2D_output)

    # No test required for backprop method
    # as it doesn't actually do anything
    def test_backprop(self):
        assert True


# LeakyReLU Tests
class Test_LeakyReLU:

    # data for testing
    number_inputs = 3
    number_nodes = 5
    epsilon = 0.2

    known_1D_input = np.transpose(np.array([[1,2,3]]))
    expected_1D_output = np.transpose(np.array([[-0.077, 0.84, -0.091, 0.98, 0.75]]))
    known_1D_error = np.transpose(np.array([[0.18923, -0.23037, 0.27152, -0.31266, 0.11230]]))
    known_1D_delta = np.transpose(np.array([[0.01892, 0.03785, 0.05677],
                                            [-0.23037, -0.46074, -0.69111],
                                            [0.02715, 0.05430, 0.08146],
                                            [-0.31266, -0.62532, -0.93798],
                                            [0.11230, 0.2246, 0.33690]]))

    # Batch of inputs
    known_2D_input = np.transpose(np.array([[1, 2, 3],
                                            [3, 2, 1],
                                            [2, 1, 3],
                                            [3, 1, 2]]))
                                            
    expected_2D_output = np.transpose(np.array([[-0.077, 0.84, -0.091, 0.98, 0.75],
                                            [-0.079, 0.68, -0.057, 0.46, 1.45],
                                            [-0.007, -0.006, 0.19, -0.032, 1.35],
                                            [-0.008, -0.014, 0.36, -0.058, 1.7]]))


    known_weights = np.array([[0.1, -0.2, 0.3, -0.4, 0.5],
                                [-0.6, 0.7, -0.8, 0.9, -0.1],
                                [0.11, -0.12, 0.13, -0.14, 0.15]])

    

    def test_init_weight_shape(self):
        relu_layer = Layers.LeakyReLU(self.number_inputs, self.number_nodes)
        assert relu_layer.weights.shape[0] == self.number_inputs
        assert relu_layer.weights.shape[1] == self.number_nodes

    def test_init_epsilon_override(self):
        relu_layer = Layers.LeakyReLU(self.number_inputs, self.number_nodes, self.epsilon)
        assert relu_layer.epsilon == self.epsilon

    def test_feedforward(self):
        relu_layer = Layers.LeakyReLU(self.number_inputs, self.number_nodes)
        relu_layer.weights = self.known_weights
        relu_layer.feedforward(self.known_1D_input)
        assert np.array_equal(np.round(relu_layer.outputs,3), self.expected_1D_output)

        relu_layer.feedforward(self.known_2D_input)
        assert np.array_equal(np.round(relu_layer.outputs,3), self.expected_2D_output)

    def test_backprop(self):
        relu_layer = Layers.LeakyReLU(self.number_inputs, self.number_nodes)
        relu_layer.weights = self.known_weights
        relu_layer.feedforward(self.known_1D_input)
        relu_layer.backprop(self.known_1D_input, self.known_1D_error)
        assert np.array_equal(np.round(relu_layer.delta, 5), self.known_1D_delta)


# Softmax Layer Tests
class Test_Softmax:

    # data for testing
    number_inputs = 5
    number_nodes = 3

    known_1D_inputs = np.transpose(np.array([[-0.077, 0.84, -0.091, 0.98, 0.75]]))
    expected_1D_output = np.transpose(np.array([[0.129, 0.732, 0.139]]))

    # Batch of inputs
    known_2D_inputs = np.transpose(np.array([[-0.077, 0.84, -0.091, 0.98, 0.75],
                                            [-0.079, 0.68, -0.057, 0.46, 1.45],
                                            [-0.007, -0.006, 0.19, -0.032, 1.35],
                                            [-0.008, -0.014, 0.36, -0.058, 1.7]]))

    expected_2D_output = np.transpose(np.array([[0.129, 0.732, 0.139],
                                                [0.303, 0.479, 0.219],
                                                [0.514, 0.178, 0.308],
                                                [0.577, 0.129, 0.294]]))

    known_weights = np.array([[0.1, -0.6, 0.11],
                                [-0.2, 0.7, -0.12],
                                [0.3, -0.8, 0.13],
                                [-0.4, 0.9, -0.14],
                                [0.5, -0.1, 0.15]])

    known_1D_label = np.array([1])

    known_1D_delta = np.array([[-0.00995, 0.02066, -0.01071],
                                [0.10852, -0.22540, 0.11688],
                                [-0.01176, 0.02442, -0.01266],
                                [0.12661, -0.26297, 0.13636],
                                [0.09690, -0.20125, 0.10436]])

    known_1D_bp_error = np.transpose(np.array([[0.18923, -0.23037, 0.27152, -0.31266, 0.11230]]))

    known_2D_labels = np.array([1, 1, 0, 2])

    known_2D_delta = np.array([[-0.00877, 0.01490, -0.00613],
                                [0.07730, -0.14572, 0.06842],
                                [0.02158, 0.03359, -0.05517],
                                [0.06198, -0.12901, 0.06702],
                                [0.21513, -0.12450, -0.09064]])

    known_2D_bp_error = np.transpose(np.array([[0.18923, -0.23037, 0.27152, -0.31266, 0.11230],
                                                [0.36722, -0.45183, 0.53643, -0.62103, 0.23631],
                                                [-0.12151, 0.18481, -0.24810, 0.31139, -0.21449],
                                                [-0.09728, 0.05955, -0.02183, -0.01590, 0.16963]]))

    def test_init_weight_shape(self):
        sm_layer = Layers.Softmax(self.number_inputs, self.number_nodes)
        assert sm_layer.weights.shape[0] == self.number_inputs
        assert sm_layer.weights.shape[1] == self.number_nodes

    def test_feedforward(self):
        sm_layer = Layers.Softmax(self.number_inputs, self.number_nodes)
        sm_layer.weights = self.known_weights
        sm_layer.feedforward(self.known_1D_inputs)
        assert np.array_equal(np.round(sm_layer.outputs,3), self.expected_1D_output)

        sm_layer.feedforward(self.known_2D_inputs)
        assert np.array_equal(np.round(sm_layer.outputs,3), self.expected_2D_output)

    def test_backprop(self):
        sm_layer = Layers.Softmax(self.number_inputs, self.number_nodes)
        sm_layer.weights = self.known_weights
        sm_layer.feedforward(self.known_1D_inputs)
        sm_layer.backprop(self.known_1D_inputs, self.known_1D_label)
        assert np.array_equal(np.round(sm_layer.delta, 5), self.known_1D_delta)
        assert np.array_equal(np.round(sm_layer.bp_error, 5), self.known_1D_bp_error)

        sm_layer.feedforward(self.known_2D_inputs)
        sm_layer.backprop(self.known_2D_inputs, self.known_2D_labels)
        assert np.array_equal(np.round(sm_layer.delta, 5), self.known_2D_delta)
        assert np.array_equal(np.round(sm_layer.bp_error, 5), self.known_2D_bp_error)