import Network
import Layers
import Loss
import numpy as np

class Test_Network:
    
    # Test Data
    number_inputs = 3
    number_nodes = 5

    relu = Layers.LeakyReLU(number_inputs, number_nodes)

    relu.weights = np.array([[0.1, -0.2, 0.3, -0.4, 0.5],
                                [-0.6, 0.7, -0.8, 0.9, -0.1],
                                [0.11, -0.12, 0.13, -0.14, 0.15]])

    sm = Layers.Softmax(number_inputs, number_nodes)

    sm.weights = np.array([[0.1, -0.6, 0.11],
                                [-0.2, 0.7, -0.12],
                                [0.3, -0.8, 0.13],
                                [-0.4, 0.9, -0.14],
                                [0.5, -0.1, 0.15]])

    loss = Loss.Cross_Entropy()

    known_1D_input = np.array([[1,2,3]])
    known_1D_xe_loss = 0.312
    expected_1D_output = np.transpose(np.array([[0.129, 0.732, 0.139]]))
    known_1D_label = np.array([1])

    known_2D_input = np.array([[1, 2, 3],
                                [3, 2, 1],
                                [2, 1, 3],
                                [3, 1, 2]])
    known_2D_xe_loss = 0.734
    expected_2D_output = np.transpose(np.array([[0.129, 0.732, 0.139],
                                                [0.303, 0.479, 0.219],
                                                [0.514, 0.178, 0.308],
                                                [0.577, 0.129, 0.294]]))
    known_2D_labels = np.array([1, 1, 0, 2])

    def test_add_layer(self) -> None:
        nn = Network.Neural_Network()
        nn.add_layer(self.relu)

        assert len(nn.layers) == 1
        assert isinstance(nn.layers[0], Layers.LeakyReLU)

    def test_add_loss(self) -> None:
        nn = Network.Neural_Network()
        nn.add_loss(self.loss)

        assert isinstance(nn.loss, Loss.Cross_Entropy)

    def test_train(self) -> None:
        nn = Network.Neural_Network()

        nn.add_layer(self.relu)
        nn.add_layer(self.sm)
        nn.add_loss(self.loss)

        nn.train(self.known_1D_input, self.known_1D_label, categories = 3, batch_size = 1, epochs = 1)
        assert np.round(nn.loss.loss,3) == self.known_1D_xe_loss

        nn.train(self.known_2D_input, self.known_2D_labels, categories = 3, batch_size = 4, epochs = 1)
        assert np.round(nn.loss.loss,3) == self.known_2D_xe_loss
