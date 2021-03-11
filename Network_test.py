import Network
import Layers
import Loss
import numpy as np

class Test_Network:

    loss = Loss.Cross_Entropy()

    known_single_input = np.array([[1,2,3]])
    known_single_xe_loss = 0.312
    expected_single_output = np.transpose(np.array([[0.129, 0.732, 0.139]]))
    known_single_label = np.array([1])

    known_batch_input = np.array([[1, 2, 3],
                                [3, 2, 1],
                                [2, 1, 3],
                                [3, 1, 2]])
    known_batch_xe_loss = 0.734
    expected_batch_output = np.transpose(np.array([[0.129, 0.732, 0.139],
                                                [0.303, 0.479, 0.219],
                                                [0.514, 0.178, 0.308],
                                                [0.577, 0.129, 0.294]]))
    known_batch_labels = np.array([1, 1, 0, 2])

    def create_relu(self) -> Layers.LeakyReLU:
        relu = Layers.LeakyReLU(3, 5)
        relu.weights = np.array([[0.1, -0.2, 0.3, -0.4, 0.5],
                                [-0.6, 0.7, -0.8, 0.9, -0.1],
                                [0.11, -0.12, 0.13, -0.14, 0.15]])
        return relu

    def create_sm(self) -> Layers.Softmax:
        sm = Layers.Softmax(5, 3)
        sm.weights = np.array([[0.1, -0.6, 0.11],
                                [-0.2, 0.7, -0.12],
                                [0.3, -0.8, 0.13],
                                [-0.4, 0.9, -0.14],
                                [0.5, -0.1, 0.15]])
        return sm
    
    def test_add_layer(self) -> None:
        nn = Network.Neural_Network()
        nn.add_layer(self.create_relu())

        assert len(nn.layers) == 1
        assert isinstance(nn.layers[0], Layers.LeakyReLU)

    def test_add_loss(self) -> None:
        nn = Network.Neural_Network()
        nn.add_loss(self.loss)

        assert isinstance(nn.loss, Loss.Cross_Entropy)

    def test_train(self) -> None:
        nn = Network.Neural_Network()

        nn.add_layer(self.create_relu())
        nn.add_layer(self.create_sm())
        nn.add_loss(self.loss)

        nn.train(self.known_single_input, self.known_single_label, categories = 3, batch_size = 1, epochs = 1)
        assert np.round(nn.loss.loss,3) == self.known_single_xe_loss

    def test_batch_train(self) -> None:
        nn = Network.Neural_Network()

        nn.add_layer(self.create_relu())
        nn.add_layer(self.create_sm())
        nn.add_loss(self.loss)
        
        nn.train(self.known_batch_input, self.known_batch_labels, categories = 3, batch_size = 4, epochs = 1)
        assert np.round(nn.loss.loss,3) == self.known_batch_xe_loss

class Test_Network_With_Bias:

    loss = Loss.Cross_Entropy()

    known_single_input = np.array([[1,2,3]])
    known_single_label = np.array([1])
    known_single_xe_loss = 0.347

    known_batch_input = np.array([[1, 2, 3],
                                [3, 2, 1],
                                [2, 1, 3],
                                [3, 1, 2]])
    known_batch_labels = np.array([1, 1, 0, 2])
    known_batch_xe_loss = 0.815

    def create_relu(self) -> Layers.LeakyReLU:
        relu = Layers.LeakyReLU(3, 5, bias=True)
        relu.weights = np.array([[0.2, -0.8, 0.3, -0.7, 0.4],
                                [0.1, -0.2, 0.3, -0.4, 0.5],
                                [-0.6, 0.7, -0.8, 0.9, -0.1],
                                [0.11, -0.12, 0.13, -0.14, 0.15]]) 
        return relu

    def create_sm(self) -> Layers.Softmax:
        sm = Layers.Softmax(5, 3, bias=True)
        sm.weights = np.array([[-0.9, 0.8, -0.7],
                            [0.1, -0.6, 0.11],
                            [-0.2, 0.7, -0.12],
                            [0.3, -0.8, 0.13],
                            [-0.4, 0.9, -0.14],
                            [0.5, -0.1, 0.15]])
        return sm

    def test_train(self) -> None:
        nn = Network.Neural_Network()

        nn.add_layer(self.create_relu())
        nn.add_layer(self.create_sm())
        nn.add_loss(self.loss)

        nn.train(self.known_single_input, self.known_single_label, categories = 3, batch_size = 1, epochs = 1)
        assert np.round(nn.loss.loss,3) == self.known_single_xe_loss

    def test_batch_train(self) -> None:
        nn = Network.Neural_Network()

        nn.add_layer(self.create_relu())
        nn.add_layer(self.create_sm())
        nn.add_loss(self.loss)
        
        nn.train(self.known_batch_input, self.known_batch_labels, categories = 3, batch_size = 4, epochs = 1)
        assert np.round(nn.loss.loss,3) == self.known_batch_xe_loss