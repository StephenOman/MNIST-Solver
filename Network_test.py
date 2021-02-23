import Network
import Layers
import Loss

class Test_Network:
    
    # Test Data
    number_inputs = 3
    number_nodes = 5

    relu = Layers.LeakyReLU(number_inputs, number_nodes)
    sm = Layers.Softmax(number_inputs, number_nodes)
    loss = Loss.Cross_Entropy()

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
        pass # ToDo
