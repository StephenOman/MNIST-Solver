import Loss
import numpy as np

class Test_Base:

    known_target = np.array([[1]])
    known_1hot_target = np.transpose(np.array([[0,1,0]]))
    known_categories = 3
    known_1D_inputs = np.transpose(np.array([[0.129, 0.732, 0.139]]))

     # Base is an abstract class, so create a new class
    class Conc_Base(Loss.Base):
        pass

    def test_one_hot(self):
        base = self.Conc_Base()
        one_hot_target = base.one_hot(self.known_target, self.known_categories)
        assert one_hot_target[0].shape[0] == self.known_categories
        assert np.array_equal(np.transpose(one_hot_target), self.known_1hot_target)

class Test_Cross_Entropy:

    known_1D_estimates = np.transpose(np.array([[0.129, 0.732, 0.139]]))
    known_categories = 3
    known_1D_targets = np.transpose(np.array([[0,1,0]]))
    known_loss = 0.312

    def test_calc_loss(self):
        xe_loss = Loss.Cross_Entropy()
        xe_loss.calc_loss(self.known_1D_targets, self.known_categories, self.known_1D_estimates)
        assert np.round(xe_loss.loss, 3) 