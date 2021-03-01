import Loss
import numpy as np

class Test_Base:

    known_1D_label = np.array([1])
    known_1hot_label = np.array([[0,1,0]])

    known_categories = 3

    known_batch_labels = np.array([1, 1, 0, 2])
    known_batch_1hot_labels = np.array([[0,1,0],
                                        [0,1,0],
                                        [1,0,0],
                                        [0,0,1]])

     # Base is an abstract class, so create a new class
    class Conc_Base(Loss.Base):
        pass

    def test_one_hot(self):
        base = self.Conc_Base()
        one_hot_target = base.one_hot(self.known_1D_label, self.known_categories)
        assert one_hot_target.shape[1] == self.known_categories
        assert np.array_equal(one_hot_target, self.known_1hot_label)

        one_hot_target = base.one_hot(self.known_batch_labels, self.known_categories)
        assert one_hot_target.shape[0] == self.known_batch_labels.shape[0]
        assert np.array_equal(one_hot_target, self.known_batch_1hot_labels)

class Test_Cross_Entropy:

    known_1D_estimates = np.transpose(np.array([[0.129, 0.732, 0.139]]))
    known_1D_label = np.array([[1]])

    known_categories = 3
    
    known_loss = 0.312

    known_batch_estimates = np.transpose(np.array([[0.129, 0.732, 0.139],
                                                    [0.303, 0.479, 0.219],
                                                    [0.514, 0.178, 0.308],
                                                    [0.577, 0.129, 0.294]]))
    known_batch_labels = np.array([1, 1, 0, 2])
    known_batch_loss = 0.734

    def test_calc_loss(self):
        xe_loss = Loss.Cross_Entropy()
        xe_loss.calc_loss(self.known_1D_label, self.known_categories, self.known_1D_estimates)
        assert np.round(xe_loss.loss, 3) == self.known_loss

    def test_batch_calc_loss(self):
        xe_loss = Loss.Cross_Entropy()
        xe_loss.calc_loss(self.known_batch_labels, self.known_categories, self.known_batch_estimates)
        assert np.round(xe_loss.loss, 3) == self.known_batch_loss