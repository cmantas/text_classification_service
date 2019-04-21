from numpy import array

class BinaryModel:
    ACTIVATION = 'sigmoid'
    LOSS_FUNCTION = 'binary_crossentropy'

    def num_labels(self):
        return 1

    def vectorize_labels(self, labels):
        return array(labels)
