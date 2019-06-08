from numpy import array
from ml_models import TextModel

class BinaryModel(TextModel):
    ACTIVATION = 'sigmoid'
    LOSS_FUNCTION = 'binary_crossentropy'

    def num_labels(self):
        return 1

    def vectorize_labels(self, labels):
        return array(labels)

    @classmethod
    def encoder(cls, texts):
        """A binary model does not need a labelEncoder"""
        None
