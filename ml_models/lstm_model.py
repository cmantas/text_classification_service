from keras.layers import LSTM
from abc import ABC
from ml_models import SequenceModel


class LSTMModel(SequenceModel, ABC):
    BATCH_SIZE = 500

    @classmethod
    def recurrent_layers(cls):
        return [LSTM(128)]


