from keras.layers import LSTM, CuDNNLSTM, SimpleRNN
from abc import ABC
from ml_models import BinaryModel, MultiClassModel, SequenceModel

__all__ = ['LSTMModel', 'LSTMBinaryModel', 'LSTMMultiClassModel',
           'CUDALSTMModel', 'DropoutLSTMModel', 'SmallerLSTMModel']

class LSTMModel(SequenceModel, ABC):
    @classmethod
    def recurrent_layers(cls):
        return [LSTM(128)]

class UnembeddedLSTMMultiClassModel(LSTMModel, MultiClassModel):
    def hidden_layers(self):
        return [*self.recurrent_layers()]

class LSTMBinaryModel(LSTMModel, BinaryModel):
    pass


class LSTMMultiClassModel(LSTMModel, MultiClassModel):
    pass


class CUDALSTMModel(LSTMModel):
    # Override the LSTMModel's recurrent_layers so as to use the CUDA-specific
    # layer (for performance)
    @classmethod
    def recurrent_layers(cls):
        return [CuDNNLSTM(128)]


class DropoutLSTMModel(LSTMMultiClassModel):
    @classmethod
    def recurrent_layers(cls):
        return [LSTM(128, dropout=.05)]


class SmallerLSTMModel(LSTMMultiClassModel):
    # Override the LSTMModel's recurrent_layers so as to use the CUDA-specific
    # layer (for performance)
    @classmethod
    def recurrent_layers(cls):
        return [LSTM(64)]


class RNNModel(SequenceModel, ABC):
    @classmethod
    def recurrent_layers(cls):
        return [SimpleRNN(128)]


class RNNBinaryModel(RNNModel, BinaryModel):
    pass


class RNNMultiClassModel(RNNModel, MultiClassModel):
    pass
