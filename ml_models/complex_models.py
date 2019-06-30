from ml_models import *
from keras.layers import *


class FFDropoutModel(FeedForwardMultiClassModel):
    @classmethod
    def hidden_layers(cls):
        return [
            Dense(256, input_shape=(cls.VOCAB_SIZE,)),
            LeakyReLU(alpha=0.1),
            Dropout(0.15)
        ]


class DeeperFFDropoutModel(FeedForwardMultiClassModel):
    @classmethod
    def hidden_layers(cls):
        return [
            Dense(256, input_shape=(cls.VOCAB_SIZE,)),
            LeakyReLU(alpha=0.1),
            Dropout(0.1),
            Dense(128, input_shape=(cls.VOCAB_SIZE,)),
            LeakyReLU(alpha=0.1),
            Dropout(0.1)
        ]


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
