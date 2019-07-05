from ml_models import *
from keras.layers import *
from keras import regularizers


class FFDropoutModel(FeedForwardMultiClassModel):
    def hidden_layers(self):
        return [
            Dense(256, input_shape=(self.vocabulary_size(),)),
            LeakyReLU(alpha=0.1),
            Dropout(0.15)
        ]


class DeeperFFDropoutModel(FeedForwardMultiClassModel):
    def hidden_layers(self):
        return [
            Dense(256, input_shape=(self.vocabulary_size(),)),
            LeakyReLU(alpha=0.1),
            Dropout(0.1),
            Dense(128, input_shape=(self.vocabulary_size(),)),
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

class Bigger1DCNN(SequenceModel, MultiClassModel):
    num_filters = 64
    weight_decay = 1e-4
    BATCH_SIZE = 1000

    @classmethod
    def recurrent_layers(cls):
        #num_filters = 20
        num_filters = 5
        weight_decay = 1e-4
        return [
            Conv1D(num_filters, 7, activation='relu', padding='same'),
            MaxPooling1D(2),
            Conv1D(num_filters, 7, activation='relu', padding='same'),
            GlobalMaxPooling1D(),
            Dropout(0.05),
            Dense(32, activation='relu',
                  kernel_regularizer=regularizers.l2(weight_decay))
        ]

