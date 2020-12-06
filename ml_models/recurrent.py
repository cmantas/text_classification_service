from tensorflow.keras.layers import LSTM, SimpleRNN, \
     GlobalMaxPooling1D, Dense, Embedding
from tensorflow.keras.models import Sequential
from abc import ABC
from ml_models import BinaryModel, MultiClassModel, SequenceModel,\
    CharacterLevelModel

__all__ = ['LSTMModel', 'LSTMBinaryModel', 'LSTMMultiClassModel',
           'CUDALSTMModel', 'DropoutLSTMModel', 'SmallerLSTMModel',
           'LSTMMaxPooledModel', 'LSTMCharacterMultiClassModel']

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


class LSTMCharacterMultiClassModel(CharacterLevelModel, LSTMModel):
    pass


class CUDALSTMModel(LSTMModel):
    # Override the LSTMModel's recurrent_layers so as to use the CUDA-specific
    # layer (for performance)
    @classmethod
    def recurrent_layers(cls):
        return [LSTM(128)]


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

class LSTMMaxPooledModel(LSTMModel, MultiClassModel):
    def model_description(self):
        layers = [
            Embedding(self.vocabulary_size(), self.EMBEDDING_DIMENTION,
                      input_length=self.MAX_SEQ_LEN, mask_zero=False), # masking not supported by GlobalMaxPooling
            LSTM(128, return_sequences = True),
            GlobalMaxPooling1D(),
            # The last layer of the model will be a dense layer with a size
            # equal to the number of layers
            Dense(self.num_labels(), activation=self.ACTIVATION)
        ]
        model = Sequential(layers)

        model.compile(loss=self.LOSS_FUNCTION,
                      optimizer='adam', metrics=['accuracy'])
        return model
