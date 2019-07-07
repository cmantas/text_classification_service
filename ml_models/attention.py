from keras.layers import LSTM, SpatialDropout1D, Bidirectional, Conv1D, \
    MaxPooling1D, GlobalMaxPooling1D
from abc import ABC
from ml_models import SequenceModel, MultiClassModel, BinaryModel, FastTextModel
from ml_models.layers import Attention


class LSTMAttentionModel(SequenceModel, ABC):
    BATCH_SIZE = 1000

    @classmethod
    def recurrent_layers(cls):
        recurrent_dropout = 0
        dropout = 0
        spacial_dropout = 0.05

        layers = [
            #SpatialDropout1D(spacial_dropout),
            Bidirectional(
                LSTM(128, return_sequences=True, dropout=dropout,
                     recurrent_dropout=recurrent_dropout)
            ),
            Attention()
        ]

        return layers


class LSTMAttentionMulticlassModel(LSTMAttentionModel, MultiClassModel):
    pass


class LSTMAttentionBinaryModel(LSTMAttentionModel, BinaryModel):
    pass


class CNNAttentionModel(SequenceModel, MultiClassModel):
    BATCH_SIZE = 1000

    @classmethod
    def recurrent_layers(cls):
        num_filters = 5

        layers = [
            Conv1D(num_filters, 5, activation='relu', padding='same'),
            MaxPooling1D(2),
            Conv1D(num_filters, 5, activation='relu', padding='same'),
            Attention()
        ]

        return layers


class CNNAttentionMulticlassModel(CNNAttentionModel, MultiClassModel):
    pass


class CNNAttentionBinaryModel(CNNAttentionModel, BinaryModel):
    pass


class CNNAttentionMulticlassFasttextModel(CNNAttentionModel, FastTextModel,
                                          MultiClassModel):
    pass