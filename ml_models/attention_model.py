from keras.layers import LSTM, SpatialDropout1D, Bidirectional
from abc import ABC
from ml_models import SequenceModel, MultiClassModel, BinaryModel
from ml_models.layers import Attention


class AttentionModel(SequenceModel, ABC):
    BATCH_SIZE = 500

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


class AttentionMulticlassModel(AttentionModel, MultiClassModel):
    pass


class AttentionBinaryModel(AttentionModel, BinaryModel):
    pass


