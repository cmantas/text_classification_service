from keras.layers import Dense, Embedding, Conv1D, GlobalMaxPooling1D, Dropout
from keras.preprocessing.sequence import pad_sequences

from ml_models import SequenceModel


class Conv1DModel(SequenceModel):
    BATCH_SIZE = 4000
    # we can handle a bigger sequence size with a convolutional architecture
    MAX_SEQ_LEN = 40

    @classmethod
    def recurrent_layers(cls):
        num_filters = 20
        filter_size = 5
        return [
            Conv1D(num_filters, filter_size, activation='relu'),
            GlobalMaxPooling1D(),
            # Dropout(0.1),
            Dense(20, activation='relu'),
            Dropout(0.05),
        ]
