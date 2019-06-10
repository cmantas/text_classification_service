from keras.layers import Dense, LSTM, Embedding
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences

from ml_models import WordLevelModel, MultiClassModel


class LSTMModel(WordLevelModel):
    VOCAB_SIZE = 30000
    BATCH_SIZE = 500
    EMBEDDING_DIMENTION = 128
    MAX_SEQ_LEN = 100

    @classmethod
    def recurrent_layers(cls):
        return [LSTM(128)]

    @classmethod
    def model_description(cls, num_labels):
        layers = [
            Embedding(cls.VOCAB_SIZE, cls.EMBEDDING_DIMENTION,
                      input_length=cls.MAX_SEQ_LEN),
            *cls.recurrent_layers(),
            Dense(num_labels, activation=cls.ACTIVATION)
        ]

        model = Sequential(layers)
        model.compile(loss=cls.LOSS_FUNCTION,
                      optimizer='adam', metrics=['accuracy'])

        return model

    def vectorize_texts(self, texts):
        seqs = self.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(seqs, maxlen=self.MAX_SEQ_LEN)
        return X


