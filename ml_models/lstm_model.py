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
    def hidden_layers(cls):
        layers = [
            # All LSTM models' first layer is an Embedding layer
            Embedding(cls.VOCAB_SIZE, cls.EMBEDDING_DIMENTION,
                      input_length=cls.MAX_SEQ_LEN),
            # Then one or more recurrent layers follow
            *cls.recurrent_layers(),
        ]
        return layers

    def vectorize_texts(self, texts):
        seqs = self.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(seqs, maxlen=self.MAX_SEQ_LEN)
        return X


