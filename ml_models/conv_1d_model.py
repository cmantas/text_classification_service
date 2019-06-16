from keras.layers import Dense, Embedding, Conv1D, GlobalMaxPooling1D, Dropout
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences

from ml_models import WordLevelModel, MultiClassModel


class Conv1DModel(WordLevelModel):
    VOCAB_SIZE = 30000
    BATCH_SIZE = 4000
    MAX_SEQ_LEN = 100

    @classmethod
    def hidden_layers(cls):
        embedding_dim = 128
        num_filters = 20
        filter_size = 5
        layers = [
            Embedding(cls.VOCAB_SIZE, embedding_dim, input_length=cls.MAX_SEQ_LEN),
            Conv1D(num_filters, filter_size, activation='relu'),
            GlobalMaxPooling1D(),
            # Dropout(0.1),
            Dense(20, activation='relu'),
            Dropout(0.05),
        ]
        return layers

    def vectorize_texts(self, texts):
        seqs = self.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(seqs, maxlen=self.MAX_SEQ_LEN)
        return X


