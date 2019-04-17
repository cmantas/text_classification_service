from keras.layers import Dense, LSTM, Embedding
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences

from ml_models import TextModel, WordLevelModel, MultiClassModel


class LSTMModel(MultiClassModel, WordLevelModel):
    VOCAB_SIZE = 30000
    BATCH_SIZE = 500
    EMBEDDING_DIMENTION = 128
    MAX_SEQ_LEN = 100

    @classmethod
    def model_description(cls, encoder):
        num_labels = len(encoder.classes_)
        layers = [
            Embedding(cls.VOCAB_SIZE, cls.EMBEDDING_DIMENTION,
                      input_length=cls.MAX_SEQ_LEN),
            #LSTM(128, return_sequences=True),
            LSTM(128),
            Dense(num_labels, activation='softmax')
        ]

        model = Sequential(layers)
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])

        return model

    def vectorize_texts(self, texts):
        seqs = self.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(seqs, maxlen=self.MAX_SEQ_LEN)
        return X

