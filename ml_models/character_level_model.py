DELIMITERS = "!\"#$%&()*+,-./:;<=>?@[\]^_`{|}~'“”—…«»′"
from ml_models import MultiClassModel
from keras.preprocessing.text import Tokenizer
from keras.layers import LSTM, Embedding
from keras.preprocessing.sequence import pad_sequences


class CharacterLevelModel(MultiClassModel):
    MAX_VOCAB_SIZE = 100
    MAX_SEQ_LEN = 150
    BATCH_SIZE = 2000

    def vectorize_texts(self, texts):
        seqs = self.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(seqs, maxlen=self.MAX_SEQ_LEN)
        return X  # to_categorical(X, num_classes=self.vocabulary_size())

    def recurrent_layers(self):
        return [LSTM(128)]  # , input_shape=(None, self.vocabulary_size()))]

    def hidden_layers(self):
        return [
            Embedding(self.vocabulary_size(), 20,
                      input_length=self.MAX_SEQ_LEN),
            *self.recurrent_layers()
        ]

    @classmethod
    def tokenizer(cls, texts):
        tokenizer = Tokenizer(filters=DELIMITERS, lower=True, char_level=True)
        tokenizer.fit_on_texts(texts)

        corpus_vocabulary_size = len(tokenizer.word_index)
        # If our vocabulary size exceeds the maximum allowed vocab size we
        # need to limit it to a smaller number. Otherwise we just use it as
        # `num_words` for our tokenize
        tokenizer.num_words = min(cls.MAX_VOCAB_SIZE, corpus_vocabulary_size)

        return tokenizer

    def vocabulary_size(self):
        return self.tokenizer.num_words


model_class = CharacterLevelModel