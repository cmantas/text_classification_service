from ml_models import MultiClassModel
from ml_models import SequenceModel
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import LSTM, Embedding, Lambda

from tensorflow import one_hot

DELIMITERS = "!\"#$%&()*+,-./:;<=>?@[]^_`{|}~'“”—…«»′"


class CharacterLevelModel(MultiClassModel, SequenceModel):
    MAX_VOCAB_SIZE = 50
    MAX_SEQ_LEN = 150
    BATCH_SIZE = 2000

    def embedding_layer(self):
        return Lambda(lambda x: one_hot(x, self.vocabulary_size()))

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
