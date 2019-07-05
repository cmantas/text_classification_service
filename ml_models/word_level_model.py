from keras.preprocessing.text import Tokenizer
from ml_models import TextModel

DELIMITERS = "!\"#$%&()*+,-./:;<=>?@[\]^_`{|}~'“”—…«»′"


class WordLevelModel(TextModel):
    MAX_VOCAB_SIZE = 30_000

    @classmethod
    def tokenizer(cls, texts):
        tokenizer = Tokenizer(filters=DELIMITERS, lower=True)
        tokenizer.fit_on_texts(texts)

        corpus_vocabulary_size = len(tokenizer.word_index)
        # If our vocabulary size exceeds the maximum allowed vocab size we
        # need to limit it to a smaller number. Otherwise we just use it as
        # `num_words` for our tokenize
        tokenizer.num_words = min(cls.MAX_VOCAB_SIZE, corpus_vocabulary_size)

        return tokenizer

    def vocabulary_size(self):
        return self.tokenizer.num_words


    def describe(self):
        super_dict = super().describe()
        return {**super_dict, 'vocab_size': self.vocabulary_size()}
