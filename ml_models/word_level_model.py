from keras.preprocessing.text import Tokenizer
from abc import ABC


class WordLevelModel(ABC):
    @classmethod
    def tokenizer(cls, texts):
        tokenizer = Tokenizer(cls.VOCAB_SIZE,
                              filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~',
                              lower=True)
        tokenizer.fit_on_texts(texts)
        return tokenizer
