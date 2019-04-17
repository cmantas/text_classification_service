from keras.preprocessing.text import Tokenizer
from ml_models import TextModel


class WordLevelModel(TextModel):
    @classmethod
    def tokenizer(cls, texts):
        tokenizer = Tokenizer(cls.VOCAB_SIZE,
                              filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~',
                              lower=True)
        tokenizer.fit_on_texts(texts)
        return tokenizer
