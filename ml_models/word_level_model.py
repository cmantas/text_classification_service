from keras.preprocessing.text import Tokenizer
from ml_models import TextModel


class WordLevelModel(TextModel):
    VOCAB_SIZE = 30_000

    @classmethod
    def tokenizer(cls, texts):
        tokenizer = Tokenizer(cls.VOCAB_SIZE,
                              filters="!\"#$%&()*+,-./:;<=>?@[\]^_`{|}~'“”—",
                              lower=True)
        tokenizer.fit_on_texts(texts)
        return tokenizer

    def describe(self):
        super_dict = super().describe()
        return {**super_dict, 'vocab_size': self.VOCAB_SIZE}
