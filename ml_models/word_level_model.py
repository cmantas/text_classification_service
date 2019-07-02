from keras.preprocessing.text import Tokenizer
from ml_models import TextModel

DELIMITERS = "!\"#$%&()*+,-./:;<=>?@[\]^_`{|}~'“”—…«»′"


class WordLevelModel(TextModel):
    VOCAB_SIZE = 30_000

    @classmethod
    def tokenizer(cls, texts):
        tokenizer = Tokenizer(cls.VOCAB_SIZE, filters=DELIMITERS, lower=True)
        tokenizer.fit_on_texts(texts)
        real_vocab_size = len(tokenizer.word_index)

        if real_vocab_size < cls.VOCAB_SIZE:
            print('You are using a VOCAB_SIZE of', cls.VOCAB_SIZE,
                  'when you only have', real_vocab_size, 'tokens')
        return tokenizer

    def describe(self):
        super_dict = super().describe()
        return {**super_dict, 'vocab_size': self.VOCAB_SIZE}
