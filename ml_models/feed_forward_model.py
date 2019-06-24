from keras.layers import Dense
from keras.models import Sequential

from ml_models import WordLevelModel


class FeedForwardModel(WordLevelModel):
    BATCH_SIZE = 4000

    @classmethod
    def hidden_layers(cls):
        return [
            Dense(128, input_shape=(cls.VOCAB_SIZE,), activation='relu')
        ]

    def vectorize_texts(self, texts):
        return self.tokenizer.texts_to_matrix(texts, mode='count')


