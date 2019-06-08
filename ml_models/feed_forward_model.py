from keras.layers import Dense
from keras.models import Sequential

from ml_models import WordLevelModel


class FeedForwardModel(WordLevelModel):
    VOCAB_SIZE = 30000
    BATCH_SIZE = 4000

    @classmethod
    def model_description(cls, num_labels):
        vocab_size = cls.VOCAB_SIZE
        layers = [
            Dense(128, input_shape=(vocab_size,), activation='relu'),
            Dense(num_labels, activation=cls.ACTIVATION)
        ]
        model = Sequential(layers)

        model.compile(loss=cls.LOSS_FUNCTION,
                      optimizer='adam', metrics=['accuracy'])
        return model

    def vectorize_texts(self, texts):
        return self.tokenizer.texts_to_matrix(texts, mode='count')


