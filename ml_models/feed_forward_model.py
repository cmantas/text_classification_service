from keras.layers import Dense
from keras.models import Sequential

from ml_models import WordLevelModel, MultiClassModel


class FeedForwardModel(MultiClassModel, WordLevelModel):
    VOCAB_SIZE = 30000
    BATCH_SIZE = 4000

    @classmethod
    def model_description(cls, encoder):
        vocab_size = cls.VOCAB_SIZE
        num_labels = len(encoder.classes_)
        layers = [
            Dense(128, input_shape=(vocab_size,), activation='relu'),
            Dense(num_labels, activation='softmax')
        ]
        model = Sequential(layers)

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
        return model

    def vectorize_texts(self, texts):
        return self.tokenizer.texts_to_matrix(texts, mode='count')


