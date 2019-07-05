from keras.layers import Dense
from keras.models import Sequential

from ml_models import WordLevelModel


class FeedForwardModel(WordLevelModel):
    BATCH_SIZE = 4000

    def hidden_layers(self):
        return [
            Dense(128, input_shape=(self.vocabulary_size(),),
                  activation='relu')
        ]

    def vectorize_texts(self, texts):
        return self.tokenizer.texts_to_matrix(texts, mode='tfidf')


