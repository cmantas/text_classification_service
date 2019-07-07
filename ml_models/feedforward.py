from keras.layers import Dense, Dropout, LeakyReLU

from ml_models import WordLevelModel, BinaryModel, MultiClassModel


class FeedForwardModel(WordLevelModel):
    BATCH_SIZE = 8000

    def hidden_layers(self):
        return [
            Dense(128, input_shape=(self.vocabulary_size(),),
                  activation='relu')
        ]

    def vectorize_texts(self, texts):
        return self.tokenizer.texts_to_matrix(texts, mode='tfidf')


class FeedForwardBinaryModel(FeedForwardModel, BinaryModel):
    pass


class FeedForwardMultiClassModel(FeedForwardModel, MultiClassModel):
    pass

class FFDropoutModel(FeedForwardMultiClassModel):
    def hidden_layers(self):
        return [
            Dense(256, input_shape=(self.vocabulary_size(),)),
            LeakyReLU(alpha=0.1),
            Dropout(0.15)
        ]


class DeeperFFDropoutModel(FeedForwardMultiClassModel):
    def hidden_layers(self):
        return [
            Dense(256, input_shape=(self.vocabulary_size(),)),
            LeakyReLU(alpha=0.1),
            Dropout(0.1),
            Dense(128, input_shape=(self.vocabulary_size(),)),
            LeakyReLU(alpha=0.1),
            Dropout(0.1)
        ]
