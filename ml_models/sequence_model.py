from keras.layers import Embedding
from abc import abstractmethod
from keras.preprocessing.sequence import pad_sequences

from ml_models import WordLevelModel


class SequenceModel(WordLevelModel):
    EMBEDDING_DIMENTION = 128
    MAX_SEQ_LEN = 50
    BATCH_SIZE = 1000

    @classmethod
    @abstractmethod
    def recurrent_layers(cls):
        pass

    def embedding_layer(self):
        return Embedding(self.vocabulary_size(), self.EMBEDDING_DIMENTION,
                         input_length=self.MAX_SEQ_LEN, mask_zero=True)

    def hidden_layers(self):
        layers = [
            # All sequence models' first layer is an embedding layer
            self.embedding_layer(),
            # Then one or more recurrent layers follow
            *self.recurrent_layers(),
        ]
        return layers

    def vectorize_texts(self, texts):
        seqs = self.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(seqs, maxlen=self.MAX_SEQ_LEN)
        return X


