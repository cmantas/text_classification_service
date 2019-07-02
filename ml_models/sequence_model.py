from keras.layers import Dense, LSTM, Embedding
from abc import abstractmethod
from keras.preprocessing.sequence import pad_sequences

from ml_models import WordLevelModel, MultiClassModel


class SequenceModel(WordLevelModel):
    EMBEDDING_DIMENTION = 128
    MAX_SEQ_LEN = 20

    @classmethod
    @abstractmethod
    def recurrent_layers(cls):
        pass

    @classmethod
    def embedding_layer(cls):
        return Embedding(cls.VOCAB_SIZE, cls.EMBEDDING_DIMENTION,
                         input_length=cls.MAX_SEQ_LEN)

    @classmethod
    def hidden_layers(cls):
        layers = [
            # All sequence models' first layer is an embedding layer
            cls.embedding_layer(),
            # Then one or more recurrent layers follow
            *cls.recurrent_layers(),
        ]
        return layers

    def vectorize_texts(self, texts):
        seqs = self.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(seqs, maxlen=self.MAX_SEQ_LEN)
        return X


