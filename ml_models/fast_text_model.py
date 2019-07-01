from keras.layers import LSTM, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense
from keras.models import Sequential
import codecs
from tqdm import tqdm
import numpy as np
from ml_models import WordLevelModel, MultiClassModel

#FASTTEXT_EMBEDDINGS_FILE = 'data/embeddings/wiki-news-300d-1M-subword.vec'
FASTTEXT_EMBEDDINGS_FILE = 'data/embeddings/wiki.simple.vec'


class FastTextModel(WordLevelModel, MultiClassModel):

    BATCH_SIZE = 1500
    MAX_SEQ_LEN = 50

    def __init__(self, tokenizer, encoder):
        self.tokenizer = tokenizer
        self.embeddings_matrix = self.create_embedding_matrix()
        super().__init__(tokenizer, encoder)

    @classmethod
    def recurrent_layers(cls):
        return [LSTM(128)]


    def hidden_layers(self):
        layers = [
            # All LSTM models' first layer is an Embedding layer
            Embedding(self.VOCAB_SIZE, self.EMBEDDING_DIMENTION,
                      weights=[self.embeddings_matrix],
                      input_length=self.MAX_SEQ_LEN, trainable=False),
            # Then one or more recurrent layers follow
            *self.recurrent_layers(),
        ]
        return layers

    def vectorize_texts(self, texts):
        seqs = self.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(seqs, maxlen=self.MAX_SEQ_LEN)
        return X

    @staticmethod
    def embeddings_index():
        embeddings_index = {}
        print('Building the embedding index')
        with codecs.open(FASTTEXT_EMBEDDINGS_FILE, encoding='utf-8') as f:
            for line in tqdm(f):
                values = line.rstrip().rsplit(' ')
                if len(values) == 2:
                    continue
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        return embeddings_index



    EMBEDDINGS_INDEX = embeddings_index.__func__()
    EMBEDDING_DIMENTION = len(list(EMBEDDINGS_INDEX.values())[0])

    def create_embedding_matrix(self):
        print('creating embedding matrix')
        words_not_found = set()
        nb_words = self.VOCAB_SIZE
        embed_dim = len(list(self.EMBEDDINGS_INDEX.values())[0])
        embedding_matrix = np.zeros((nb_words, embed_dim))
        for word, i in self.tokenizer.word_index.items():
            if i >= nb_words:
                continue
            embedding_vector = self.EMBEDDINGS_INDEX.get(word)
            if (embedding_vector is not None) and len(embedding_vector) > 0:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
            else:
                words_not_found.add(word)
        print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
        print("sample words not found: ", np.random.choice(list(words_not_found), 10))
        return embedding_matrix

    def model_description(self, num_labels):
        layers = [
            *self.hidden_layers(),
            # The last layer of the model will be a dense layer with a size
            # equal to the number of layers
            Dense(num_labels, activation=self.ACTIVATION)
        ]
        model = Sequential(layers)

        model.compile(loss=self.LOSS_FUNCTION,
                      optimizer='adam', metrics=['accuracy'])
        return model
