from abc import ABC
from keras.layers import Embedding
from os.path import isfile
import codecs
from tqdm import tqdm
import numpy as np
from ml_models import SequenceModel, MultiClassModel, LSTMModel
import pickle

__all__ = ['FastTextModel']

#FASTTEXT_EMBEDDINGS_FILE = 'data/embeddings/wiki-news-300d-1M-subword.vec'
FASTTEXT_EMBEDDINGS_FILE = 'data/embeddings/wiki.simple.vec'


class FastTextModel(SequenceModel, ABC):
    TRAINABLE_EMBEDDINGS = True
    BATCH_SIZE = 1500

    def __init__(self, tokenizer, encoder):
        self.tokenizer = tokenizer
        self.embeddings_matrix = self.create_embedding_matrix()
        super().__init__(tokenizer, encoder)

    def embedding_layer(self):
        return Embedding(self.vocabulary_size(), self.EMBEDDING_DIMENTION,
                         weights=[self.embeddings_matrix],
                         input_length=self.MAX_SEQ_LEN,
                         trainable=self.TRAINABLE_EMBEDDINGS)

    @staticmethod
    def embeddings_index():
        """Loads the embeddings from a file and creates an index from a token
           to its embedding vector.
           The index contains all tokens for which there are available 
           embeddings.
        """
        pickle_cache = FASTTEXT_EMBEDDINGS_FILE + '.pickle'
        # check if there is a version of the embeddings file saved a pickle
        if isfile(pickle_cache):
            print('Using cached embedding index')
            return pickle.load(open(pickle_cache, 'rb'))

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

        # save the embedding index as a pickle for future use
        pickle.dump(embeddings_index, open(pickle_cache, 'wb'))
        return embeddings_index

    EMBEDDINGS_INDEX = embeddings_index.__func__()
    # We can deduce the embedding dim. from the size of the loaded embeddings
    EMBEDDING_DIMENTION = len(list(EMBEDDINGS_INDEX.values())[0])

    def create_embedding_matrix(self):
        """It looks up all the tokens of the model's tokenizer (aka. the top
           tokens of our corpus) and creates and matrix embedding the ones that
           are known from our embeddings corpus to their respective vectors 
           (the rest are embedded to zero)
        """
        print('creating embedding matrix')
        words_not_found = set()
        nb_words = self.vocabulary_size()
        embed_dim = len(list(self.EMBEDDINGS_INDEX.values())[0])

        if self.TRAINABLE_EMBEDDINGS:
            # If we are to train the embedding weights, we need to initialize
            # them randomly
            embedding_matrix = np.random.random_sample((nb_words, embed_dim))
        else:
            # If we are not to train the embedding weights, we will have unknown
            # tokens have zero embedding vectors
            embedding_matrix = np.zeros((nb_words, embed_dim))

        for word, i in self.tokenizer.word_index.items():
            if i >= nb_words:
                print('Reached the limit of', nb_words,
                      '. The tokenizer has a total of ',
                      len(self.tokenizer.word_index), 'words')
                break
            embedding_vector = self.EMBEDDINGS_INDEX.get(word)
            if (embedding_vector is not None) and len(embedding_vector) > 0:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
            else:
                words_not_found.add(word)

        print("# words not found: ", len(words_not_found))
        print("sample words not found: ", np.random.choice(list(words_not_found), 200))
        return embedding_matrix
