from keras.models import load_model
import pickle
from abc import ABC
from random import shuffle

def batcher(phrases, batch_size):
  for i in range(0, len(phrases), batch_size):
    frrom = i
    to = i+batch_size
    yield phrases[frrom:to]


class TextModel(ABC):
    VOCAB_SIZE = 30000
    BATCH_SIZE = 4000

    def __init__(self, model, tokenizer, encoder):
        self.model = model
        self.tokenizer = tokenizer
        self.label_encoder = encoder

    def predict_category(self, txt):
        return self.predict_categories([txt])[0]

    @classmethod
    def create_from_files(cls, model_fname, tokenizer_fname, encoder_fname):
        model = load_model(model_fname)
        tokenizer = pickle.load(open(tokenizer_fname, 'rb'))
        label_encoder = pickle.load(open(encoder_fname, 'rb'))
        return cls(model, tokenizer, label_encoder)

    @classmethod
    def create_from_corpus(cls, texts, labels):
        tok = cls.tokenizer(texts)
        enc = cls.encoder(labels)
        model = cls.model_description(enc)
        return cls(model, tok, enc)

    def vectorize_batch(self, batch):
        texts, cats = zip(*batch)
        X = self.vectorize_texts(texts)
        Y = self.label_encoder.transform(cats)
        return (X, Y)

    def training_gen(self, texts):
        while True:
            shuffle(texts)
            for batch in batcher(texts, self.BATCH_SIZE):
                X, Y = self.vectorize_batch(batch)
                yield (X, Y)

    def save_to_files(self, model_fname, tokenizer_fname, encoder_fname):
        self.model.save(model_fname)
        pickle.dump(self.tokenizer, open(tokenizer_fname, 'wb'))
        pickle.dump(self.label_encoder, open(encoder_fname, 'wb'))