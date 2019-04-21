from keras.models import load_model
import pickle
from abc import ABC, abstractmethod
from random import shuffle
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.style.use('seaborn')

def batcher(phrases, batch_size):
  for i in range(0, len(phrases), batch_size):
    frrom = i
    to = i+batch_size
    yield phrases[frrom:to]


class TextModel(ABC):
    def __init__(self, tokenizer, encoder):
        self.tokenizer = tokenizer
        self.label_encoder = encoder
        num_classes = self.num_labels()
        self.model = self.model_description(num_classes)
        self.training_history = None

    @abstractmethod
    def num_labels(self):
        pass

    @classmethod
    @abstractmethod
    def tokenizer(cls, labels):
        pass

    @classmethod
    @abstractmethod
    def encoder(cls, texts):
        pass

    @classmethod
    @abstractmethod
    def model_description(cls, num_labels):
        pass

    @abstractmethod
    def vectorize_texts(self, texts):
        pass


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
        return cls(tok, enc)

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

    def predict(self, texts, probabilities=False):
        if probabilities:
            return self.predict_labels_with_probability(texts)
        else:
            return self.predict_labels(texts)

    def predict_labels(self, texts):
        X = self.vectorize_texts(texts)
        encoded_predictions = self.model.predict_classes(X)
        return list(self.label_encoder.inverse_transform(encoded_predictions))

    def predict_labels_with_probability(self, texts):
        X = self.vectorize_texts(texts)
        prob_data = self.model.predict_proba(X)
        # get indices of max values (-> aka. classes (encoded))
        idx = np.argmax(prob_data, axis=1)
        # get the values at those indices (the probabilies)
        probs = prob_data[np.arange(len(prob_data)), idx]
        probs = [float(p) for p in probs]
        # decode the predicted classes to get the category ids
        predictions = self.label_encoder.inverse_transform(idx)
        # zip the predicted cids with their prediction probability
        return list(zip(predictions, probs))

    def train(self, texts, labels, test_size, epochs):
        all_data = list(zip(texts, labels))
        train_set, val_set = train_test_split(all_data, test_size=test_size)

        val_data = self.vectorize_batch(val_set)

        gen = self.training_gen(train_set)
        steps_per_epoch = len(train_set) / self.BATCH_SIZE
        hist = self.model.fit_generator(gen,
                                        epochs=epochs,
                                        steps_per_epoch=steps_per_epoch,
                                        validation_data=val_data,
                                        max_queue_size=2)
        self.training_history = hist
        return hist

    def plot_training_history(self):
        """plots the metrics of an instance of this model's training history"""
        if self.training_history is None:
            raise Exception('Model has not been trained yet')
        history = self.training_history.history
        plt.plot(history['loss'])
        labels = ['training loss']
        metrics = {'val_loss': 'validation loss', 'acc': 'training accuracy',
                   'val_acc': 'validation accuracy'}
        for metric, label in metrics.items():
            if metric in history:
                plt.plot(history[metric])
                labels.append(label)

        plt.title(self.__class__.__name__ + ': Training metrics')
        plt.xlabel('epoch')
        plt.legend(labels, loc='center right')
        plt.show()

    def save_to_files(self, model_fname, tokenizer_fname, encoder_fname):
        self.model.save(model_fname)
        pickle.dump(self.tokenizer, open(tokenizer_fname, 'wb'))
        pickle.dump(self.label_encoder, open(encoder_fname, 'wb'))