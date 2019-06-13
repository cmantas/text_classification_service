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
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
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
    def model_description(cls, num_labels):
        pass

    @abstractmethod
    def vectorize_texts(self, texts):
        pass

    @abstractmethod
    def vectorize_labels(self, labels):
        pass

    @classmethod
    @abstractmethod
    def create_from_corpus(cls, texts, labels):
        pass

    def vectorize_batch(self, batch):
        texts, labels = zip(*batch)
        X = self.vectorize_texts(texts)
        Y = self.vectorize_labels(labels)
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
        predictions = self.model.predict_classes(X)

        return predictions

    def predict_labels_with_probability(self, texts):
        X = self.vectorize_texts(texts)
        probability_data = self.model.predict_proba(X)
        return probability_data

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


    def describe(self):
        rv = {'num_labels': self.num_labels()}
        return rv
