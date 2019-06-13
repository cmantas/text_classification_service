from abc import ABC
from sklearn.preprocessing import LabelEncoder
from ml_models import TextModel
import numpy as np


class MultiClassModel(TextModel):
    ACTIVATION = 'softmax'
    LOSS_FUNCTION = 'sparse_categorical_crossentropy'

    def __init__(self, tokenizer, encoder):
        self.label_encoder = encoder
        super().__init__(tokenizer)

    @classmethod
    def encoder(cls, labels):
        enc = LabelEncoder()
        enc.fit(labels)
        return enc

    def vectorize_labels(self, labels):
        return self.label_encoder.transform(labels)

    def num_labels(self):
        return len(self.label_encoder.classes_)

    @classmethod
    def create_from_corpus(cls, texts, labels):
        tok = cls.tokenizer(texts)
        enc = cls.encoder(labels)
        return cls(tok, enc)

    def predict_labels(self, texts):
        encoded_predictions = super().predict_labels(texts)

        # Use the label encoder to decode the predictions
        decoded_predictions = self.label_encoder.inverse_transform(encoded_predictions)
        # Return the predictions in a list form
        return list(decoded_predictions)

    def predict_labels_with_probability(self, texts):
        probability_data = super().predict_labels_with_probability(texts)

        # get indices of max values (-> aka. classes (encoded))
        idx = np.argmax(probability_data, axis=1)
        # get the values at those indices (the probabilies)
        probs = probability_data[np.arange(len(probability_data)), idx]
        probs = [float(p) for p in probs]

        # decode the predicted classes to get the category ids
        predictions = self.label_encoder.inverse_transform(idx)

        # zip the predicted cids with their prediction probability
        return list(zip(predictions, probs))

    def describe(self):
        super_dict = super().describe()
        return {**super_dict, 'type': 'multi_class'}
