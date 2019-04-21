from abc import ABC
from sklearn.preprocessing import LabelEncoder


class MultiClassModel(ABC):
    ACTIVATION = 'softmax'
    LOSS_FUNCTION = 'sparse_categorical_crossentropy'

    @classmethod
    def encoder(cls, labels):
        enc = LabelEncoder()
        enc.fit(labels)
        return enc

    def vectorize_labels(self, labels):
        return self.label_encoder.transform(labels)

    def num_labels(self):
        return len(self.label_encoder.classes_)

