from abc import ABC
from sklearn.preprocessing import LabelEncoder


class MultiClassModel(ABC):
    @classmethod
    def encoder(cls, labels):
        enc = LabelEncoder()
        enc.fit(labels)
        return enc

    def num_labels(self):
        return len(self.label_encoder.classes_)

