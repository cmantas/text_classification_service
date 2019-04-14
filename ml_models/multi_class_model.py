from abc import ABC
from sklearn.preprocessing import LabelEncoder


class MultiClassModel(ABC):
    @classmethod
    def encoder(cls, labels):
        enc = LabelEncoder()
        enc.fit(labels)
        return enc
