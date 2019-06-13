from numpy import array
from ml_models import TextModel

class BinaryModel(TextModel):
    ACTIVATION = 'sigmoid'
    LOSS_FUNCTION = 'binary_crossentropy'

    def num_labels(self):
        return 1

    def vectorize_labels(self, labels):
        return array(labels)

    @classmethod
    def create_from_corpus(cls, texts, _labels):
        tok = cls.tokenizer(texts)
        return cls(tok)

    def predict_labels(self, texts):
        predictions = super().predict_labels(texts)
        # return the predictions in a Boolean form
        return [l == 1 for l in predictions.flatten()]

    def predict_labels_with_probability(self, texts):
        probability_data = super().predict_labels_with_probability(texts)
        print("boolean predict with prob")
        probs = [float(p) for p in probability_data]

        predictions = [prob > 0.5 for prob in probs]

        # zip the predicted cids with their prediction probability
        return list(zip(predictions, probs))

    def describe(self):
        super_dict = super().describe()
        return {**super_dict, 'type': 'binary'}
