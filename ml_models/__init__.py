from ml_models.text_model import TextModel
# Abstract classes for Binary/MultiClass classification
from ml_models.multi_class_model import MultiClassModel
from ml_models.binary_model import BinaryModel

from ml_models.word_level_model import WordLevelModel
from ml_models.sequence_model import SequenceModel
from ml_models.character_level_model import CharacterLevelModel

# Abstract classes for various network architectures
from ml_models.feedforward import *
from ml_models.recurrent import *
from ml_models.convolutional import *
from ml_models.fast_text_model import FastTextModel

from ml_models.attention import *


class LSTMFastTextMulticlassModel(LSTMModel, FastTextModel, MultiClassModel):
    pass
class LSTMFastTextMulticlassModel(LSTMModel, FastTextModel, MultiClassModel):
    pass
class LSTMFastTextMulticlassModel(LSTMModel, FastTextModel, MultiClassModel):
    pass