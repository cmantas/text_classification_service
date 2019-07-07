from ml_models.text_model import TextModel
# Abstract classes for Binary/MultiClass classification
from ml_models.multi_class_model import MultiClassModel
from ml_models.binary_model import BinaryModel

from ml_models.word_level_model import WordLevelModel
from ml_models.sequence_model import SequenceModel

# Abstract classes for various network architectures
from ml_models.feed_forward_model import FeedForwardModel
from ml_models.lstm_model import LSTMModel
from ml_models.cuda_lstm_model import CUDALSTMModel
from ml_models.conv_1d_model import Conv1DModel
from ml_models.conv_2d_model import Conv2DModel
from ml_models.fast_text_model import FastTextModel

from ml_models.attention_model import AttentionBinaryModel, \
     AttentionMulticlassModel

# Concrete binary model classes

class FeedForwardBinaryModel(FeedForwardModel, BinaryModel):
    pass


class LSTMBinaryModel(LSTMModel, BinaryModel):
    pass


class Conv1DBinaryModel(Conv1DModel, BinaryModel):
    pass


class Conv2DBinaryModel(Conv2DModel, BinaryModel):
    pass

# Concrete multi class model classes

class FeedForwardMultiClassModel(FeedForwardModel, MultiClassModel):
    pass


class LSTMMultiClassModel(LSTMModel, MultiClassModel):
    pass


class Conv1DMultiClassModel(Conv1DModel, MultiClassModel):
    pass


class Conv2DMultiClassModel(Conv2DModel, MultiClassModel):
    pass


class LSTMFastTextMulticlassModel(LSTMModel, FastTextModel, MultiClassModel):
    pass

# Import some more elaborate models
from ml_models.complex_models import *
