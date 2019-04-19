from ml_models import LSTMModel
from keras.layers import CuDNNLSTM


class CUDALSTMModel(LSTMModel):

    # Override the LSTMModel's recurrent_layers so as to use the CUDA-specific
    # layer (for performance)
    @classmethod
    def recurrent_layers(cls):
        return [CuDNNLSTM(128)]
