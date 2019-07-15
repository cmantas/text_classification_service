from keras import Model
from keras.layers import Dense, Conv1D, GlobalMaxPooling1D, Dropout, MaxPool2D,\
    Conv2D, Input, Reshape, Flatten, Concatenate, MaxPooling1D
from keras import regularizers

from abc import ABC

from ml_models import SequenceModel, BinaryModel, MultiClassModel

__all__ = ['Conv1DModel', 'Conv2DModel', 'Conv1DBinaryModel',
           'Conv1DMultiClassModel', 'Conv2DBinaryModel', 'Conv2DMultiClassModel']

class Conv1DModel(SequenceModel, ABC):
    BATCH_SIZE = 4000
    # we can handle a bigger sequence size with a convolutional architecture
    MAX_SEQ_LEN = 40

    @classmethod
    def recurrent_layers(cls):
        num_filters = 20
        filter_size = 5
        return [
            Conv1D(num_filters, filter_size, activation='relu'),
            GlobalMaxPooling1D(),
            # Dropout(0.1),
            Dense(20, activation='relu'),
            Dropout(0.05),
        ]
class Conv1DBinaryModel(Conv1DModel, BinaryModel):
    pass


class Conv1DMultiClassModel(Conv1DModel, MultiClassModel):
    pass


class Conv2DModel(SequenceModel, ABC):
    BATCH_SIZE = 4000
    MAX_SEQ_LEN = 100

    @classmethod
    def hidden_layers(cls):
        """This method is not used, since `model_description` itself is
        overriden"""
        pass

    def recurrent_layers(self):
        """This method is not used, since `model_description` itself is
        overriden"""
        pass

    def model_description(self):
        filter_sizes = [3, 4, 5]
        num_filters = 10
        drop = 0.1
        embedding_dim = self.EMBEDDING_DIMENTION

        inputs = Input(shape=(self.MAX_SEQ_LEN,), dtype='int32')
        embedding = self.embedding_layer()(inputs)

        reshape = Reshape((self.MAX_SEQ_LEN, embedding_dim, 1))(embedding)

        conv_0 = Conv2D(num_filters,
                        kernel_size=(filter_sizes[0], embedding_dim),
                        padding='valid', kernel_initializer='normal',
                        activation='relu')(reshape)
        conv_1 = Conv2D(num_filters,
                        kernel_size=(filter_sizes[1], embedding_dim),
                        padding='valid', kernel_initializer='normal',
                        activation='relu')(reshape)
        conv_2 = Conv2D(num_filters,
                        kernel_size=(filter_sizes[2], embedding_dim),
                        padding='valid', kernel_initializer='normal',
                        activation='relu')(reshape)

        maxpool_0 = MaxPool2D(pool_size=(self.MAX_SEQ_LEN - filter_sizes[0] + 1, 1),
                              strides=(1, 1),
                              padding='valid')(conv_0)
        maxpool_1 = MaxPool2D(pool_size=(self.MAX_SEQ_LEN - filter_sizes[1] + 1, 1),
                              strides=(1, 1),
                              padding='valid')(conv_1)
        maxpool_2 = MaxPool2D(pool_size=(self.MAX_SEQ_LEN - filter_sizes[2] + 1, 1),
                              strides=(1, 1),
                              padding='valid')(conv_2)

        concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
        flatten = Flatten()(concatenated_tensor)
        dropout = Dropout(drop)(flatten)
        output = Dense(self.num_labels(), activation=self.ACTIVATION)(dropout)

        # this creates a model that includes
        model = Model(inputs=inputs, outputs=output)

        # adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        model.compile(optimizer='adam', loss=self.LOSS_FUNCTION,
                      metrics=['accuracy'])

        return model


class Conv2DBinaryModel(Conv2DModel, BinaryModel):
    pass


class Conv2DMultiClassModel(Conv2DModel, MultiClassModel):
    pass

#https://www.kaggle.com/vsmolyakov/keras-cnn-with-fasttext-embeddings
class MultiStep1DCNN(SequenceModel, ABC):
    num_filters = 64
    weight_decay = 1e-4
    BATCH_SIZE = 1000

    @classmethod
    def recurrent_layers(cls):
        #num_filters = 20
        num_filters = 5
        weight_decay = 1e-4
        return [
            Conv1D(num_filters, 7, activation='relu', padding='same'),
            MaxPooling1D(2),
            Conv1D(num_filters, 7, activation='relu', padding='same'),
            GlobalMaxPooling1D(),
            Dropout(0.05),
            Dense(32, activation='relu',
                  kernel_regularizer=regularizers.l2(weight_decay))
        ]


class MultiStep1DCNNMultiClass(MultiStep1DCNN, MultiClassModel):
    pass


class MultiStep1DCNNBinary(MultiStep1DCNN, BinaryModel):
    pass

