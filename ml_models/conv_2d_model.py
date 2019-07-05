from keras.layers import Dense, Embedding, Input, Reshape, Conv2D, MaxPool2D, \
                         Concatenate, Flatten, Dropout
from keras import Model
from keras.preprocessing.sequence import pad_sequences

from ml_models import SequenceModel


class Conv2DModel(SequenceModel):
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

