from keras.layers import Dense, Embedding, Input, Reshape, Conv2D, MaxPool2D, \
                         Concatenate, Flatten, Dropout
from keras import Model
from keras.preprocessing.sequence import pad_sequences

from ml_models import WordLevelModel, MultiClassModel


class Conv2DModel(MultiClassModel, WordLevelModel):
    VOCAB_SIZE = 30000
    BATCH_SIZE = 4000
    MAX_SEQ_LEN = 100

    @classmethod
    def model_description(cls, encoder):
        filter_sizes = [3, 4, 5]
        num_filters = 10
        drop = 0.1
        num_labels = len(encoder.classes_)
        embedding_dim = 128

        inputs = Input(shape=(cls.MAX_SEQ_LEN,), dtype='int32')
        embedding = Embedding(input_dim=cls.VOCAB_SIZE,
                              output_dim=embedding_dim,
                              input_length=cls.MAX_SEQ_LEN)(inputs)

        reshape = Reshape((cls.MAX_SEQ_LEN, embedding_dim, 1))(embedding)

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

        maxpool_0 = MaxPool2D(pool_size=(cls.MAX_SEQ_LEN - filter_sizes[0] + 1, 1),
                              strides=(1, 1),
                              padding='valid')(conv_0)
        maxpool_1 = MaxPool2D(pool_size=(cls.MAX_SEQ_LEN - filter_sizes[1] + 1, 1),
                              strides=(1, 1),
                              padding='valid')(conv_1)
        maxpool_2 = MaxPool2D(pool_size=(cls.MAX_SEQ_LEN - filter_sizes[2] + 1, 1),
                              strides=(1, 1),
                              padding='valid')(conv_2)

        concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
        flatten = Flatten()(concatenated_tensor)
        dropout = Dropout(drop)(flatten)
        output = Dense(num_labels, activation='softmax')(dropout)

        # this creates a model that includes
        model = Model(inputs=inputs, outputs=output)

        # adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        return model

    def vectorize_texts(self, texts):
        seqs = self.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(seqs, maxlen=self.MAX_SEQ_LEN)
        return X


