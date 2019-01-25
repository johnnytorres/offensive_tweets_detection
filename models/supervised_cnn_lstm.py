from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
from keras.datasets import imdb

from models.supervised_fasttext import FastTextModel


class CnnLstmModel(FastTextModel):
    def __init__(self, task):
        super(CnnLstmModel, self).__init__(task)
        # set parameters:
        #self.max_features = 4000
        #self.max_len = 400
        #self.batch_size = 32

        # Convolution
        self.filters = 64 # 250
        self.kernel_size = 5 # 3
        #self.hidden_dims = 250
        self.pool_size = 4

        # LSTM
        self.lstm_output_size = 70

        # training
        self.epochs = 25
        self.batch_size = 30


    def build_model(self):
        print('Build model...')
        model = Sequential()
        model.add(Embedding(self.max_features, self.embeddings_dim, input_length=self.max_len))
        model.add(Dropout(0.25))
        model.add(Conv1D(self.filters,
                         self.kernel_size,
                         padding='valid',
                         activation='relu',
                         strides=1))
        model.add(MaxPooling1D(pool_size=self.pool_size))
        model.add(LSTM(self.lstm_output_size))
        model.add(Dense(self.num_labels))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        return model


if __name__ == '__main__':

    from keras.preprocessing import sequence
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation
    from keras.layers import Embedding
    from keras.layers import LSTM
    from keras.layers import Conv1D, MaxPooling1D
    from keras.datasets import imdb

    # Embedding
    max_features = 20000
    maxlen = 100
    embedding_size = 128

    # Convolution
    kernel_size = 5
    filters = 64
    pool_size = 4

    # LSTM
    lstm_output_size = 70

    # Training
    batch_size = 30
    epochs = 2

    '''
    Note:
    batch_size is highly sensitive.
    Only 2 epochs are needed as the dataset is very small.
    '''

    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    print('Build model...')

    model = Sequential()
    model.add(Embedding(max_features, embedding_size, input_length=maxlen))
    model.add(Dropout(0.25))
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(LSTM(lstm_output_size))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print('Train...')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)
