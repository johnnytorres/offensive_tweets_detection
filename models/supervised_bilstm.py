import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.datasets import imdb

from models.supervised_fasttext import FastTextModel


class BiLstmModel(FastTextModel):
    def __init__(self, task):
        super(BiLstmModel, self).__init__(task)
        # set parameters:
        #self.max_features = 20000
        # cut texts after this number of words (among top max_features most common words)
        #self.maxlen = 80
        #self.batch_size = 32
        self.epochs = 5

    def build_model(self):
        print('Build model...')
        weights = None if self.embeddings_matrix is None else [self.embeddings_matrix]
        model = Sequential()
        model.add(
            Embedding(
                self.max_features,
                self.embeddings_dim,
                input_length=self.max_len,
                # mask_zero=True,
                weights=weights,
            )
        )
        model.add(Bidirectional(LSTM(64)))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_labels, activation='sigmoid'))

        # try using different optimizers and different optimizer configs
        model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

        return model


if __name__ == '__main__':

    max_features = 20000
    # cut texts after this number of words
    # (among top max_features most common words)
    maxlen = 100
    batch_size = 32

    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    model = Sequential()
    model.add(Embedding(max_features, 128, input_length=maxlen))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # try using different optimizers and different optimizer configs
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

    print('Train...')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=4,
              validation_data=[x_test, y_test])
