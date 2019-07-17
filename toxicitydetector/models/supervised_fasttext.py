'''

Based on Joulin et al's paper:

Bags of Tricks for Efficient Text Classification
https://arxiv.org/abs/1607.01759

Results on IMDB datasets with uni and bi-gram embeddings:
    Uni-gram: 0.8813 test accuracy after 5 epochs. 8s/epoch on i7 cpu.
    Bi-gram : 0.9056 test accuracy after 5 epochs. 2s/epoch on GTx 980M gpu.
'''

import numpy as np
import tensorflow as tf

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from tqdm import tqdm

from toxicitydetector.models.supervised import SupervisedBaseModel


def create_ngram_set(input_list, ngram_value=2):
    """
    Extract a set of n-grams from a list of integers.

    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
    {(4, 9), (4, 1), (1, 4), (9, 4)}

    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
    [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
    """
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngram(sequences, token_indice, ngram_range=2):
    """
    Augment the input list of list (sequences) by appending n-grams values.

    Example: adding bi-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
    >>> add_ngram(sequences, token_indice, ngram_range=2)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]

    Example: adding tri-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
    >>> add_ngram(sequences, token_indice, ngram_range=3)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42, 2018]]
    """
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for ngram_value in range(2, ngram_range + 1):
            for i in range(len(new_list) - ngram_value + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences


def get_embedding_vectors(embeddings_path, vocab_dict, num_words, embeddings_dim, small_embedding_path=None):
    """
    Load embedding vectors from a .txt file.
    Optionally limit the vocabulary to save memory. `vocab` should be a set.
    """

    if embeddings_path is None:
        return None

    #dct = {}
    #vectors = np.array.array('d')
    #dct['UNK'] = 0
    #vectors.extend(float(x) for x in np.zeros(embeddings_dim))
    current_idx = 1  # 0 UNK
    num_found = 0
    embeddings_matrix = np.random.rand(num_words, embeddings_dim)
    #embeddings_matrix = np.zeros((num_words, embeddings_dim))
    embeddings_matrix[0] = np.zeros(embeddings_dim)

    small_embeddings = []

    with tf.gfile.GFile(embeddings_path) as f:
        num_embeddings, _ = next(f).split(' ')
        num_embeddings = int(num_embeddings)
        for _, line in tqdm(enumerate(f), 'loading embeddings'):
            tokens = line.rstrip().split(" ")
            word = tokens[0]
            entries = tokens[1:]
            if word in vocab_dict:
                #dct[word] = current_idx
                #vectors.extend(float(x) for x in entries)
                #current_idx += 1
                num_found += 1
                # if small_embedding_path: # dont validate here to avoid performance issue
                #small_embeddings.append(line)
                word_embeddings = np.array(entries, np.float)
                embeddings_matrix[vocab_dict[word]] = word_embeddings

    #word_dim = len(entries)
    #num_vectors = len(dct)
    tf.logging.info("Found embeddings for {} out of {} words in vocabulary".format(num_found, num_words))

    # if small_embedding_path:
    #     with open(small_embedding_path, 'w') as f:
    #         f.writelines(['{} {}\n'.format(num_vectors, embeddings_dim)])
    #         f.writelines(small_embeddings)

    #return np.array(vectors).reshape(num_vectors, word_dim), dct

    return embeddings_matrix



class FastTextModel(SupervisedBaseModel):
    def __init__(self, task):
        super(FastTextModel, self).__init__(task)
        self.args = task.args
        self.epochs = 5
        self.max_len = 50
        self.batch_size = 32
        self.max_features = None # use all features in the dataset instead of #self.max_features = 5000
        self.embeddings_dim = self.args.embeddings_size
        self.embeddings_matrix = None
        self.ngram_range = 1
        self.tokenizer = Tokenizer(num_words=self.max_features)
        self.model = None # keras model
        self.token_indice = None # for n-grams
        self.num_labels = len(self.args.labels)


    def build_model(self):
        print('Build model...')
        model = Sequential()
        # we start off with an efficient embedding layer which maps
        # our vocab indices into embedding_dims dimensions
        weights = None if self.embeddings_matrix is None else [self.embeddings_matrix]
        model.add(
            Embedding(
                self.max_features,
                self.embeddings_dim,
                input_length=self.max_len,
                trainable=self.args.embeddings_trainable,
                weights = weights,
                #mask_zero=True  # not useful in CNN like models

            ),
        )
        # we add a GlobalAveragePooling1D, which will average the embeddings
        # of all words in the document
        model.add(GlobalAveragePooling1D())
        # We project onto a single unit output layer, and squash it with a sigmoid:
        model.add(Dense(self.num_labels, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

    def show_text_info(self, X_text):
        print(len(X_text), 'text sequences')
        X_text_lens = list(map(len, X_text))
        print('Average sequence length: {}'.format(np.mean(X_text_lens, dtype=int)))
        print('Max sequence length: {}'.format(np.max(X_text_lens)))

    def add_ngrams(self, X_text):
        if self.ngram_range == 1:
            return X_text

        if self.token_indice is None:
            print('Adding {}-gram features'.format(self.ngram_range))
            # Create set of unique n-gram from the training set.
            ngram_set = set()
            for input_list in X_text:
                for i in range(2, self.ngram_range + 1):
                    set_of_ngram = create_ngram_set(input_list, ngram_value=i)
                    ngram_set.update(set_of_ngram)

            # Dictionary mapping n-gram token to a unique integer.
            # Integer values are greater than max_features in order
            # to avoid collision with existing features.
            start_index = self.max_features + 1
            self.token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
            indice_token = {self.token_indice[k]: k for k in self.token_indice}

            # max_features is the highest integer that could be found in the dataset.
            self.max_features = np.max(list(indice_token.keys())) + 1

        # Augmenting x_train and x_test with n-grams features
        X_text = add_ngram(X_text, self.token_indice, self.ngram_range)
        self.show_text_info(X_text)
        return X_text

    def fit_text(self, X_text, y=None):

        X_unlabeled = self.dataset.X_train_unlabeled.values
        X_unlabeled_text = X_unlabeled[:, self.args.text_col]
        X = np.append(X_text, X_unlabeled_text, axis=0)

        #X = self.preprocess_text(X)
        self.tokenizer.fit_on_texts(X)
        X = self.tokenizer.texts_to_sequences(X)
        X = self.tokenizer.sequences_to_texts(X)
        self.text_rep_model = self.build_fit_w2v(X)

    def transform_text(self, X_text):
        X = self.tokenizer.texts_to_sequences(X_text)
        X = self.tokenizer.sequences_to_texts(X)
        X = self.transform_text_to_w2v(self.text_rep_model, X)
        return X

    def preprocess_text(self, X_text):

        X = self.dataset.X_labeled[self.args.text_field].values

        if self.dataset.X_unlabeled is not None:
            X = np.append(X, self.dataset.X_unlabeled[self.args.text_field].values, axis=0)

        # TODO: this is wrong, in real systems we don't have this data
        if self.dataset.X_test is not None:
            X = np.append(X, self.dataset.X_test[self.args.text_field].values, axis=0)

        self.tokenizer.fit_on_texts(X)

        num_words = len(self.tokenizer.word_index)
        # to use all features , set to the number of found features
        #self.max_features = np.minimum(self.max_features, num_words) + 1 # add padding
        #TODO: add oov, use max_features
        self.max_features = num_words + 1 #add paddings 

        self.embeddings_matrix = get_embedding_vectors(
            self.args.embeddings_path,
            self.tokenizer.word_index,
            self.max_features,
            self.embeddings_dim
        )

        X = self.tokenizer.texts_to_sequences(X_text)
        self.show_text_info(X)
        X = self.add_ngrams(X)
        self.max_len = int(np.max(list(map(len, X))))
        X = sequence.pad_sequences(X, maxlen=self.max_len)
        return X

    def train(self, X, y):
        print('TRAINING')
        X, y = self.augment_instances(X, y)
        # convert to sequences
        X_text = X[:,self.args.text_col]

        X_text = self.preprocess_text(X_text)

        X = X_text # todo: add other features

        self.model = self.build_model()

        self.model.fit(
            X, y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            #validation_data=(x_test, y_test)
        )


    def predict(self, X):
        print('PREDICT')
        X_text = X[:, self.args.text_col]
        X_text = self.tokenizer.texts_to_sequences(X_text)
        self.show_text_info(X_text)

        X_text = self.add_ngrams(X_text)

        X = sequence.pad_sequences(X_text, maxlen=self.max_len)
        y = self.model.predict(X, verbose=1)
        y = (y > 0.5).astype(int)
        return y

if __name__ == '__main__':

    # Set parameters:
    # ngram_range = 2 will add bi-grams features
    ngram_range = 1
    max_features = 20000
    maxlen = 400
    batch_size = 32
    embedding_dims = 50
    epochs = 5

    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')
    print('Average train sequence length: {}'.format(
        np.mean(list(map(len, x_train)), dtype=int)))
    print('Average test sequence length: {}'.format(
        np.mean(list(map(len, x_test)), dtype=int)))

    if ngram_range > 1:
        print('Adding {}-gram features'.format(ngram_range))
        # Create set of unique n-gram from the training set.
        ngram_set = set()
        for input_list in x_train:
            for i in range(2, ngram_range + 1):
                set_of_ngram = create_ngram_set(input_list, ngram_value=i)
                ngram_set.update(set_of_ngram)

        # Dictionary mapping n-gram token to a unique integer.
        # Integer values are greater than max_features in order
        # to avoid collision with existing features.
        start_index = max_features + 1
        token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
        indice_token = {token_indice[k]: k for k in token_indice}

        # max_features is the highest integer that could be found in the dataset.
        max_features = np.max(list(indice_token.keys())) + 1

        # Augmenting x_train and x_test with n-grams features
        x_train = add_ngram(x_train, token_indice, ngram_range)
        x_test = add_ngram(x_test, token_indice, ngram_range)
        print('Average train sequence length: {}'.format(
            np.mean(list(map(len, x_train)), dtype=int)))
        print('Average test sequence length: {}'.format(
            np.mean(list(map(len, x_test)), dtype=int)))

    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    print('Build model...')
    model = Sequential()

    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    model.add(Embedding(max_features,
                        embedding_dims,
                        input_length=maxlen))

    # we add a GlobalAveragePooling1D, which will average the embeddings
    # of all words in the document
    model.add(GlobalAveragePooling1D())

    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test))
