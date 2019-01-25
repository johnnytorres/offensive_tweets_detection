import os
import logging
import argparse
import numpy as np
import tensorflow as tf

from keras_preprocessing.text import Tokenizer
from tqdm import tqdm

from data import DataLoader


class EmbeddingsBuilder:
    def __init__(self, args):
        logging.info('initializing...')
        self.args = args
        self.dataset = DataLoader(self.args)
        self.embeddings_path = args.embeddings_path
        self.small_embeddings_path = os.path.split(self.embeddings_path)[1]
        self.train_dir = os.path.dirname(self.args.train_path)
        self.small_embeddings_path = os.path.join(self.train_dir, self.small_embeddings_path)
        logging.info('initializing...[ok]')

    def build_embedding(self, vocab_dict):
        """
        Load embedding vectors from a .txt file.
        Optionally limit the vocabulary to save memory. `vocab` should be a set.
        """
        num_words = len(vocab_dict)
        num_found = 0

        with open(self.small_embeddings_path, 'w') as out_file:
            with tf.gfile.GFile(self.embeddings_path) as f:
                header =next(f)
                num_embeddings, embeddings_dim = header.split(' ')
                num_embeddings = int(num_embeddings)
                out_file.write(header)
                for _, line in tqdm(enumerate(f), 'loading embeddings', total=num_embeddings):
                    tokens = line.rstrip().split(" ")
                    word = tokens[0]
                    if word in vocab_dict:
                        num_found += 1
                        out_file.write(line)

        tf.logging.info("Found embeddings for {} out of {} words in vocabulary".format(num_found, num_words))

    def run(self):

        self.dataset.load()

        X = self.dataset.X_labeled[self.args.text_field].values

        if self.dataset.X_unlabeled is not None:
            X = np.append(X, self.dataset.X_unlabeled[self.args.text_field].values, axis=0)

        if self.dataset.X_test is not None:
            X = np.append(X, self.dataset.X_test[self.args.text_field].values, axis=0)

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(X)

        self.build_embedding(tokenizer.word_index)





if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
    logging.info('initializing task...')
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-path', type=lambda x:os.path.expanduser(x))
    parser.add_argument('--unlabeled-path', type=lambda x: os.path.expanduser(x))
    parser.add_argument('--test-path', type=lambda x: os.path.expanduser(x))
    parser.add_argument('--embeddings-path', type=str, default=None)
    parser.add_argument('--num-unlabeled', type=int, default=0)
    parser.add_argument('--text-field', type=str)
    parser.add_argument('--labels', type=lambda x: x.split(','))
    parser.add_argument('--use-allfeats', action='store_true', default=False)
    parser.add_argument('--predict', action='store_true', default=True)
    builder = EmbeddingsBuilder(args=parser.parse_args())
    builder.run()
    logging.info('task finished...[ok]')








