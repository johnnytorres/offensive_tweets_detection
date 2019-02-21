import os
import logging
import argparse
import numpy as np
import pandas as pd
from keras_preprocessing.text import Tokenizer
from tqdm import tqdm


class EmbeddingsBuilder:
    def __init__(self, args):
        logging.info('initializing...')
        self.args = args
        self.embeddings_path = args.embeddings_file
        embeddings_name = os.path.split(self.embeddings_path)[1]
        self.small_embeddings_path = os.path.join(self.args.output_dir, embeddings_name)
        logging.info('initializing...[ok]')

    def build_embedding(self, vocab_dict):
        """
        Load embedding vectors from a .txt file.
        Optionally limit the vocabulary to save memory. `vocab` should be a set.
        """
        num_words = len(vocab_dict)
        num_found = 0

        with open(self.small_embeddings_path, 'w') as out_file:
            with open(self.embeddings_path, 'r') as f:
                header = next(f)
                num_embeddings, embeddings_dim = header.split(' ')
                num_embeddings = int(num_embeddings)
                out_file.write(header)
                for _, line in tqdm(enumerate(f), 'loading embeddings', total=num_embeddings):
                    tokens = line.rstrip().split(" ")
                    word = tokens[0]
                    if word in vocab_dict:
                        num_found += 1
                        out_file.write(line)

        logging.info("Found embeddings for {} out of {} words in vocabulary".format(num_found, num_words))

    def run(self):

        X = None

        for data_file in self.args.data_files:
            ds = pd.read_csv(data_file, sep='\t', keep_default_na=False)
            if X is None:
                X = ds[self.args.text_field].values
            else:
                X = np.append(X, ds[self.args.text_field].values, axis=0)

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(X)

        self.build_embedding(tokenizer.word_index)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
    logging.info('initializing task...')
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-files', type=lambda x:os.path.expanduser(x), nargs='+')
    parser.add_argument('--embeddings-file', type=lambda x:os.path.expanduser(x))
    parser.add_argument('--output-dir', type=lambda x: os.path.expanduser(x))
    parser.add_argument('--num-unlabeled', type=int, default=0)
    parser.add_argument('--text-field', type=str)
    builder = EmbeddingsBuilder(args=parser.parse_args())
    builder.run()
    logging.info('task finished...[ok]')








