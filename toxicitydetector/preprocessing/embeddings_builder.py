import os
import logging
import argparse
import numpy as np
import pandas as pd
from keras_preprocessing.text import Tokenizer
from tqdm import tqdm
from gensim.models import Word2Vec


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
        lines = []

        with open(self.embeddings_path, 'r') as f:
            header = next(f)
            num_embeddings, embeddings_dim = header.split(' ')
            num_embeddings = int(num_embeddings)
            embeddings_dim = int(embeddings_dim)
            #out_file.write(header)
            for _, line in tqdm(enumerate(f), 'loading embeddings', total=num_embeddings):
                tokens = line.rstrip().split(" ")
                word = tokens[0]
                if word in vocab_dict:
                    num_found += 1
                    #out_file.write(line)
                    lines.append(line)

        with open(self.small_embeddings_path, 'w') as out_file:
            header = f'{num_found} {embeddings_dim}\n'
            out_file.write(header)
            for l in lines:
                out_file.write(l)

        logging.info("Found embeddings for {} out of {} words in vocabulary".format(num_found, num_words))

    def build_fit_w2v(self, X_text):

        X_text = [ sentence.split() for sentence in X_text ]

        w2v_model = Word2Vec(size=300)
        w2v_model.build_vocab(X_text)
        w2v_model.train(
            X_text,
            total_examples=w2v_model.corpus_count,
            epochs=w2v_model.iter
        )
        w2v_model.wv.save_word2vec_format(self.small_embeddings_path)
        #return w2v_model

    def run(self):

        X = None

        for data_file in self.args.data_files:
            ds = pd.read_csv(data_file, sep=self.args.sep, keep_default_na=False)
            if X is None:
                X = ds[self.args.text_field].values
            else:
                X = np.append(X, ds[self.args.text_field].values, axis=0)

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(X)

        if self.args.w2v:
            X = tokenizer.texts_to_sequences(X)
            X = tokenizer.sequences_to_texts(X)
            self.build_fit_w2v(X)
        else:
            self.build_embedding(tokenizer.word_index)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
    logging.info('initializing task...')
    parser = argparse.ArgumentParser()
    # 
    parser.add_argument('--data-files', type=lambda x:os.path.expanduser(x), nargs='+')
    parser.add_argument('--embeddings-file', type=lambda x:os.path.expanduser(x))
    parser.add_argument('--output-dir', type=lambda x: os.path.expanduser(x))
    #parser.add_argument('--num-unlabeled', type=int, default=0)
    parser.add_argument('--text-field', type=str, default='text')
    parser.add_argument('--sep', type=str, default=',')
    parser.add_argument('--w2v', action='store_true')
    #parser.add_argument('--no-embeddings-header', action='store_true')
    builder = EmbeddingsBuilder(args=parser.parse_args())
    builder.run()
    logging.info('task finished...[ok]')








