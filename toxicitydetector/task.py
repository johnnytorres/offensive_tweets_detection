
import os
import uuid
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from argparse import ArgumentParser
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from toxicitydetector.preprocessing.csv_reader import DataLoader
from toxicitydetector.models.factory import get_model


class ClassificationTask:
    def __init__(self, args):
        self.args = args
        self.args.run_id = str(uuid.uuid4())
        self.args.initial_timestamp = datetime.now().timestamp()
        self.dataset = DataLoader(self.args)
        self.output_path = args.output_file
        self.write_header = not self.args.no_output_headers
        # set random state
        np.random.seed(args.random_state)
        print("PARAMS: {}".format(self.args))

    def split_dataset(self):

        X = self.dataset.X_labeled.values
        y = self.dataset.y_labeled.values

        if self.args.predict:
            X_train, y_train = X, y
            X_test, y_test = self.dataset.X_test.values, None
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2,random_state=self.args.random_state)

        return X_train, X_test, y_train, y_test

    def run(self):

        self.dataset.load()

        X_train, X_test,  y_train, y_test = self.split_dataset()

        logging.info("Train data: {}".format(X_train.shape))
        logging.info("Test data: {}".format(X_test.shape))

        labels = self.dataset.y_cols
        results = None
        k_folds = self.args.kfolds
        scores = []
        best_model = None
        best_score = 0
        cv = KFold(k_folds, random_state=self.args.random_state)

        for k, fold in enumerate(cv.split(X_train, y_train)):

            logging.info('training fold {}'.format(k))
            train, valid = fold
            X_kfold, X_valid = X_train[train], X_train[valid]
            y_kfold, y_valid = y_train[train], y_train[valid]

            model = get_model(self)
            model.train(X_kfold, y_kfold)
            y_pred = model.predict(X_valid)

            preds = (y_pred > 0.5).astype(int) if self.args.predict_probs else y_pred
            score = precision_recall_fscore_support(y_valid, preds, average='weighted')
            score = score[2] #F1
            scores.append(score)
            print(f"CV {k} F1: {score}")

            if score > best_score:
                best_score = score
                best_model = model

            y_pred = pd.DataFrame(y_pred, columns=labels)
            y_pred = self.dataset.decode_features(y_pred, labels)
            y_valid = pd.DataFrame(y_valid, columns=labels)
            y_valid = self.dataset.decode_features(y_valid, labels)

            results_df = y_valid.merge(y_pred, left_index=True,right_index=True, suffixes=('', '_pred'))
            results_df['run_id'] = self.args.run_id
            results_df['timestamp'] = datetime.now().timestamp()
            results_df['model'] = model.args.model
            results_df['set'] = 'cv'
            results_df['kfold'] = k
            results_df['id'] = X_valid[:,0]
            results = results_df if results is None else results.append(results_df, ignore_index=True)

        # predict on tests set
        y_pred = best_model.predict(X_test)

        if self.args.predict:

            if len(self.dataset.y_cols) == 1:
                y_pred = np.expand_dims(y_pred, axis=1)

            for ix, col in enumerate(self.dataset.y_cols):
                self.dataset.X_test.loc[:,col] = y_pred[:, ix]

            cols = ['id']
            cols.extend(self.dataset.y_cols)
            results = self.dataset.X_test[cols]
            results = self.dataset.decode_features(results, labels)
            results.to_csv(self.output_path, index=False, header=self.write_header)

        else:
            preds = (y_pred > 0.5).astype(int) if self.args.predict_probs else y_pred
            score = precision_recall_fscore_support(y_test, preds, average='weighted')
            score = score[2] # f1
            scores = np.array(scores)
            print(f"CV F1: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
            print(f"Test F1: %0.2f" % (score))

            y_pred = pd.DataFrame(y_pred, columns=labels)
            y_pred = self.dataset.decode_features(y_pred, labels)
            y_test = pd.DataFrame(y_test, columns=labels)
            y_test = self.dataset.decode_features(y_test, labels)

            results_df = y_test.merge(y_pred, left_index=True,right_index=True, suffixes=('', '_pred'))
            results_df['run_id'] = self.args.run_id
            results_df['timestamp'] = datetime.now().timestamp()
            results_df['model'] = model.args.model
            results_df['set'] = 'test'
            results_df['kfold'] = 0
            results_df['id'] = X_test[:, 0]
            results = results.append(results_df, ignore_index=True)
            write_header = not os.path.exists(self.output_path)
            # save results
            with open(self.output_path, 'a') as f:
                results.to_csv(path_or_buf=f, index=False, header= write_header)

        # save hyperparams
        self.args.final_timestamp = datetime.now().timestamp()
        filepath = os.path.splitext(self.output_path)[0] + '.json'
        with open(filepath, 'a') as f:
            config = json.dumps(self.args.__dict__)
            f.write(config +'\r')


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
    logging.info('initializing task...')
    parser = ArgumentParser()
    parser.add_argument('--train-path', type=lambda x:os.path.expanduser(x))
    parser.add_argument('--unlabeled-path', type=lambda x: os.path.expanduser(x))
    parser.add_argument('--test-path', type=lambda x: os.path.expanduser(x))
    parser.add_argument('--labels', type=lambda x: x.split(','))
    parser.add_argument('--field-sep', type=str, default=',')
    parser.add_argument('--text-field', type=str, default='text')
    parser.add_argument('--use-allfeats', action='store_true')
    parser.add_argument('--num-unlabeled', type=int, default=0)
    parser.add_argument('--kfolds', type=int, default=2)
    parser.add_argument('--ngrams', type=int, default=1)
    parser.add_argument('--embeddings-size', type=int, default=300)
    parser.add_argument('--embeddings-path', type=str, default=None)
    parser.add_argument('--no-embeddings-header', action='store_true')
    parser.add_argument('--random-state', type=int, default=1)
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--predict-probs', action='store_true')
    parser.add_argument('--no-output-headers', action='store_true')
    parser.add_argument('--output-file', type=lambda x:os.path.expanduser(x), default='../results/predictions.csv')
    parser.add_argument('--models')
    args = parser.parse_args()

    for model in args.models.split(','):
        logging.info(f'running {model} model...')
        args.model = model
        task = ClassificationTask(args)
        task.run()
        logging.info(f'running {model} model...[OK]')
    logging.info('task finished...[ok]')








