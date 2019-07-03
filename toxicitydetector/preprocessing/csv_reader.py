
import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class DataLoader:
    def __init__(self, args):
        self.args = args

        self.train_labeled_path = args.train_path
        self.train_unlabeled_path = args.unlabeled_path
        self.test_path = args.test_path

        self.X_labeled = None
        self.y_labeled = None
        self.X_unlabeled = None
        self.y_unlabeled = None

        self.X_test = None

        self.y_cols = []
        self.x_cols = None

        self.age_scaler = None
        self.feature_labelers = {}

    def load(self):

        ds = pd.read_csv(self.train_labeled_path, sep=self.args.field_sep, keep_default_na=False)
        ds_unlabeled = None
        ds_test = None

        # augment unlabeled instances
        if self.args.num_unlabeled > 0:
            ds_unlabeled = pd.read_csv(self.train_unlabeled_path, dtype=object)

        if self.args.predict:
            with open(self.test_path) as datafile:
                ds_test = pd.read_csv(datafile, sep=self.args.field_sep, keep_default_na=False)

        # fill nan in text
        tweet_field = self.args.text_field
        ds[tweet_field] = ds[tweet_field].fillna('')

        if ds_test is not None:
            ds_test[tweet_field] = ds_test[tweet_field].fillna('')

        # fill y cols
        self.y_cols = self.args.labels

        # augment features

        if self.args.use_allfeats:
            self.x_cols = ['id', tweet_field]
        else:
            self.x_cols = ['id', tweet_field]

        # standarize
        #self.standarize_feats(ds,ds_unlabeled, ds_test)
        self.encode_features(ds, ds, self.y_cols)

        self.X_labeled = ds[self.x_cols]
        self.y_labeled = ds[self.y_cols]

        # load unlabeled train set
        if self.args.num_unlabeled > 0:
            #self.standarize_feats(ds_unlabeled)
            self.X_unlabeled = ds_unlabeled[self.x_cols]
            y_train_unlabeled = np.full((ds_unlabeled.shape[0], self.y_labeled.shape[1]), -1)
            self.y_unlabeled = pd.DataFrame(y_train_unlabeled, columns=self.y_cols)

        # load tests set
        if self.args.predict:
            #self.standarize_feats(ds_test)
            self.X_test = ds_test[self.x_cols]

    def standarize_feats(self, ds, ds_unlabeled=None, ds_test=None):
        if not self.args.use_allfeats:
            return

        X = ds
        X_all = ds

        if ds_unlabeled is not None:
            X_all = X_all.append(ds_unlabeled, ignore_index=True, sort=False)

        if ds_test is not None:
            X_all = X_all.append(ds_test, ignore_index=True, sort=False)

        feats = [ '' ]

        self.encode_features(X, X_all, feats)

    def encode_features(self, X, X_all, feats):
        for f in feats:
            X[f].fillna('unk', inplace=True)
            if f not in self.feature_labelers:
                X_all[f].fillna('unk', inplace=True)
                labeler = LabelEncoder()
                labeler.fit(X_all[f].values)
                self.feature_labelers[f] = labeler
            labeler = self.feature_labelers[f]
            X[f] = labeler.transform(X[f].values)

    def decode_features(self, data, features):
        for f in features:
            data.loc[:,f] = self.feature_labelers[f].inverse_transform(data[f])
        return data


if __name__=='__main__':

    parser = ArgumentParser()
    parser.add_argument('--data-labeled', type=lambda x:os.path.expanduser(x))
    parser.add_argument('--data-unlabeled', type=lambda x: os.path.expanduser(x))
    parser.add_argument('--data-test', type=lambda x: os.path.expanduser(x))
    parser.add_argument('--num-unlabeled', type=int, default=0)
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--use-allfeats', action='store_true')
    args = parser.parse_args()
    loader = DataLoader(args)
    loader.load()

    print(f'{loader.X_labeled.shape}')
    print(f'{loader.y_labeled.shape}')
    print(f'{loader.y_labeled.describe()}')