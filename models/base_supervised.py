import numpy as np
from sklearn.semi_supervised import LabelSpreading
from tqdm import tqdm

from models.base import BaseModel


class SupervisedBaseModel(BaseModel):

    def fit_text(self, X_text, y=None):
        pass

    def transform_text(self, X_text):
        pass

    def augment_features(self, X_text, X_allfeats):
        return X_text

    def augment_instances(self, X_train, y_train):

        if self.args.num_unlabeled == 0:
            return X_train, y_train

        X_unlabeled = self.dataset.X_train_unlabeled
        y_unlabeled = self.dataset.y_train_unlabeled

        X_unlabeled = X_unlabeled.values
        y_unlabeled = y_unlabeled.values


        X_train_text = X_train[:, self.args.TEXT_COL]
        self.fit_text(X_train_text, y_train)
        X_train_rep = self.transform_text(X_train_text)
        X_train_rep = self.augment_features(X_train_rep, X_train)

        chunk_size = 1000
        num_instances = X_unlabeled.shape[0]
        num_cols = y_train.shape[1]
        for row in tqdm(range(0, self.args.num_unlabeled, chunk_size), desc='spreading labels in rows',
                        total=int(self.args.num_unlabeled / chunk_size)):
            end_row = row + chunk_size
            end_row = np.minimum(end_row, num_instances)
            for col in tqdm(range(num_cols), desc='spreading labels in cols', leave=False):

                X_unlabeled_rep = self.transform_text(X_unlabeled[row:end_row, self.args.TEXT_COL])
                X_unlabeled_rep = self.augment_features(X_unlabeled_rep, X_unlabeled[row:end_row, :])

                X_spread = np.append(X_train_rep, X_unlabeled_rep, axis=0)
                y_spread = np.append(y_train[:, col], y_unlabeled[row:end_row, col], axis=0)

                labeling = LabelSpreading()
                labeling.fit(X_spread, y_spread)
                y_unlabeled[row:end_row, col] = labeling.predict(X_unlabeled_rep)

        X_train = np.append(X_train, X_unlabeled[:row + chunk_size], axis=0)
        y_train = np.append(y_train, y_unlabeled[:row + chunk_size], axis=0)
        return X_train, y_train

