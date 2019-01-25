import numpy as np
from models.base import BaseModel


class UnsupervisedBaseModel(BaseModel):

    def augment_instances(self, X_train, y_train):

        if self.args.num_unlabeled == 0:
            return X_train, y_train

        X_unlabeled = self.dataset.X_train_unlabeled
        y_unlabeled = self.dataset.y_train_unlabeled

        X_unlabeled = X_unlabeled.values
        y_unlabeled = y_unlabeled.values

        row = np.minimum( self.args.num_unlabeled, X_unlabeled.shape[0])

        X_train = np.append(X_train, X_unlabeled[:row ], axis=0)
        y_train = np.append(y_train, y_unlabeled[:row ], axis=0)

        return X_train, y_train