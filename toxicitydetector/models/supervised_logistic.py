import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from toxicitydetector.models.supervised import SupervisedBaseModel


class LogisticModel(SupervisedBaseModel):
    def __init__(self, task):
        super(LogisticModel, self).__init__(task)
        self.args = task.args
        self.dataset = task.dataset
        self.text_repr_model = self.get_text_representation_model()
        self.clf_model = OneVsRestClassifier(
            LogisticRegression(
                solver='lbfgs',
                #max_iter=1000,
                class_weight='balanced'
        ))

    def augment_features(self, X_text, X_all_feats):

        #TODO: this is not scalable to big datasets
        if not self.args.use_allfeats:
            return X_text.toarray()

        # age = X_all_feats[:, 2].reshape(-1, 1)
        # gender = X_all_feats[:, 3].reshape(-1, 1)
        # married = X_all_feats[:, 4].reshape(-1, 1)
        # parenthood = X_all_feats[:, 5].reshape(-1, 1)
        # country = X_all_feats[:, 6].reshape(-1, 1)
        # reflection = X_all_feats[:, 7].reshape(-1, 1)
        # duration = X_all_feats[:, 8].reshape(-1, 1)
        #
        # X_all = np.concatenate(
        #     [X_text.toarray(), age, gender, married,parenthood,country,reflection, duration],
        #     axis=1)

        X_all = np.concatenate([X_text.toarray(), X_all_feats[:, 2:]],axis=1)

        return X_all

    def get_text_representation_model(self):
        steps = []
        vectorizer = TfidfVectorizer(
            ngram_range=(1, self.args.ngrams),
            min_df=3,  # do not affect results
            max_df=.9, # do not affect results
            #stop_words="english",
            use_idf=False
        )
        steps.append(('vec', vectorizer))
        #ch2 = SelectKBest(chi2, k=100) # do not affect results
        #steps.append(('chi2', ch2))
        repr_model = Pipeline(steps)
        return repr_model

    def fit_text(self, X_text, y=None):
        #text_repr = self.get_text_representation_model()
        self.text_repr_model.fit(X_text, y)

    def transform_text(self, X_text):
        return self.text_repr_model.transform(X_text)

    def train(self, X, y):
        X, y = self.augment_instances(X, y)
        X_text = self.text_repr_model.fit_transform(X[:, self.args.text_col])
        X_all_feats = self.augment_features(X_text, X)
        self.clf_model.fit(X_all_feats, y)

    def predict(self, X):
        X_text = self.text_repr_model.transform(X[:, self.args.text_col])
        X_all_feats = self.augment_features(X_text, X)
        
        if self.args.predict_probs:
            y_pred = self.clf_model.predict_proba(X_all_feats)  

            if len(self.args.labels) == 1:
                y_pred = y_pred[:,1]

        else:
            y_pred = self.clf_model.predict(X_all_feats)
        return y_pred