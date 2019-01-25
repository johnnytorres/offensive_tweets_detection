
import numpy as np
from gensim.models import Word2Vec


class BaseModel:
    def __init__(self, task):
        self.args = task.args
        self.dataset = task.dataset

    def train(self, X, y):
        pass

    def predict(self, X):
        pass

    def build_fit_w2v(self, X_text):
        w2v_model = Word2Vec()
        w2v_model.build_vocab(X_text)
        w2v_model.train(
            X_text,
            total_examples=w2v_model.corpus_count,
            epochs=w2v_model.iter
        )
        return w2v_model

    def transform_text_to_w2v(self, wv2_model, X_text):
        X_tmp = []
        for sentence in X_text:
            embeddings = []
            for word in sentence.split():
                if word in wv2_model:
                    embeddings.append(wv2_model[word])
            if len(embeddings) == 0:
                emb_avg = np.zeros(wv2_model.vector_size)
            else:
                emb_avg = np.average(embeddings, axis=0)
            X_tmp.append(emb_avg)
        return np.array(X_tmp)
