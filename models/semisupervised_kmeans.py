from keras.layers import Input, Dense, Embedding, Concatenate, Conv2D, MaxPooling2D, Dropout, Convolution2D
from keras.layers.core import Reshape, Flatten
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model
from keras.preprocessing import sequence
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import random as rd
import time
import os
from gensim.models.word2vec import Word2Vec
# import MeCab
import subprocess
import itertools
import string

from models.supervised_fasttext import FastTextModel


class SemisupervisedKmeansModel(FastTextModel):
    def __init__(self, task):
        super(SemisupervisedKmeansModel, self).__init__(task)
        # set parameters:
        #self.max_features = 4000
        #self.max_len = 400
        #self.batch_size = 32
        #self.embedding_dim = 50
        #self.filters = 250
        #self.kernel_size = 3
        #self.hidden_dims = 250
        #self.epochs = 5

    def build_model(self):
        print('Build model...')
        model = Model()
        return model

    def train(self, X, y):
        print('TRAINING')
        X, y = self.augment_instances(X, y)
        # convert to sequences
        X_text = X[:, self.args.TEXT_COL]

        X_text = self.preprocess_text(X_text)
        self.max_len = int(self.max_len*0.75)

        X = X_text  # todo: add other features

        X = self.tokenizer.sequences_to_texts(X)
        word2vec = self.build_fit_w2v(X)
        X = self.transform_text_to_w2v(word2vec, X)

        # model params
        n_clusters = 4 # binary agency/social
        clusterID = np.arange(n_clusters)
        m_trains = len(X)
        # Window width
        filter_sizes = [3, 5, 7]
        # Dimension of distributed representation
        vector_length = word2vec.vector_size
        sequence_length = self.max_len
        num_filters = 16
        nb_epoch = 1
        batch_size = 32
        # centroids initialization
        np.random.seed(42)
        output_dim = 50
        centroids = np.random.rand(n_clusters, output_dim)

        # CNN
        inputs = Input(shape=(sequence_length, vector_length, 1), dtype='float32')
        # embedding = Embedding(output_dim=embedding_dim, input_dim=vocabulary_size, input_length=sequence_length)(inputs)
        # reshape = Reshape((sequence_length,embedding_dim,1))(embedding)
        conv_0 = Conv2D(
            filters = num_filters,
            kernel_size = (filter_sizes[0],vector_length),
            padding ='valid',
            kernel_initializer ='normal',
            activation='relu',
            #data_format='tf'
        )(inputs)
        conv_1 = Conv2D(
            filters=num_filters,
            kernel_size=(filter_sizes[1],vector_length),
            padding='valid',
            kernel_initializer ='normal',
            activation='relu',
            #data_format='tf'
        )(inputs)
        conv_2 = Conv2D(
            filters=num_filters,
            kernel_size=(filter_sizes[2],vector_length),
            padding='valid',
            kernel_initializer ='normal',
            activation='relu',
            #data_format='tf'
        )(inputs)

        maxpool_0 = MaxPooling2D(
            pool_size=(sequence_length - filter_sizes[0] + 1, 1),
            strides=(1, 1),
            border_mode='valid',
            dim_ordering='tf')(conv_0)
        maxpool_1 = MaxPooling2D(
            pool_size=(sequence_length - filter_sizes[1] + 1, 1),
            strides=(1, 1),
            border_mode='valid',
            dim_ordering='tf')(conv_1)
        maxpool_2 = MaxPooling2D(
            pool_size=(sequence_length - filter_sizes[2] + 1, 1),
            strides=(1, 1),
            border_mode='valid',
            dim_ordering='tf')(conv_2)

        merged_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
        flatten = Flatten()(merged_tensor)
        output = Dense(output_dim=output_dim, activation='linear', init='uniform')(flatten)
        model = Model(input=inputs, output=output)
        adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        # convert labels to kmeans format
        y = y.astype(np.uint8)
        y_pad = np.zeros((len(y), 6), dtype=np.uint8)
        y = np.concatenate([y_pad, y], axis=1)
        y = np.packbits(y, axis=1)
        y = y.flatten()

        # Choose k each from all clusters and use it for learning as supervision
        # Index of supervised train_X of supervised data

        k = 1

        supervised = []

        # Extract the same sample each time, label it
        rd.seed(42)
        for i in range(n_clusters):
            elements = list(np.arange(len(X))[y == i])
            supervised.extend(rd.sample(elements, k))

        Sup = pd.DataFrame({"train_index": supervised, "label": y[supervised]})

        # r, label 2 id initialization
        r = np.array([rd.sample(list(clusterID), 1)[0] for _ in range(len(X))])
        label2id = {i: i for i in clusterID}

        # Global variable
        sup_cent = []
        for cluster in y:
            sup_cent.append(list(centroids[label2id[cluster]]))

        delta = 0
        alpha = 0.01
        margin = 1.0

        # ## Objective function

        # Definition of objective function
        def _cost(true_y, pred_y):
            #global centroids, r, sup_pred, supervised, label2id, Data, sup_cent, alpha, l
            supervision = tf.constant(supervised)
            term1 = alpha * tf.reduce_sum(tf.square(pred_y - centroids[r]))
            #term1_1 = tf.cast(term1, tf.float32)

            l2_norm = tf.reduce_sum(tf.square(tf.gather(pred_y, supervision) - np.array(sup_cent)), 1)
            term2_1 = (1 - alpha) * tf.reduce_sum(l2_norm)
            l_margin = tf.cast(margin, tf.float32)

            l2_norm_1 = tf.cast(l2_norm, tf.float32)
            A = tf.add(l_margin, l2_norm_1)
            B = tf.reshape(tf.gather(pred_y, supervision), [tf.gather(pred_y, supervision).get_shape().as_list()[0], 1,
                                                            tf.gather(pred_y, supervision).get_shape().as_list()[1]])
            B_1 = tf.cast(B, tf.float32)
            C = tf.cast(
                ds.ix[supervised, "label"].apply(lambda x: np.delete(centroids, label2id[x], 0)).values[0][np.newaxis,:,:],
                tf.float32)
            term2_2 = tf.reshape(A, [A.get_shape().as_list()[0], 1]) - tf.reduce_sum(tf.square(B_1 - C), 2)
            condition = tf.greater(term2_2, 0)
            term2_2 = tf.where(condition, term2_2, tf.zeros_like(term2_2))
            term2_2 = tf.reduce_sum(term2_2)

            return tf.add(term1, tf.add(term2_1, term2_2))

        # In[148]:

        model.compile(optimizer=adam, loss=_cost)

        # # 4. Learning

        # In[154]:

        def train(train_X, train_y, supervised):
            global centroids, clusterID, Sup, model, epoch, r, sup_pred, label2id, delta, sup_cent, alpha, l_margin
            # 1-1. Labeling with k - NN, r (Clustering result) updated
            pred_y = model.predict(train_X)
            neigh = KNeighborsClassifier(n_neighbors=1)
            # print(centroids)
            neigh.fit(centroids, clusterID)
            # Cluster ID of predicted train_X
            r = neigh.predict(pred_y)

            # 1-2. Link Hungarian algorithm with labeled data and centroid
            sup_pred = pred_y[supervised]
            hg = pd.concat([Sup, pd.DataFrame(sup_pred)], axis=1).groupby("label")[np.arange(output_dim)].mean()
            hglabel = hg.index
            #    print(hg)
            hgx = hg.values
            HgMatrix = np.linalg.norm(hgx[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
            #    print(HgMatrix)
            label2id = {hglabel[i]: np.argmin(HgMatrix[i]) for i in range(len(supervised))}
            Sup["ID"] = Sup["label"].apply(lambda x: label2id[x])

            l_margin = 1.0
            l2_norm = np.sum(np.square(sup_pred - np.array(sup_cent)), axis=1)
            delta = (l_margin + l2_norm)[:, np.newaxis] - np.sum(np.square(
                sup_pred[:, np.newaxis, :] - centroids[np.newaxis, :, :]), axis=2)
            delta[delta > 0] = 1
            delta[delta <= 0] = 0

            # 2. Update the centroids with r
            for k in range(len(centroids)):
                sum1 = np.sum(alpha * len(np.where(r == k)[0]))
                sum2 = np.sum(alpha * pred_y[r == k])
                delta_k = delta.copy()
                delta_k[:, np.where(np.arange(len(centroids)) != k)] *= -1
                elements = map(lambda x: label2id[x], Sup["label"].values)
                id_of_gn = np.array(list(elements))
                print(delta_k)
                delta_k[np.arange(len(supervised)), id_of_gn] = 0
                w = np.sum(delta_k, axis=1)
                w[np.where(id_of_gn == k)] += 1
                w *= (1 - alpha)
                sum3 = np.sum(w)
                sum4 = np.sum(w[:, np.newaxis] * sup_pred)
                centroids[k] = (sum2 + sum4) / (sum1 + sum3)

            # 3.NN parameter update
            sup_cent = []
            ds.ix[supervised, "label"].apply(lambda x: sup_cent.append(list(centroids[label2id[x]])))

            model.fit(train_X, train_y, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1)

            # Repeat above 1 to 3
            return None


        n_epochs = 5
        for epoch in range(n_epochs):
            start = time.time()
            train(X_train, y_train, supervised)
            end = time.time()
            print("time: " + str(end - start) + " [s]")



        # # Improvement points (If you do)
        # * How to respond to "Product name +?"
        #   * ** I will do it as it is **
        # How to respond to differences in the number of words by * Question
        # * _ tokenize -> part of speech or decreasing
        # * Mini batch to be able to learn.

        # # Reference
        # * [Semi-supervised Clustering for Short Text via Deep Representation Learning (Wang et al., 2016)](https://arxiv.org/abs/1602.06797)
        # * [TensorFlow: using a tensor to index another tensor](https://stackoverflow.com/questions/35842598/tensorflow-using-a-tensor-to-index-another-tensor)
        #

    def predict(self, X):

        return model.predict(X)




if __name__ == '__main__':


    pass