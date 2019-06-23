#!/usr/bin/env bash

set -e

BASE_DIR=$([ "$1" = "" ] && echo "." || echo "$1" )

DATA_DIR=${BASE_DIR}/data
EMBEDDINGS_DIR=${BASE_DIR}/embeddings/fasttext
TRAINING_DIR=${DATA_DIR}/offens_eval/training
TEST_DIR=${DATA_DIR}/offens_eval/test


mkdir -p ${DATA_DIR}
mkdir -p ${EMBEDDINGS_DIR}

# copy from the original reposistory
DATA_FILE=${DATA_DIR}/offens_eval.zip
if [ ! -f ${DATA_FILE} ]; then
    echo "fasttext embeddings not found!"
    wget -O ${DATA_FILE} https://storage.googleapis.com/ml-research-datasets/toxicity/offens_eval.zip
    unzip -d ${DATA_DIR} ${DATA_FILE}
fi

# ---------- get embeddings

# for fasttext download and filter to words in the dataset
EMBEDDINGS_FILE=${EMBEDDINGS_DIR}/crawl-300d-2M.vec.zip
if [ ! -f ${EMBEDDINGS_FILE} ]; then
    echo "fasttext embeddings not found!"
    wget -O ${EMBEDDINGS_FILE} https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
    unzip -d ${EMBEDDINGS_DIR} ${EMBEDDINGS_FILE}
fi


EMBEDDINGS_DIR=${BASE_DIR}/embeddings/fasttext
python -m preprocessing.embeddings_builder \
    --data-files \
    ${TRAINING_DIR}/offenseval-training-v1.tsv \
    ${TEST_DIR}/testset-taska.tsv \
    ${TEST_DIR}/testset-taskb.tsv \
    ${TEST_DIR}/testset-taskc.tsv \
    --embeddings-file=${EMBEDDINGS_DIR}/crawl-300d-2M.vec \
    --output-dir=${TRAINING_DIR} \
    --text-field=tweet

# for word2vec train with words in the dataset
EMBEDDINGS_DIR=${BASE_DIR}/embeddings/w2v
python -m preprocessing.embeddings_builder \
    --data-files \
    ${TRAINING_DIR}/offenseval-training-v1.tsv \
    ${TEST_DIR}/testset-taska.tsv \
    ${TEST_DIR}/testset-taskb.tsv \
    --embeddings-file=${EMBEDDINGS_DIR}/w2v.vec \
    --output-dir=${TRAINING_DIR} \
    --text-field=tweet \
    --w2v

# future work with Glove
#EMBEDDINGS_DIR=~/data/embeddings/glove
#python -m preprocessing.embeddings_builder \
#    --data-files \
#    ${TRAINING_DIR}/offenseval-training-v1.tsv \
#    ${TEST_DIR}/testset-taska.tsv \
#    ${TEST_DIR}/testset-taskb.tsv \
#    --embeddings-file=${EMBEDDINGS_DIR}/glove.twitter.27B.200d.txt \
#    --output-dir=${TRAINING_DIR} \
#    --text-field=tweet \
#    --no-embeddings-header


# PREPROCESSING TWEETS

python -m preprocessing.text_tokenizer \
    --input-file=${TRAINING_DIR}/offenseval-training-v1.tsv \
    --output-file=${TRAINING_DIR}/offenseval_preprocessed.tsv \
    --language=english \
    --text-field=tweet

python -m preprocessing.text_tokenizer \
    --input-file=${TEST_DIR}/testset-taska.tsv \
    --output-file=${TEST_DIR}/testset_taska_preprocessed.tsv \
    --language=english \
    --text-field=tweet

python -m preprocessing.text_tokenizer \
    --input-file=${TEST_DIR}/testset-taskb.tsv \
    --output-file=${TEST_DIR}/testset_taskb_preprocessed.tsv \
    --language=english \
    --text-field=tweet

python -m preprocessing.text_tokenizer \
    --input-file=${TEST_DIR}/testset-taskc.tsv \
    --output-file=${TEST_DIR}/testset_taskc_preprocessed.tsv \
    --language=english \
    --text-field=tweet



