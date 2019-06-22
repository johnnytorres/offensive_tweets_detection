#!/usr/bin/env bash

BASE_DIR=$([ "$1" = "" ] && echo "." || echo "$1" )

# CHANGE THIS DIRECTORIES
DATA_DIR=${BASE_DIR}/data/offens_eval
EMBEDDINGS_DIR=~data/embeddings/fasttext

TRAINING_DIR=${DATA_DIR}/training
TEST_DIR=${DATA_DIR}/test

mkdir -p ${DATA_DIR}
mkdir -p ${EMBEDDINGS_DIR}

wget -O ${DATA_DIR}/offens_eval.zip https://storage.googleapis.com/ml-research-datasets/toxicity/offens_eval/offens_eval.zip
unzip -d ${DATA_DIR} ${DATA_DIR}/offens_eval.zip

wget -O ${EMBEDDINGS_DIR}/crawl-300d-2M.vec.zip https://s3-us-west-1.amazonaws.com/fasttext-vectors/crawl-300d-2M.vec.zip
unzip -d ${EMBEDDINGS_DIR} ${EMBEDDINGS_DIR}/crawl-300d-2M.vec.zip

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

EMBEDDINGS_DIR=~/data/embeddings/glove
python -m preprocessing.embeddings_builder \
    --data-files \
    ${TRAINING_DIR}/offenseval-training-v1.tsv \
    ${TEST_DIR}/testset-taska.tsv \
    ${TEST_DIR}/testset-taskb.tsv \
    --embeddings-file=${EMBEDDINGS_DIR}/glove.twitter.27B.200d.txt \
    --output-dir=${TRAINING_DIR} \
    --text-field=tweet \
    --no-embeddings-header

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



