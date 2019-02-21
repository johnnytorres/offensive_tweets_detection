#!/usr/bin/env bash

BASE_DIR=$([ "$1" = "" ] && echo "." || echo "$1" )

# CHANGE THIS DIRECTORIES
DATA_DIR=${BASE_DIR}/data/offens_eval
EMBEDDINGS_DIR=${BASE_DIR}/embeddings/fasttext

TRAINING_DIR=${DATA_DIR}/training
TEST_DIR=${DATA_DIR}/test

mkdir -p ${DATA_DIR}
mkdir -p ${EMBEDDINGS_DIR}

wget -O ${DATA_DIR}/offense_eval.zip https://storage.googleapis.com/ml-research-datasets/toxicity/twitter/semeval_offensive_tweets.zip
unzip -d ${DATA_DIR} ${DATA_DIR}/offense_eval.zip

wget -O ${EMBEDDINGS_DIR}/crawl-300d-2M.vec.zip https://s3-us-west-1.amazonaws.com/fasttext-vectors/crawl-300d-2M.vec.zip
unzip -d ${EMBEDDINGS_DIR} ${EMBEDDINGS_DIR}/crawl-300d-2M.vec.zip

# embeddings with raw data
python -m preprocessing.embeddings \
    --data-files \
    ${TRAINING_DIR}/offenseval-training-v1.tsv \
    ${TEST_DIR}/testset-taska.tsv \
    ${TEST_DIR}/testset-taskb.tsv \
    ${TEST_DIR}/testset-taskc.tsv \
    --embeddings-file=${EMBEDDINGS_DIR}/crawl-300d-2M.vec \
    --output-dir=${TRAINING_DIR} \
    --text-field=tweet


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

#EMBEDDINGS



# embeddings, first with preprocessed data
#python -m embeddings \
#    --data-files \
#    ${TRAINING_DIR}/offenseval-training-v1.tsv \
#    ${TEST_DIR}/testset-taska.tsv \
#    ${TEST_DIR}/testset-taskb.tsv \
#    --embeddings-file=${EMBEDDINGS_DIR}/crawl-300d-2M.vec \
#    --output-dir=${TRAINING_DIR} \
#    --text-field=tweet \
