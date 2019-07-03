#!/usr/bin/env bash

set -e

BASE_DIR=$([ "$1" = "" ] && echo "$HOME/data" || echo "$1" )

DATA_DIR=${BASE_DIR}/toxicity/semeval2019
TRAIN_DIR=${DATA_DIR}/training
TEST_DIR=${DATA_DIR}/test
EMBEDDINGS_DIR=${BASE_DIR}/embeddings/fasttext

mkdir -p ${DATA_DIR}
mkdir -p ${EMBEDDINGS_DIR}

# copy from the original reposistory
DATA_FILE=${DATA_DIR}/semeval2019v3.zip
if [[ ! -f ${DATA_FILE} ]]; then
    echo "downloading semeval dataset..."
    wget -O ${DATA_FILE} https://storage.googleapis.com/ml-research-datasets/toxicity/semeval2019v3.zip
    unzip -d ${DATA_DIR} ${DATA_FILE}
fi

# ---------- fasttext embeddings ----------

# for fasttext download and filter to words in the dataset
EMBEDDINGS_FILE=${EMBEDDINGS_DIR}/crawl-300d-2M.vec.zip
if [[ ! -f ${EMBEDDINGS_FILE} ]]
then
    echo "downloading fasttext embeddings..."
    wget -O ${EMBEDDINGS_FILE} https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
fi

EMBEDDINGS_FILE=${EMBEDDINGS_DIR}/crawl-300d-2M.vec
if [[ ! -f ${EMBEDDINGS_FILE} ]]
then
    echo "decompressing fasttext embeddings..."
    unzip -d ${EMBEDDINGS_DIR} ${EMBEDDINGS_FILE}.zip
fi

python -m toxicitydetector.preprocessing.embeddings_builder \
    --data-files \
    ${TRAIN_DIR}/offenseval-training-v1.tsv \
    ${TEST_DIR}/testset-taska.tsv \
    ${TEST_DIR}/testset-taskb.tsv \
    ${TEST_DIR}/testset-taskc.tsv \
    --embeddings-file=${EMBEDDINGS_DIR}/crawl-300d-2M.vec \
    --output-dir=${TRAIN_DIR} \
    --text-field=tweet

# ------------ word2vec ----------
# train with words in the dataset

EMBEDDINGS_DIR=~/data/embeddings/w2v
python -m toxicitydetector.preprocessing.embeddings_builder \
    --data-files \
    ${TRAIN_DIR}/offenseval-training-v1.tsv \
    ${TEST_DIR}/testset-taska.tsv \
    ${TEST_DIR}/testset-taskb.tsv \
    --embeddings-file=${EMBEDDINGS_DIR}/w2v.vec \
    --output-dir=${TRAIN_DIR} \
    --text-field=tweet \
    --w2v

# TODO: future work with Glove
#EMBEDDINGS_DIR=~/data/embeddings/glove
#python -m toxicitydetector.preprocessing.embeddings_builder \
#    --data-files \
#    ${TRAIN_DIR}/offenseval-training-v1.tsv \
#    ${TEST_DIR}/testset-taska.tsv \
#    ${TEST_DIR}/testset-taskb.tsv \
#    --embeddings-file=${EMBEDDINGS_DIR}/glove.twitter.27B.200d.txt \
#    --output-dir=${TRAIN_DIR} \
#    --text-field=tweet \
#    --no-embeddings-header


# TODO: tweets preprocessing

#python -m toxicitydetector.preprocessing.text_tokenizer \
#    --input-file=${TRAIN_DIR}/offenseval-training-v1.tsv \
#    --output-file=${TRAIN_DIR}/offenseval_preprocessed.tsv \
#    --language=english \
#    --text-field=tweet
#
#python -m toxicitydetector.preprocessing.text_tokenizer \
#    --input-file=${TEST_DIR}/testset-taska.tsv \
#    --output-file=${TEST_DIR}/testset_taska_preprocessed.tsv \
#    --language=english \
#    --text-field=tweet
#
#python -m toxicitydetector.preprocessing.text_tokenizer \
#    --input-file=${TEST_DIR}/testset-taskb.tsv \
#    --output-file=${TEST_DIR}/testset_taskb_preprocessed.tsv \
#    --language=english \
#    --text-field=tweet
#
#python -m toxicitydetector.preprocessing.text_tokenizer \
#    --input-file=${TEST_DIR}/testset-taskc.tsv \
#    --output-file=${TEST_DIR}/testset_taskc_preprocessed.tsv \
#    --language=english \
#    --text-field=tweet



