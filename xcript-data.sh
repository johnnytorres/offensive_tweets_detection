#!/usr/bin/env bash


DATA_DIR=../data/offense_eval
EMBEDDINGS_DIR=../embeddings/fasttext

mkdir -p ${DATA_DIR}
mkdir -p ${EMBEDDINGS_DIR}

wget -O ${DATA_DIR}/offense_eval.zip https://storage.googleapis.com/ml-research-datasets/toxicity/twitter/semeval_offensive_tweets.zip
unzip -d ${DATA_DIR} ${DATA_DIR}/offense_eval.zip

wget -O ${EMBEDDINGS_DIR}/crawl-300d-2M.vec.zip https://s3-us-west-1.amazonaws.com/fasttext-vectors/crawl-300d-2M.vec.zip
unzip -d ${EMBEDDINGS_DIR} ${EMBEDDINGS_DIR}/crawl-300d-2M.vec.zip

python -m preprocessing.text_tokenizer \
    --input-file=${DATA_DIR}/training/offenseval-training-v1.tsv \
    --output-file=${DATA_DIR}/training/offenseval_preprocessed.tsv \
    --language=english \
    --text-field=tweet

python -m preprocessing.text_tokenizer \
    --input-file=${DATA_DIR}/test/testset-taska.tsv \
    --output-file=${DATA_DIR}/test/testset_taska_preprocessed.tsv \
    --language=english \
    --text-field=tweet


python -m embeddings \
    --train-path=${DATA_DIR}/training/offenseval_preprocessed.tsv \
    --test-path=${DATA_DIR}/test/testset_taska_preprocessed.tsv \
    --embeddings-path=${EMBEDDINGS_DIR}/crawl-300d-2M.vec \
    --text-field=tweet \
    --labels=subtask_a