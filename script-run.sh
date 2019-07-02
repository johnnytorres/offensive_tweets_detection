#!/usr/bin/env bash

set -e

BASE_DIR=$([ "$1" = "" ] && echo "$HOME/data" || echo "$1" )

DATA_DIR=${BASE_DIR}/toxicity/semeval2019
TRAIN_DIR=${DATA_DIR}/training
TEST_DIR=${DATA_DIR}/test
OUTPUT_DIR=${DATA_DIR}/results

mkdir -p $OUTPUT_DIR

# classification models evaluation for TASK A
python -m toxicitydetector.task \
    --train-path=${TRAIN_DIR}/offenseval-training-v1.tsv \
    --test-path=${TEST_DIR}/testset-taska.tsv \
    --embeddings-path=${TRAIN_DIR}/crawl-300d-2M.vec \
    --output-file=${OUTPUT_DIR}/predictions_taska.csv \
    --labels=subtask_a \
    --text-field=tweet \
    --kfolds=10 \
    --model=lr,fasttext,cnn,lstm,bilstm

# TODO: evaluate additional tasks: TASK B,C

#python -m toxicitydetector.task \
#    --train-path=${TRAIN_DIR}/offenseval-training-v1.tsv \
#    --test-path=${TEST_DIR}/testset-taskb.tsv \
#    --embeddings-path=${TRAIN_DIR}/crawl-300d-2M.vec \
#    --output-file=../results/task_b_validation.csv \
#    --labels=subtask_b \
#    --text-field=tweet \
#    --model=cnn \
#    --kfolds=10

#python -m toxicitydetector.task \
#    --train-path=${TRAIN_DIR}/offenseval-training-v1.tsv \
#    --test-path=${TEST_DIR}/testset-taskc.tsv \
#    --embeddings-path=${TRAIN_DIR}/crawl-300d-2M.vec \
#    --output-file=../results/task_c_validation.csv \
#    --labels=subtask_c \
#    --text-field=tweet \
#    --model=cnn \
#    --kfolds=10



# embeddings evaluation

python -m toxicitydetector.task \
    --train-path=${TRAIN_DIR}/offenseval-training-v1.tsv \
    --test-path=${TEST_DIR}/testset-taska.tsv \
    --output-file=${OUTPUT_DIR}/predictions_embeddings_random.csv \
    --labels=subtask_a \
    --text-field=tweet \
    --kfolds=10 \
    --model=cnn

python -m toxicitydetector.task \
    --train-path=${TRAIN_DIR}/offenseval-training-v1.tsv \
    --test-path=${TEST_DIR}/testset-taska.tsv \
    --embeddings-path=${TRAIN_DIR}/w2v.vec \
    --output-file=${OUTPUT_DIR}/predictions_embeddings_word2vec.csv \
    --labels=subtask_a \
    --text-field=tweet \
    --kfolds=10 \
    --model=cnn

python -m toxicitydetector.task \
    --train-path=${TRAIN_DIR}/offenseval-training-v1.tsv \
    --test-path=${TEST_DIR}/testset-taska.tsv \
    --embeddings-path=${TRAIN_DIR}/crawl-300d-2M.vec \
    --output-file=${OUTPUT_DIR}/predictions_embeddings_fasttext.csv \
    --labels=subtask_a \
    --text-field=tweet \
    --kfolds=10 \
    --model=cnn

# TODO: evalute Glove embeddings
#python -m toxicitydetector.task \
#    --train-path=${TRAIN_DIR}/offenseval-training-v1.tsv \
#    --test-path=${TEST_DIR}/testset-taska.tsv \
#    --embeddings-path=${TRAIN_DIR}/glove.twitter.27B.200d.txt \
#    --embeddings-size=200 \
#    --no-embeddings-header \
#    --output-file=${OUTPUT_DIR}/predictions_embeddings.csv \
#    --labels=subtask_a \
#    --text-field=tweet \
#    --kfolds=10 \
#    --model=cnn
#-----------------------------------------------------------


# TODO: evaluate with preprocessing of the tweets
#python -m toxicitydetector.task \
#    --train-path=${TRAIN_DIR}/offenseval_preprocessed.tsv \
#    --test-path=${TEST_DIR}/testset_taska_preprocessed.tsv \
#    --embeddings-path=${TRAIN_DIR}/crawl-300d-2M.vec \
#    --labels=subtask_a \
#    --text-field=tweet \
#    --kfolds=10 \
#    --model=cnn
#
## prediction with best preprocessing / model
#
#python -m toxicitydetector.task \
#    --train-path=${TRAIN_DIR}/offenseval-training-v1.tsv \
#    --test-path=${TEST_DIR}/testset-taska.tsv \
#    --embeddings-path=${TRAIN_DIR}/crawl-300d-2M.vec \
#    --output-file=../results/predictions_cnn.csv \
#    --labels=subtask_a \
#    --text-field=tweet \
#    --model=cnn \
#    --kfolds=10 \
#    --predict
#

