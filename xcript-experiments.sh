#!/usr/bin/env bash

# EVALUATION

# no preprocessing
python -m task \
    --train-path=../data/offens_eval/training/offenseval-training-v1.tsv \
    --test-path=../data/offens_eval/test/testset-taska.tsv \
    --embeddings-path=../data/offens_eval/training/crawl-300d-2M.vec \
    --labels=subtask_a \
    --text-field=tweet \
    --kfolds=10 \
    --model=cnn \

# preprocessing
python -m task \
    --train-path=../data/offens_eval/training/offenseval_preprocessed.tsv \
    --test-path=../data/offens_eval/test/testset_taska_preprocessed.tsv \
    --embeddings-path=../data/offens_eval/training/crawl-300d-2M.vec \
    --labels=subtask_a \
    --text-field=tweet \
    --kfolds=10 \
    --model=cnn


# task A

python -m task \
    --train-path=../data/offens_eval/training/offenseval-training-v1.tsv \
    --test-path=../data/offens_eval/test/testset-taska.tsv \
    --embeddings-path=../data/offens_eval/training/crawl-300d-2M.vec \
    --output-file=../results/predictions_cnn.csv \
    --labels=subtask_a \
    --text-field=tweet \
    --model=cnn \
    --kfolds=10

python -m task \
    --train-path=../data/offens_eval/training/offenseval-training-v1.tsv \
    --test-path=../data/offens_eval/test/testset-taska.tsv \
    --embeddings-path=../data/offens_eval/training/crawl-300d-2M.vec \
    --output-file=../results/predictions_cnn.csv \
    --labels=subtask_a \
    --text-field=tweet \
    --model=cnn \
    --kfolds=10 \
    --predict


# task B

python -m task \
    --train-path=../data/offens_eval/training/offenseval-training-v1.tsv \
    --test-path=../data/offens_eval/test/testset-taskb.tsv \
    --embeddings-path=../data/offens_eval/training/crawl-300d-2M.vec \
    --output-file=../results/task_b_validation.csv \
    --labels=subtask_b \
    --text-field=tweet \
    --model=lr \
    --kfolds=10

python -m task \
    --train-path=../data/offens_eval/training/offenseval-training-v1.tsv \
    --test-path=../data/offens_eval/test/testset-taskb.tsv \
    --embeddings-path=../data/offens_eval/training/crawl-300d-2M.vec \
    --output-file=../results/task_b_validation.csv \
    --labels=subtask_b \
    --text-field=tweet \
    --model=cnn \
    --kfolds=10


#TASK C

python -m task \
    --train-path=../data/offens_eval/training/offenseval-training-v1.tsv \
    --test-path=../data/offens_eval/test/testset-taskc.tsv \
    --embeddings-path=../data/offens_eval/training/crawl-300d-2M.vec \
    --output-file=../results/task_c_validation.csv \
    --labels=subtask_c \
    --text-field=tweet \
    --model=cnn \
    --kfolds=10


