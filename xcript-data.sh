#!/usr/bin/env bash


python -m preprocessing.text_tokenizer \
    --input-file=../data/offens_eval/training-v1/offenseval-training-v1.tsv \
    --output-file=../data/offens_eval/training-v1/offenseval_preprocessed.tsv \
    --language=english \
    --text-field=tweet

python -m preprocessing.text_tokenizer \
    --input-file=../data/offens_eval/test_a/testset-taska.tsv \
    --output-file=../data/offens_eval/test_a/testset_taska_preprocessed.tsv \
    --language=english \
    --text-field=tweet


python -m embeddings \
    --train-path=../data/offens_eval/training-v1/offenseval_preprocessed.tsv \
    --test-path=../data/offens_eval/test_a/testset_taska_preprocessed.tsv \
    --embeddings-path=../embeddings/fasttext/crawl-300d-2M.vec \
    --text-field=tweet \
    --labels=subtask_a