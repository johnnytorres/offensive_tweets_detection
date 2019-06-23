
# Offensive Tweets Identification using Convolutional Neural Networks

This is the implementation of the model for the task of identifying offensive tweets in the Task 6 of SemEval-2019
(International Workshop on Semantic Evaluation), as detailed in the paper:

[JTML at SemEval-2019 Task 6: Offensive Tweets Identification using Convolutional Neural Networks] (https://aclweb.org/anthology/papers/S/S19/S19-2117/)

Please cite the paper if you use the code. Thanks!

## Preparing the environment

Please install the requirements in your python environment (virtual is recommended)

```bash
pip install -r requirements.txt
```

## Reproducing the results

First, run the script to download the dataset and prepare the embeddings
```bash
./script-getdata.sh
```

Then, run the scripts to train the models with k-Fold (k=10)
```bash
./script-run.sh
```



