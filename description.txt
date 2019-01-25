
The model used for this text classification task is a Convolutional Neural Network.
For the embedding layer, it uses Fasttext pretrained embeddings and fine tune it during training.
To prevent overfitting it uses a dropout layer of rate 0.5.
Followed by a 1D convolutional layer with 250 filters as output, and kernel size 3.
We use relu activations, and binary cross entropy loss fuction, and optimized with Adam algorithm.
The results on 10-fold cross validation as 0.7

