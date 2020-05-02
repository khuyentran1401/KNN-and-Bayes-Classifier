# KNearest Neighbor Classifier and Bayes Classifier
This repo compares optimal bayes classifier-a probabilistic approach and knearest neighbor classifier-a geometric approach

## Optimal Bayes Classifier
The Bayes Optimal Classifier is a probabilistic model that makes the most probable prediction for a new example. The implementation of this file could be found [here](./optimal_bayes.py). This classifier assumes the loss function is symmetric. The mathematical derivation of optimal bayes classifier with symmetric loss function could be found [here](https://github.com/khuyentran1401/KNN-and-Bayes-Classifier/blob/master/OptimalBayesDerivation.pdf)

## K Nearest Neighbor Classifier
k-NN is a type of instance-based learning where the function is only approximated locally and all computation is deferred until function evaluation. This algorithm is simple but works surprisingly well in many data. The implementation of this algorithm could be found [here](./knearneighbor.py)

## Comparision betweetn 2 classifier
To visualize the performance between 2 classifiers, run
```
python optimal_bayes.py
```
![image](https://github.com/khuyentran1401/KNN-and-Bayes-Classifier/blob/master/Figure_1.png)



