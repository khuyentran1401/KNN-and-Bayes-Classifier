import gzip, pickle 
from knearneighbor import *

with gzip.open("mnist.pkl.gz", "rb") as f:
    train_set, val_set, test_set = pickle.load(f, encoding="latin1")

X_train, y_train = train_set[0][:100], (train_set[1] == 1).astype('int')[:100]
X_val, y_val = val_set[0][:10], (val_set[1]==1).astype('int')[:10]


def prediction(k):
	pred = knn(X_train, y_train, X_val, k)
	
	print('Error of k = {} is {}'.format(k,error(pred, y_val)))

for k in [1, 3, 5, 9, 11, 15]:
	prediction(k)

