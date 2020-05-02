import numpy as np 
import matplotlib.pyplot as plt 
import operator

def distance(q, p):
	return np.sqrt((sum((p - q)**2)))

def knn(points, labels, new_points, n_neighbors=None):

	res = []

	for q in new_points:

		neighbors_dis = [(distance(q, points[i]), labels[i]) for i in range(len(points))]


		neighbors_dis.sort(key = operator.itemgetter(0))

		nearest = {}

		for dis, ele in neighbors_dis[:n_neighbors+1]:

			if ele in nearest:
				nearest[ele] +=1

			else:
				nearest[ele] = 1

		pred = max([(value, key) for key, value in nearest.items()])[1]
		res.append(pred)

	return np.array(res)

def error(pred, y):

	n = len(y)

	err = 0
	for i in range(n):
		err += np.abs(pred[i] - y[i])

	return err/n 

if __name__=='__main__':
 
	np.random.seed(40)
	# Means
	mu_1 = np.array([-2, 0], dtype=np.float32)
	mu_2 = np.array([2, 0], dtype=np.float32)

	# Sigma
	sigma_1 = np.array([[1, 0],
	                    [0, 1]], dtype=np.float32)
	sigma_2 = np.array([[1, 0],
	                    [0, 1]], dtype=np.float32)

	# Multivariate normal sample
	x_1 = np.random.multivariate_normal(mu_1, sigma_1, 100)
	x_2 = np.random.multivariate_normal(mu_2, sigma_2, 100)
	print(x_1.shape)
	print(x_2.shape)

	# Joint sample
	X = np.concatenate([x_1, x_2], axis=0)
	print(X.shape)

	# Create labels
	y_1 = np.zeros(len(x_1))
	y_2 = np.ones(len(x_2))
	Y = np.concatenate([y_1, y_2])

	new_1 = np.random.multivariate_normal(np.array([-1,0]), sigma_1, 100)
	new_2 = np.random.multivariate_normal(np.array([3,0]), sigma_2, 100)
	new_points = np.vstack((new_1,new_2))
	print(new_points.shape)

	res = knn(X, Y, new_points, 3)

	# Plot sample
	plt.scatter(X[:, 0], X[:, 1], c=Y, marker='o', label = 'train')
	plt.scatter(new_points[:, 0], new_points[:, 1], c = res, marker = 'x', label = 'test')

	plt.legend()
	plt.show()
