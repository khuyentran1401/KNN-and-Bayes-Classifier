import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.neighbors import KNeighborsClassifier

def optimal_bayes(mu1, mu0, sigma, X):

	l = np.linalg.inv(sigma) @ (mu1-mu0)

	return np.array([1 if l.T@x > 1 else 0 for x in X])

def error(pred, y):

	n = len(y)

	err = 0
	for i in range(n):
		err += np.abs(pred[i] - y[i])

	return err/n 

def data_error(N):
	# Multivariate normal sample
	x_0 = np.random.multivariate_normal(mu_0, sigma, N)
	x_1 = np.random.multivariate_normal(mu_1, sigma, N)
		# Concatenate samples
	X = np.concatenate([x_0, x_1], axis=0)

	pred = optimal_bayes(mu_1, mu_0, sigma, X)

	# Create labels
	y_0 = np.zeros(len(x_0))
	y_1 = np.ones(len(x_1))
	Y = np.concatenate([y_0, y_1])

	return error(pred, Y)


def plot(N, k, ax):


	# Multivariate normal sample
	x_0 = np.random.multivariate_normal(mu_0, sigma, N)
	x_1 = np.random.multivariate_normal(mu_1, sigma, N)
		# Concatenate samples
	X = np.concatenate([x_0, x_1], axis=0)

	pred = optimal_bayes(mu_1, mu_0, sigma, X)

	# Create labels
	y_0 = np.zeros(len(x_0))
	y_1 = np.ones(len(x_1))
	Y = np.concatenate([y_0, y_1])


	xlim = (np.amin(X[:, 0])-1, np.amax(X[:,0])+1)
	ylim = (np.amin(X[:, 1])-1, np.amax(X[:,1])+1)

	xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], N),
		np.linspace(ylim[0], ylim[1], N))

	c = np.c_[xx.ravel(), yy.ravel()]

	bayes =optimal_bayes(mu_1, mu_0, sigma, c).reshape(-1,N)

	kneigh = KNeighborsClassifier(n_neighbors=k).fit(X, Y)
	kneigh_z = kneigh.predict(c).reshape(-1,N)


	#print(xx.shape, kneigh.predict(c).shape)

	#------------------------------------------------------------
	# Plot the results
	#fig = plt.figure(figsize=(5, 3.75))#------------------------------------------------------------
	#ax = fig.add_subplot(111)
	ax.scatter(X[:, 0], X[:, 1], c=Y, zorder=2)

	con1 = ax.contour(xx, yy, bayes, [0.5], colors='g')
	con2 = ax.contour(xx, yy, kneigh_z, [0.5], colors='r')

	h1, _ = con1.legend_elements()
	h2, _ = con2.legend_elements()

	ax.set_title('Size: {}, k: {}'.format(N, k))

	ax.legend([h1[0], h2[0]], ['Bayes', 'KNN'])

	ax.set_xlim(xlim)
	ax.set_ylim(ylim)

if __name__ == '__main__':

	np.random.seed(42)

	mu_0 = np.array([0, 0], dtype=np.float32)
	mu_1 = np.array([1, 2], dtype=np.float32)

	# Sigma
	sigma= np.array([[1, 0.4],
	                    [0.4, 1]], dtype=np.float32)


	# Number of samples
	N = [50, 100, 500]
	k = [1, 3, 5]

	fig, axs = plt.subplots(3,3)

	for i in range(3):
		for j in range(3):
			plot(N[i], k[j], axs[i,j])

	plt.show()

	print('Error of data of size {} is {}'.format(N[2],data_error(N[2])))




