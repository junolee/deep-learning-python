# forward propagation example
import numpy as np
import matplotlib.pyplot as plt

# 500 samples
Nclass = 500

# samples created from three clusters
X1 = np.random.randn(Nclass, 2) + np.array([0, -2])
X2 = np.random.randn(Nclass, 2) + np.array([2, 2])
X3 = np.random.randn(Nclass, 2) + np.array([-2, 2])
X = np.vstack([X1, X2, X3])

# correct labels for samples
Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)

# let's see what it looks like
plt.scatter(X[:,0], X[:,1], c=Y, s=100, alpha=0.5)
plt.show()

# randomly initialize weights
D = 2 # dimensionality of input
M = 3 # hidden layer size
K = 3 # number of classes
W1 = np.random.randn(D, M) # first weights
b1 = np.random.randn(M) # first bias
W2 = np.random.randn(M, K) # second weights
b2 = np.random.randn(K) # second bias

# forward action of neural network
def forward(X, W1, b1, W2, b2):
    Z = 1 / (1 + np.exp(-X.dot(W1) - b1)) # sigmoid nonlinearity in hidden layer

    # softmax of the next layer
    A = Z.dot(W2) + b2 # activation
    expA = np.exp(A) # exponentiate to make positive
    # normalize so probabilities sum to one for each class
    Y = expA / expA.sum(axis=1, keepdims=True)
    return Y

# determine the classification rate
# num correct / num total
def classification_rate(Y, P): # Y, P: true and predicted class labels
    n_correct = 0
    n_total = 0
    for i in xrange(len(Y)):
        n_total += 1
        if Y[i] == P[i]:
            n_correct += 1
    return float(n_correct) / n_total

# call forward action with the randomly initialized weights
P_Y_given_X = forward(X, W1, b1, W2, b2)
P = np.argmax(P_Y_given_X, axis=1)

# verify we chose the correct axis
assert(len(P) == len(Y)) # length of predictions = length of Y

# calculate classification rate
print "Classification rate for randomly chosen weights:", classification_rate(Y, P)

# This is not trained yet; classification rate is exactly 1/3 as expected
