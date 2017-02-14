# Kaggle Competition CS 155
# CM 4


#from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier
#from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

#num_train = 64667 # Number of training points we want to use

# Load in training data. This takes a few seconds...
data = np.loadtxt('train_2008.csv', delimiter=',', skiprows=1)

# Load in training data. This takes a few seconds...
#test = np.loadtxt('test_2012.csv', delimiter=',', skiprows=1)

# Load in testing data.
#test = np.loadtxt('test_2008.csv', delimiter=',', skiprows=1)

X = data[:, :len(data[0])-1] # Load in X training data
y = np.ravel(data[:, -1:]) # Load in y training data
#X_test = test[:, :len(data[0])] # Load in X testing data
t_error, val_error = [], []
n_est = np.arange(100, 800, 100)
for i in n_est:
	clf = ExtraTreesClassifier(n_estimators=500, min_samples_split=4)
	val_error.append(cross_val_score(clf, X, y))
	clf.fit(X, y)
	pred = clf.predict(X)
	score = np.mean(pred == y)
	t_error.append(score)


'''
scores = cross_val_score(clf, X, y)
print("Score: {}".format(scores.mean()))

# Predict y_test
clf.fit(X, y)

pred = clf.predict(X)

'''