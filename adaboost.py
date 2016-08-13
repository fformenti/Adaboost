import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_hastie_10_2
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

def models_N_weights(X, y, M):
	model = []
	model_weights = []
	training_errors = []

	N, _ = X.shape
	w = np.ones(N) / N

	for m in range(M):
		h = DecisionTreeClassifier(max_depth=1)
		h.fit(X, y, sample_weight=w)
		pred = h.predict(X)

		eps = w.dot(pred != y)
		alpha = (np.log(1 - eps) - np.log(eps)) / 2
		w = w * np.exp(- alpha * y * pred)
		w = w / w.sum()

		model.append(h)
		model_weights.append(alpha)

	return [model, model_weights]


def predict_joined_models(X, model, model_weights):
	N, _ = X.shape
	y = np.zeros(N)
	for (h, alpha) in zip(model, model_weights):
		y = y + alpha * h.predict(X)
	y = np.sign(y)

	return y


def error_func(y, y_hat):
	correct_pred = (y_hat + y)/2 * (y_hat + y)/2
	Err = 1 - sum(correct_pred)/len(correct_pred)
	return Err


X, y = make_hastie_10_2(n_samples=1200, random_state=0)
X_train = X[:1000]
X_test = X[1000:1200]
y_train = y[:1000]
y_test = y[1000:1200]


M = 100
M_list = []
train_err_list = []
test_err_list = []
for m in range(M):

	model_fit = models_N_weights(X_train, y_train, m+1)
	y_hat = predict_joined_models(X_train, model_fit[0], model_fit[1])
	err = error_func(y_train, y_hat)
	train_err_list.append(err)

	y_hat = predict_joined_models(X_test, model_fit[0], model_fit[1])
	err = error_func(y_test, y_hat)
	test_err_list.append(err)
	M_list.append(m)


plt.plot(M_list, train_err_list, c= 'red', linestyle='-')
plt.plot(M_list, test_err_list, c= 'green', linestyle='-')
plt.xlabel('number of weak learners')
plt.ylabel('Error')
plt.title('Error x Number of models')
plt.show()





