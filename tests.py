import matplotlib.pyplot as plt
import numpy as np
from river.ensemble.aggregated_mondrian_forest import AMFRegressor
from river import stream
from sklearn import datasets
from river.linear_model import LinearRegression
from river.ensemble.adaptive_random_forest import AdaptiveRandomForestRegressor
X, y = datasets.make_regression(n_samples=500, n_features=1, noise = 0.3)

#X = np.random.randn(500)
#X = list(map(lambda el:[el], X))
#y = np.sin(X)
#y = [item for sublist in y for item in sublist]



total_samples = 500  # set the amount of samples to iterate through
proportion_training = 1  # proportion of training samples

train_samples = proportion_training * total_samples  # set the amount of learning samples
test_samples = (1 - proportion_training) * total_samples
X_test, y_pred = [], []  # arrays for the plot

amf = AMFRegressor(n_estimators=10, step=1.0, use_aggregation=True, seed=None)
#amf =LinearRegression()
#amf = AdaptiveRandomForestRegressor(n_models=10, aggregation_method='mean')
t = 0
for x_t, y_t in stream.iter_array(X, y):
    if t < train_samples:
        amf.learn_one(x_t, y_t)  # learning sample (x_t, y_t)
    else:
        pred = amf.predict_one(x_t)
        X_test.append(x_t)
        y_pred.append(pred)

    t += 1
    if t > total_samples:
        break

for x_t, y_t in stream.iter_array(X, y):
    pred = amf.predict_one(x_t)
    X_test.append(x_t)
    y_pred.append(pred)

X_test_list = []
for i in X_test :
    X_test_list.append(list(i.values()))

X, y = np.array(X), np.array(y)
X_test_list, y_pred = np.array(X_test_list), np.array(y_pred)
plt.scatter(X,y)

plt.scatter(X_test_list,y_pred, color = 'r')
plt.show()