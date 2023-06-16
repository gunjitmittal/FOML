import numpy as np
from sklearn.datasets import make_spd_matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

np.random.seed(0)
m = 1000
ev = []
for d in range(1, 10000):
    cov = make_spd_matrix(d, random_state=0)

    x = np.random.multivariate_normal(mean=np.random.rand(d), cov=cov, size=2*m)
    y = []

    w1 = np.ones(d)
    for i in range(2*m):
        mu = w1.dot(x[i])
        y.append(np.random.normal(mu, 1))

    y = np.array(y)

    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.5, random_state=0)

    reg = LinearRegression().fit(X_train, Y_train)
    Y_pred = reg.predict(X_test)
    ev.append(explained_variance_score(y_pred=Y_pred, y_true=Y_test))

plt.plot(range(1, 100), ev, 'o')
plt.xlabel("Dimensions(d)")
plt.ylabel("Explained Variance")
plt.grid()
plt.show()