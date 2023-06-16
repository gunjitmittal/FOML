import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_spd_matrix
np.random.seed(0)

A = make_spd_matrix(10, random_state=0)
b = np.random.rand(10)
c = np.random.rand()
norm_A = np.linalg.norm(A, ord=2)
norm_b = np.linalg.norm(b)

v1 = b.T.dot(np.linalg.inv(A))
f_an = v1.T.dot(A).dot(v1) - 2*b.T.dot(v1) + c

v2 = np.ones(10)/10
f_gd = []
step_size_gd = 1 / (norm_b + 2* norm_A)
max_iterations_gd = 1000
for i in range(max_iterations_gd):
    der = np.array((A.dot(v2) - b.T)*2)
    v2 = v2 - der*step_size_gd
    f_gd.append(v2.T.dot(A).dot(v2) - 2*b.T.dot(v2) + c)

v3 = np.ones(10)/10
f_sgd = []
step_size_sgd = step_size_gd/100
max_iterations_sgd = 100000
for i in range(max_iterations_sgd):
    der = np.array((A.dot(v3) - b.T)*2) + 0.5*np.random.randn(10)
    v3 = v3 - der*step_size_sgd
    f_sgd.append(v3.T.dot(A).dot(v3) - 2 * b.T.dot(v3) + c)
# print(v3)

f_gd = np.array(f_gd)
f_sgd = np.array(f_sgd)
plt.subplot(2, 1, 1)
plt.plot(np.arange(max_iterations_gd), f_gd, label="Gradient Descent")
plt.plot(np.arange(max_iterations_gd), np.ones(max_iterations_gd)*f_an, label="Analytical solution")
plt.xlabel("No. of iterations")
plt.ylabel("f(v)")
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(np.arange(max_iterations_sgd), f_sgd, label="Stochastic Gradient Descent")
plt.plot(np.arange(max_iterations_sgd), np.ones(max_iterations_sgd)*f_an, label="Analytical solution")
plt.xlabel("No. of iterations")
plt.ylabel("f(v)")
plt.subplots_adjust(bottom=0.1, top=1)
plt.legend()
plt.show()
