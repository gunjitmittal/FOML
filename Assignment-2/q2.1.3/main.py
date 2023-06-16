import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import scipy.io

img_data = scipy.io.loadmat("DrivFace/DrivFace.mat")
drivFaceData = img_data['drivFaceD'][0]
np.random.seed(0)

data = img_data['drivFaceD'][0][0][0]
data_squared = np.square(data)
final_data = np.concatenate((np.array(np.ones(606)).reshape(-1, 1), data, data_squared), axis=1)

label_data = pd.read_csv("DrivFace/drivPoints.txt")
label = np.array(label_data["xF"])

X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.5, random_state=0)
X_test = np.array(X_test)
X_train = np.array(X_train)
y_test = np.array(y_test)
y_train = np.array(y_train)


reg = LinearRegression().fit(X_train, y_train)
y_pred = reg.predict(X_test)
v1 = explained_variance_score(y_test, y_pred)
print(v1)

X3d_train, X3d_test, y3d_train, y3d_test = train_test_split(final_data, label, test_size=0.5, random_state=0)
X3d_test = np.array(X3d_test)
X3d_train = np.array(X3d_train)
y3d_test = np.array(y3d_test)
y3d_train = np.array(y3d_train)
# print(X3d_train.shape)

# y3d_pred = reg.predict(X3d_test[5, :, 2].reshape(1, -1))
# print(y3d_pred)

reg3d = LinearRegression().fit(X3d_train, y3d_train)
y3d_pred = reg3d.predict(X3d_test)
v2 = explained_variance_score(y3d_test, y3d_pred)
print(v2)

