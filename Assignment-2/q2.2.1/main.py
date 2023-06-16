import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression

xls = pd.ExcelFile('LSVT_voice_rehabilitation/LSVT_voice_rehabilitation.xlsx')
df1 = pd.read_excel(xls, 'Data')
df2 = pd.read_excel(xls, 'Binary response')

X = df1.to_numpy()
X_squared = np.square(X)
X_ones = np.ones(X.shape)
X3d = np.concatenate(((np.array(np.ones(126)).reshape(-1, 1)), X, X_squared), axis=1)
Y = df2.to_numpy().ravel()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=0)

per = Perceptron().fit(X_train, Y_train)
v1 = per.score(X_test, Y_test)
print(v1)

lor = LogisticRegression(random_state=0, penalty='none', max_iter=1000).fit(X_train, Y_train)
v2 = lor.score(X_test, Y_test)
print(v2)

X3d_train, X3d_test, Y_train, Y_test = train_test_split(X3d, Y, test_size=0.5, random_state=0)

per3d = Perceptron().fit(X3d_train, Y_train)
v3 = per3d.score(X3d_test, Y_test)
print(v3)

lor3d = LogisticRegression(random_state=0, penalty='none', max_iter=500).fit(X3d_train, Y_train)
v4 = lor3d.score(X3d_test, Y_test)
print(v4)
