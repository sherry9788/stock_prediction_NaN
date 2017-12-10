import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split

df = pd.read_csv('11.29_vectors.csv', header=0)

X = df.iloc[:, :299]
y = df.iloc[:, 300]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33)

clf = LogisticRegression()
clf.fit(X_train,y_train)
res = clf.predict(X_test)
print ('score Scikit learn: ', clf.score(X_test, y_test))

np.savetxt('result_11-29.csv', res, delimiter=',') 

