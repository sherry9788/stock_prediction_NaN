import numpy as np
import pandas as pd
import glob
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier

all_files = glob.glob("vectors/*.csv")
df = pd.concat((pd.read_csv(f, header = None) for f in all_files), axis=0)

X = df.iloc[:, :299]
y = df.iloc[:, 300]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1)

clf = RandomForestClassifier(max_depth=2, random_state=0)

clf.fit(X_train,y_train)
res = clf.predict(X_test)
print ('score Scikit learn: ', clf.score(X_test, y_test))

np.savetxt('result_11-29.csv', res, delimiter=',') 
