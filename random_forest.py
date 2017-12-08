from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt
import pandas as pd

df = genfromtxt('my_file.csv', delimiter=',')
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75

train, test = df[df['is_train']==True], df[df['is_train']==False]

clf = RandomForestClassifier(max_depth=2, random_state=0)

features = df.columns[:4]
y = train['label']
clf.fit(train[features], y)
clf.predict(test[features])

pd.crosstab(test['label'], preds, rownames=['Actual label'], colnames=['Predicted label'])
