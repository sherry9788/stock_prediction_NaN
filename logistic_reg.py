import numpy as np
import pandas as pd
import glob
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing 
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
from scipy import interp
from itertools import cycle

# files in /new_vectors
################# training data ########################
all_files = glob.glob("new_vectors/training/*.csv")
df = pd.DataFrame()
list_ = []
for file_ in all_files:
    frame = pd.read_csv(file_,index_col=None, header=None)
    list_.append(frame)
df = pd.concat(list_, axis = 0)

#df = pd.concat((pd.read_csv(f, header = None) for f in all_files), axis=0)

#df = pd.read_csv("vectors/new_vectors_with_labels.csv", header = None)

X = df.iloc[:, :300]
y = df.iloc[:, 300]

# Binarize the output
lb = preprocessing.LabelBinarizer()
y = lb.fit_transform(y)
n_classes = y.shape[1]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33)

clf = OneVsRestClassifier(LogisticRegression())
clf.fit(X_train,y_train)
res = clf.predict(X_test)
print ('score Scikit learn: ', clf.score(X_test, y_test))
y_score = clf.predict_proba(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
fig = plt.figure()
lw = 2
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i+1, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()
fig.savefig('roc_logreg.png')



############## predicted labels (old/less accurate vectors) ###########################
## new
#for i in range(8):
#    X = df.iloc[i*100:(i+1)*100, :300]
#    y = df.iloc[i*100:(i+1)*100, 300]
#    pred = clf.predict(X)
#    pred = lb.inverse_transform(pred)
#    np.savetxt('result_log/result'+str(i)+'.csv', pred, fmt='%10.5f', delimiter=',') 
#    
#i = 8
#X = df.iloc[i*100:, :299]
#y = df.iloc[i*100:, 300]
#pred = clf.predict(X)
#pred = lb.inverse_transform(pred)
#np.savetxt('result_log/result'+str(i)+'.csv', pred, fmt='%10.5f', delimiter=',') 


############## predicted labels (new vectors) ###########################
i = 0
for vec in list_:
    X = vec.iloc[:, :300]
    pred = clf.predict(X)
    pred = lb.inverse_transform(pred)
    while(pred.size < 100):
        pred = np.append(pred, 2.0)
    np.savetxt('result_log/result'+str(i)+'.csv', pred, fmt='%10.5f', delimiter=',') 
    i += 1 

################# predicting data ########################
all_files_pred = glob.glob("new_vectors/predicting/*.csv")
df_pred = pd.DataFrame()
list_pred = []
for file_ in all_files_pred:
    frame = pd.read_csv(file_,index_col=None, header=None)
    list_pred.append(frame)
df_pred = pd.concat(list_pred, axis = 0)

############# predicted labels (last 3 days) ###################################
i = 0
for vec in list_pred:
    X = vec.iloc[:, :300]
    pred = clf.predict(X)
    pred = lb.inverse_transform(pred)
    while(pred.size < 100):
        pred = np.append(pred, 2.0)
    np.savetxt('result_log_pred/result'+str(i)+'.csv', pred, fmt='%10.5f', delimiter=',') 
    i += 1 

