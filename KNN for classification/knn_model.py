import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# input dataset
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
import seaborn as sns
from pylab import rcParams
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from sklearn import preprocessing

sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 14, 8

RANDOM_SEED = 42
LABELS = ["Normal", "Event"]

######### loading data
pandas2ri.activate()

readRDS = robjects.r['readRDS']
df = readRDS('testingkalman.rds')
df = pandas2ri.ri2py(df)
#df = df.as_matrix()

######## exploration
df = df.dropna()

### convert the Event column to 1 and 0
df = df*1
#original = df
event = df[df["EVENT"] == 1]
normal = df[df["EVENT"] == 0]

print("Number of events:",np.shape(event))
print("Number of normal:",np.shape(normal))
####### preparing the data

from sklearn.preprocessing import StandardScaler
df = df.drop(['Time'], axis=1)
"""
df = df.as_matrix()
from datetime import datetime
import time
fmt = '%Y-%m-%d %H:%M:%S'
for i in range(len(df)):
    dt = datetime.strptime(df[i,0],fmt)
    df[i,0] = int(time.mktime(dt.timetuple()))
#df[0,0] = int(dt)
original['Time'] = df[:,0]
df = original
print(df['Time'])
"""
#print(original['Time'])
# standard scale the data

# replace the missing data with mean, median, mode or any
#particular value
"""
imp = Imputer(missing_values="NaN", strategy='median', axis=0)
X = imp.fit_transform(X)
"""
X = df.drop(['EVENT'], axis=1)
Y = df['EVENT']

# splitting the dataset
#X_train, X_test, y_train, y_test = train_test_split(
#            X, Y, test_size = 0.2, random_state = 100, stratify = Y)
#y_train = y_train.ravel() # convert from 2d to 1d array
#y_test = y_test.ravel() # convert from 2d to 1d array
split =1
limit = int(round(len(df)*split))
train = df.iloc[0:limit,]
test = df.iloc[limit:-1,]

X_train = train.drop(['EVENT'], axis=1)
y_train = train['EVENT']

y_test = test['EVENT']
X_test = test.drop(['EVENT'], axis=1)

X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values
# create KNN model

#for k in range(9):
#    K_value = k + 1

K_value = 9
neigh = KNeighborsClassifier(n_neighbors = K_value, weights='uniform', algorithm='auto')
neigh.fit(X_train, y_train)


pandas2ri.activate()

readRDS = robjects.r['readRDS']
df = readRDS('testing.rds')
df = pandas2ri.ri2py(df)
#df = df.as_matrix()

######## exploration
df = df.dropna()

### convert the Event column to 1 and 0
df = df*1
#original = df
event = df[df["EVENT"] == 1]
normal = df[df["EVENT"] == 0]

print("Number of events:",np.shape(event))
print("Number of normal:",np.shape(normal))
####### preparing the data

from sklearn.preprocessing import StandardScaler
df = df.drop(['Time'], axis=1)
# standard scale the data
"""
df = df.as_matrix()
from datetime import datetime
import time
fmt = '%Y-%m-%d %H:%M:%S'
for i in range(len(df)):
    dt = datetime.strptime(df[i,0],fmt)
    df[i,0] = int(time.mktime(dt.timetuple()))
#df[0,0] = int(dt)
original['Time'] = df[:,0]

df = original
"""
#print(df['Time'])

# replace the missing data with mean, median, mode or any
#particular value

#imp = Imputer(missing_values="NaN", strategy='median', axis=0)
#X = imp.fit_transform(X)

X = df.drop(['EVENT'], axis=1)
Y = df['EVENT']

# splitting the dataset
#X_train, X_test, y_train, y_test = train_test_split(
#            X, Y, test_size = 0.2, random_state = 100, stratify = Y)
#y_train = y_train.ravel() # convert from 2d to 1d array
#y_test = y_test.ravel() # convert from 2d to 1d array
#split = 0
#limit = int(round(len(df)*split))
limit = 122334
train = df.iloc[0:limit,]
test = df.iloc[limit:-1,]

X_train = train.drop(['EVENT'], axis=1)
y_train = train['EVENT']

y_test = test['EVENT']
X_test = test.drop(['EVENT'], axis=1)

X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

y_pred = neigh.predict(X_test)
print "Accuracy is ", accuracy_score(y_test,y_pred)*100,"% for K-Value:",K_value

#### precision and recall
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
average_precision = average_precision_score(y_test, y_pred)

precision, recall, _ = precision_recall_curve(y_test, y_pred)

plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.show()
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))

##### calculate f1_score
from sklearn.metrics import f1_score
print("f1 score", f1_score(y_test, y_pred))

from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support)
CM = confusion_matrix(y_test, y_pred)

TN = CM[0][0]
FN = CM[1][0]
TP = CM[1][1]
FP = CM[0][1]

print("TP:",TP,"FP:",FP,"TN:",TN,"FN:",FN)
