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
df = readRDS('testing.rds')
df = pandas2ri.ri2py(df)
#df = df.as_matrix()

######## exploration
df = df.dropna()

### convert the Event column to 1 and 0
df = df*1
original = df

event = df[df["EVENT"] == 1]
normal = df[df["EVENT"] == 0]

print("Number of events:",np.shape(event))
print("Number of normal:",np.shape(normal))
####### preparing the data
df = df.as_matrix()
from datetime import datetime
import time
fmt = '%Y-%m-%d %H:%M:%S'
for i in range(len(df)):
    dt = datetime.strptime(df[i,0],fmt)
    df[i,0] = int(time.mktime(dt.timetuple()))
#df[0,0] = int(dt)
original['Time'] = df[:,0]
print(original['Time'])
