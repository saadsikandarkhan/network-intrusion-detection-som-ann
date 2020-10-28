# Artificial Neural Network

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import pandas as pd
import time
from sklearn import metrics
from sklearn.metrics import accuracy_score 
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score 
from sklearn.metrics import precision_score
from sklearn.utils import resample
import scipy


# Importing the dataset
#df = pd.read_csv('KDDTrain+_20Percent.csv', header=None)
df = pd.read_csv('KDDTrain+.csv', header=None)
dft = pd.read_csv('KDDtest.csv', header=None)
full = pd.concat([df, dft])


# The CSV file has no column heads, so add them

full.columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'outcome', 'score']


fulloutcome=full['outcome'].replace({ 
             'normal' : 0, 
             'neptune' : 3 ,
             'back': 3, #should be 3
             'land': 3, 
             'pod': 3,
             'smurf': 3,  
             'teardrop': 3,
             'mailbomb': 3, #should be 3
             'apache2': 3, #should be 3
             'processtable': 3, #should be 3
             'udpstorm': 3, #should be 3
             'worm': 3, # should be 3
             'ipsweep' : 1, #probe 
             'nmap' : 1, #probe
             'portsweep' : 1, #probe
             'satan' : 1, # probe
             'mscan' : 1, #probe
             'saint' : 1, #probe
             'ftp_write': 2, #R2L
             'guess_passwd': 2, #R2l
             'imap': 2, 
             'multihop': 2,
             'phf': 2,
             'spy': 2,
             'warezclient': 2,
             'warezmaster': 2,
             'sendmail': 2, #should be 2
             'named': 2, 
             'snmpgetattack': 2,
             'snmpguess': 2,
             'xlock': 2,
             'xsnoop': 2,
             'httptunnel': 2,
             'buffer_overflow': 4,
             'loadmodule': 4,
             'perl': 4,
             'rootkit': 4,
             'ps': 4,
             'sqlattack': 4,
             'xterm': 4}).astype(int)

del full['outcome']
del full['score']


full2 = pd. get_dummies (full , drop_first = True )


X_train=np.array(full2[0:df.shape[0]]) # !!drop the outcome column here
y_train=np.array(fulloutcome[0:df.shape[0]]) # !!drop the outcome column here
X_test=full2[df.shape[0]:]
y_test=fulloutcome[df.shape[0]:]           

y_train=fulloutcome[0:df.shape[0]] # !!drop the outcome column here

# Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)


from sklearn.preprocessing import normalize
X_train = normalize(X_train)
X_test = normalize(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer

classifier.add(Dense(units = 45, kernel_initializer = 'uniform', activation = 'relu', input_dim = X_train.shape[1]))
#classifier.add(Dense(units = 45, kernel_initializer = 'uniform', activation = 'relu', input_dim = 113))
classifier.add(Dropout(rate=0.2))

# Adding the second hidden layer
classifier.add(Dense(units = 45, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'softmax'))

# Compiling the ANN
start_timeprint = time.time()

classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

elapsed_time = time.time() - start_timeprint
print(elapsed_time, "seconds")


# Fitting the ANN to the Training set


import time
start_timeprint = time.time()

classifier.fit(X_train, y_train, batch_size = 10, epochs = 3) # epochs should be 15 as per gridsearch, all else is good


elapsed_time = time.time() - start_timeprint
print(elapsed_time, "seconds")

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results

start_timeprint = time.time()

y_pred = classifier.predict(X_test)
y_pred = y_pred.argmax(axis=-1)

#y_pred = (y_pred > 0.5)

elapsed_time = time.time() - start_timeprint
print(elapsed_time, "seconds")

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn import metrics

cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import multilabel_confusion_matrix
cm2 = multilabel_confusion_matrix(y_test, y_pred)

# Accuracy
accuracy_score(y_test, y_pred)

#Recall score: True Positive Rate
recall_score(y_test, y_pred, average = 'micro')

#F1 score: weighted average of precision and recall
f1_score(y_test, y_pred, average = 'micro')

#Precision: weighted average of precision and recall
precision_score(y_test, y_pred, average = 'micro')

# Print the precision and recall, among other metrics
cr = metrics.classification_report(y_test, y_pred, digits=3)


#kfold, cross validation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():    
    classifier = Sequential()
    classifier.add(Dense(units = 45, kernel_initializer = 'uniform', activation = 'relu', input_dim = X_train.shape[1]))
    classifier.add(Dropout(rate=0.2))
    classifier.add(Dense(units = 45, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'softmax'))
    classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 3)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = 1)
mean = accuracies.mean()
variance = accuracies.std()

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 3)
accuracies = cross_val_score(estimator = classifier, X = X_test, y = y_test, cv = 10, n_jobs = 1)
mean = accuracies.mean()
variance = accuracies.std()

# Tuning the ANN, Gridsearch#
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer):    
    classifier = Sequential()
    classifier.add(Dense(units = 45, kernel_initializer = 'uniform', activation = 'relu', input_dim = 87))
    classifier.add(Dense(units = 45, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'softmax'))
    classifier.compile(optimizer = optimizer, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [10, 20],
              'epochs': [5, 10, 15],
              'optimizer': ['adam', 'rmsprop']}
#create GridSearchCV instance below
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy' , n_jobs=1, cv = 10)
#Fitting GridSearchCV instance below
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_ 
best_accuracy = grid_search.best_score_








    

