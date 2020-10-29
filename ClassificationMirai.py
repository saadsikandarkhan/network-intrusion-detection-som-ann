# Artificial Neural Network

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import accuracy_score 
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score 
from sklearn.metrics import precision_score
from sklearn.utils import resample
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel


data_benign = pd.read_csv('benign_traffic.csv')
data_ack = pd.read_csv('ack.csv')
data_scan = pd.read_csv('scan.csv')
data_syn = pd.read_csv('syn.csv')
data_udp = pd.read_csv('udp.csv')

data_benign['class'] = 0
data_ack['class'] = 1
data_scan['class'] = 2
data_syn['class'] = 3
data_udp['class'] = 4

#Declaring outcome variables and converting to integer type
target_benign = data_benign.iloc[:, -1].astype(int)
target_ack = data_ack.iloc[:, -1].astype(int)
target_scan = data_scan.iloc[:, -1].astype(int)
target_syn = data_syn .iloc[:, -1].astype(int)
target_udp = data_udp.iloc[:, -1].astype(int)

#Organizng the data and target dataframes
data = np.vstack([data_benign, data_ack, data_scan, data_syn, data_udp])
df = pd.DataFrame(data)
target = np.hstack([target_benign, target_ack, target_scan, target_syn, target_udp]).astype(int)

#data = resample(data)
#target = resample(target)

data = np.delete(data, -1, axis=1)


#feature selection aka dimention reduction
clf = ExtraTreesClassifier()
clf.fit(data, target)
sfm = SelectFromModel(clf, prefit=True)
data = sfm.transform(data)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.25, random_state = 0)


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
#import keras
from tensorflow.keras import backend
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
#from keras.callbacks import EarlyStopping
#from keras.callbacks import ModelCheckpoint

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 21, kernel_initializer = 'uniform', activation = 'relu', input_dim = X_test.shape[1]))
classifier.add(Dropout(rate=0.2))


# Adding the second hidden layer
classifier.add(Dense(units = 21, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'softmax'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])


# Fitting the ANN to the Training set
import time
start_timeprint = time.time()

classifier.fit(X_train, y_train, batch_size = 10, epochs = 3)

elapsed_time = time.time() - start_timeprint
print(elapsed_time, "seconds")

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = y_pred.argmax(axis=-1)

#y_pred = y_pred.argmax(axis=-1)+1

#y_pred = (y_pred > 0.5)


from sklearn.metrics import multilabel_confusion_matrix
cm2 = multilabel_confusion_matrix(y_test, y_pred)

# Print the precision and recall, among other metrics
target_names = ['Benign', 'ACK', 'Scan', 'SYN', 'UDP']
cr = metrics.classification_report(y_test, y_pred, digits=3, target_names=target_names)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Accuracy
accuracy_score(y_test, y_pred)

#Recall score: True Positive Rate
recall_score(y_test, y_pred, average = 'micro')

#F1 score: weighted average of precision and recall
f1_score(y_test, y_pred, average = 'micro')

#Precision: weighted average of precision and recall
precision_score(y_test, y_pred, average = 'micro')


#kfold, cross validation
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
#from keras.models import Sequential
#from keras.layers import Dense
def build_classifier():    
    classifier = Sequential()
    classifier.add(Dense(units = 21, kernel_initializer = 'uniform', activation = 'relu', input_dim = X_test.shape[1]))
    classifier.add(Dropout(rate=0.2))
    classifier.add(Dense(units = 21, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'softmax'))
    classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 3)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = 1)
mean = accuracies.mean()
variance = accuracies.std()

# Tuning the ANN, Gridsearch#
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def build_classifier(optimizer):    
    classifier = Sequential()
    classifier.add(Dense(units = 21, kernel_initializer = 'uniform', activation = 'relu', input_dim = X_test.shape[1]))
    classifier.add(Dense(units = 21, kernel_initializer = 'uniform', activation = 'relu'))
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


