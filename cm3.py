###############################################################################
# CS 155 Kaggle Competition Code
# CM3
###############################################################################

import csv
import numpy as np 
import tensorflow as tf 
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.constraints import maxnorm, unitnorm
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.advanced_activations import LeakyReLU, PReLU
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.layers.convolutional import Convolution1D
from keras.optimizers import SGD

tf.python.control_flow_ops = tf

#The first training data is read and the x and y values 
#are separated into two arrays X and y. Also, the training
#and testing data sets are separated as specified.
with open('train_2008.csv','r') as dest_f: 
    data_iter = csv.reader(dest_f) 
    data = [data for data in data_iter]
    data.pop(0)
    data_array = np.asarray(data, dtype=int)    
    np.random.shuffle(data_array)
    training_data = data_array[:50000, 3:]
    np.random.shuffle(training_data)
    X_train = training_data[:,0:379]
    X_train = X_train.astype('float')
    Y_train = training_data[:,379] - 1
    Y_train = (Y_train==1).astype(np.int)
    X_test = data_array[50001:,3:382]
    Y_test = data_array[50001:,382] - 1
    X_test = X_test.astype('float')
    Y_test = (Y_test==1).astype(np.int)
    
#The model uses 200 hidden units, focusing on a wider (more nodes) system 
#near the inputs and gradually a deeper system near the outputs. In addition,
#some data were dropped out at the beginning and middle. ReLU was used as the
#nonlinearity three times and the final activation was performed with softmax.
def create_baseline():
	# create model
    model = Sequential()
    act = PReLU(init='zero', weights=None)
    model.add(Dense(200, input_dim=379, init='uniform', activation='relu',
                    W_constraint=maxnorm(1)))
    model.add(act)
    model.add(Dense(100, input_dim=200, init='uniform', 
                    W_constraint=maxnorm(1)))
    act = PReLU(init='zero', weights=None)
    model.add(act)
    model.add(Dense(1, init='uniform', activation='sigmoid'))
    # Compile model
    sgd = SGD(lr=0.01, momentum=0.8, decay=0.0, nesterov=False)
    model.compile(loss='binary_crossentropy', optimizer=sgd, 
                  metrics=['accuracy'])
    return model

# evaluate model with standardized dataset
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, 
                            nb_epoch=55, batch_size=2048, verbose=1)))
pipeline = Pipeline(estimators)

pipeline.fit(X_train, Y_train)

pipeline.score(X_test, Y_test)