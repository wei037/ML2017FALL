import numpy as np
import pandas as pd
import sys
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras import optimizers
from keras.utils import np_utils
import keras.callbacks
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from keras.models import model_from_json
from keras import regularizers

batch_size = 256
nb_classes = 2
nb_epoch   = 30
nb_data    = 106
log_filepath = './tmp/keras_log'


def scale(X):
    return (X - np.mean(X , axis = 0))/(np.std(X , axis = 0) + 1e-10)

def output(pred_list,text) :    
    out = [['id','label']]
    for i in range(len(pred_list)) :
        tmp = [str(i+1) , int(pred_list[i])]
        out.append(tmp)
    
    pd.DataFrame(out).to_csv(text , header = False , index = False) 

def save_model(model,text) :
    # serialize model to JSON
    model_json = model.to_json()
    with open(text+'.json', "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(text+'.h5')
    print("Saved model to disk")

def load_model(model_text,weight_text) :
    # load json and create model
    json_file = open(model_text, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weight_text)
    print("Loaded model from disk")
    return loaded_model

# load data and reshape

qua = [0,1,3,5]   #age,fnlwgt,capital_gain,capital_loss,hours_per_week
tri = [0,3]   #2 for sex
workclass          =  np.arange(9)  + 6 
education          =  np.arange(16) + 15
marital_status     =  np.arange(7)  + 31
occupation         =  np.arange(15) + 38
relationship       =  np.arange(6)  + 53
race               =  np.arange(5)  + 59
native_country     =  np.arange(42) + 64

X_train = pd.read_csv(sys.argv[1]).as_matrix().astype('float')
Y_train = pd.read_csv(sys.argv[2]).as_matrix().astype('float').reshape(-1,)
X_test = pd.read_csv(sys.argv[3]).as_matrix().astype('float')


features = np.concatenate([[0,1,2,3,4,5] , workclass , education , occupation , relationship ] )
X_train = X_train[:,features]
X_test = X_test[:,features]

X_train = np.concatenate([X_train , X_train[:,qua]**(2), X_train[:,tri]**(3)]  , axis = 1)
X_test = np.concatenate([X_test , X_test[:,qua]**(2) , X_test[:,tri]**(3)]  , axis = 1)

#print (X_train.shape)
nb_data = X_train.shape[1]

# rescale

X_train = scale(X_train)
X_test = scale(X_test)

# convert class vectors to binary class matrices (one hot vectors)

Y_train = np_utils.to_categorical(Y_train, nb_classes)

#old_session = KTF.get_session()




with tf.Graph().as_default():
    session = tf.Session('')
    KTF.set_session(session)
    KTF.set_learning_phase(1)
    # build model

    '''
    model = Sequential()
    model.add(Dense(512, input_shape=(nb_data,), init='normal',name='dense1' , kernel_regularizer=regularizers.l1(0.001)))
    model.add(Activation('relu', name='relu1' ))
    #model.add(Dropout(0.4, name='dropout1'))
    model.add(Dense(512, init='normal', name='dense2' ))
    model.add(Activation('relu', name='relu2'))
    #model.add(Dropout(0.2, name='dropout2'))
    model.add(Dense(256, init='normal', name='dense3' ))
    model.add(Activation('sigmoid', name='sigmoid1'))
    #model.add(Dropout(0.2, name='dropout2'))
    model.add(Dense(2, init='normal', name='dense4' ))
    model.add(Activation('softmax', name='softmax1'))
    model.summary()
    '''

    model = load_model('hw2_best.json','hw2_best.h5')
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.001), metrics=['accuracy'])
    '''
    tb_cb = keras.callbacks.TensorBoard(log_dir=log_filepath, histogram_freq=1)
    cbks = [tb_cb]
    #print ("----------------",cbks)
    history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch = nb_epoch, verbose=1, callbacks=None)
    '''
    score = model.evaluate(X_train, Y_train, verbose=0)
    print('Train score:', score[0])
    print('Train accuracy;', score[1])

    #save_model(model,'hw2_tri')

    pred = model.predict(X_test)[:,1]
    pred[pred >= 0.5] = 1
    print (np.sum(pred.astype('int')))
    output(pred.astype('int'),sys.argv[4])



#KTF.set_session(old_session)