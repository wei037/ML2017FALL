import pandas as pd
import numpy as np

from keras.models import Model , load_model
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, GRU, SimpleRNN , Bidirectional , Merge , Dot
from keras.layers import Input, Dense , RepeatVector , Flatten ,Dropout
from keras.layers.wrappers import TimeDistributed
from keras.losses import hinge , squared_hinge
import keras.backend as K
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

import time
from load_data import *

def _sen2vec(word_list , dictionary) :
    sen2vec = []
    for i in range(len(word_list)):
        tmp = np.array([np.zeros(300)]*13)
        idx = 0
        for word in word_list[i] :
            try:
                tmp[idx] = dictionary[word]
                idx += 1
            except Exception as e:
                continue
        sen2vec.append(tmp)
    return np.array(sen2vec)

def create_model():
    mfc_input = Input(shape = (246,39))
    mfc_ouput = GRU(256 , return_sequences=True , dropout=0.1, recurrent_dropout=0.1)(mfc_input)
    mfc_ouput1 = GRU(128 , return_sequences=True , dropout=0.1, recurrent_dropout=0.1)(mfc_ouput)
    mfc_ouput2 = GRU(128 , dropout=0.1, recurrent_dropout=0.1)(mfc_ouput1)

    chi_input = Input(shape = (13,300))
    chi_ouput = GRU(256 , return_sequences=True , dropout=0.1, recurrent_dropout=0.1)(chi_input)
    chi_ouput1 = GRU(128 , return_sequences=True , dropout=0.1, recurrent_dropout=0.1)(chi_ouput)
    chi_ouput2 = GRU(128 , dropout=0.1, recurrent_dropout=0.1)(chi_ouput1)

    encoder_output = Dot(axes = 1 , normalize = True)( [mfc_ouput2 , chi_ouput2] )

    model = Model(input=[mfc_input , chi_input], output=encoder_output)
    model.summary()

    return model

start = time.time()
print ('load training data...')
X_train = load_X_train('data/train.data')
Y_true , Y_fake = load_Y_train('data/train.caption')
print (Y_true[0])
print (Y_fake[0])
print ('loaded in ',time.time() - start)
start = time.time()
print ('create dict...')
dictionary = create_dict('fastText/wiki.zh.vec')
print ('created in ',time.time() - start)
start = time.time()
print ('padding sequence...')
Y_true = _sen2vec(Y_true , dictionary)
Y_fake = _sen2vec(Y_fake , dictionary)
X_train = sequence.pad_sequences(X_train, maxlen=246 , value=np.zeros(39) , padding='post' , dtype = 'float')
print ('padding in ',time.time() - start)

label_train = np.empty(90072)
chi_train = np.empty((90072,13,300))
mfc_train = np.empty((90072,246,39))
for i in range(45036):
    label_train[i*2]   = 1
    label_train[i*2+1] = 0
    chi_train[i*2]   = Y_true[i]
    chi_train[i*2+1] = Y_fake[i]
    mfc_train[i*2]   = X_train[i]
    mfc_train[i*2+1] = X_train[i]
    
print (mfc_train.shape)
print (chi_train.shape)
print (label_train.shape)

cb = []
cb.append(ModelCheckpoint('GRUmodel/model-loss_{val_loss:.5f}-acc_{val_acc:.5f}.h5', monitor='val_loss', save_best_only=True))

print ('cb done')
#model = create_model()
model = load_model('GRUmodel/model-loss_0.57340-acc_0.70315.h5')
model.compile(loss= 'binary_crossentropy', optimizer=Adam() , metrics = ['accuracy'])
model.fit([mfc_train , chi_train], label_train , batch_size=512, epochs=100 , validation_split = 0.05 , callbacks=cb)

model.save('GRU.h5')
