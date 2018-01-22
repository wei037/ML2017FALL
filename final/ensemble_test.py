import pandas as pd
import numpy as np

from keras.models import Model , load_model
from keras.preprocessing import sequence
import keras.backend as K
from keras.callbacks import ModelCheckpoint

import time
from load_data import *

####testing###
def output(pred_list,text) :    
    out = [['id','answer']]

    for i in range(len(pred_list)) :
        tmp = [str(i+1) , int(pred_list[i])]
        out.append(tmp)
    
    pd.DataFrame(out).to_csv(text , header = False , index = False) 

def out_rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

start = time.time()
print ('create dict...')
dictionary = create_dict('fastText/wiki.zh.vec')
print ('created in ',time.time() - start)

X_test = load_X_test("data/test.data")
X_test = sequence.pad_sequences(X_test, maxlen=246 , value=np.zeros(39) , padding = 'post' , dtype = 'float')
X_test = np.repeat(X_test, 4 ,axis = 0)
print ('X_test : ', X_test.shape)

Y_test = load_Y_test('data/test.csv',dictionary)
Y_test = np.array(Y_test)
Y_test = Y_test.reshape(8000,13,300)
print ('Y_test : ', Y_test.shape)

model1 = load_model('LSTMmodel/model-loss_0.59091-acc_0.70149.h5')
model2 = load_model('LSTMmodel/model-loss_0.58909-acc_0.70160.h5')
model3 = load_model('GRUmodel/model-loss_0.58003-acc_0.69316.h5')
model4 = load_model('GRUmodel/model-loss_0.57336-acc_0.69671.h5')
start = time.time()
pred1 = model1.predict([X_test,Y_test])
print ('model1 pred in ',time.time() - start)
start = time.time()
pred2 = model2.predict([X_test,Y_test])
print ('model2 pred in ',time.time() - start)
start = time.time()
pred3 = model3.predict([X_test,Y_test])
print ('model3 pred in ',time.time() - start)
start = time.time()
pred4 = model4.predict([X_test,Y_test])
print ('model4 pred in ',time.time() - start)
pred = pred1 + pred2 + pred3 + pred4
pred = pred.reshape(2000,4)

#np.set_printoptions(threshold=np.nan)
print (pred)
ans = np.argmax(pred,axis=1)
print (ans.shape)
output(ans,'best.csv')
