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

model = load_model('model/model-loss_0.58909-acc_0.70160.h5')
pred = model.predict([X_test,Y_test])
pred = pred.reshape(2000,4)
print (pred)

np.set_printoptions(threshold=np.nan)
ans = np.argmax(pred,axis=1)
print (ans.shape)
output(ans,'loss_0.58909.csv')
