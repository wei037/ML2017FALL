import numpy as np
import pandas as pd
from DataReader import *

from keras.layers import Embedding, Dropout, add, Dot, Input, Flatten
from keras.models import Model
import keras.backend as K

import sys

Reader = DataReader()
X_train , Y_train , max_userid , max_movieid = Reader.read_train(sys.argv[1])

User = X_train[0]
Movie = X_train[1]
X_train = [User , Movie]


def CFmodel(n_users , m_items , k_factors) :
    U_input = Input(shape=(1,))
    U = Embedding(n_users, k_factors)(U_input)
    U = Flatten()(U)
    U = Dropout(0.1)(U)
    U_bias = Embedding(n_users, 1)(U_input)
    U_bias = Flatten()(U_bias)
    U_bias = Dropout(0.1)(U_bias)
    
    M_input = Input(shape=(1,))
    M = Embedding(m_items, k_factors)(M_input)
    M = Flatten()(M)
    M = Dropout(0.1)(M)
    M_bias = Embedding(m_items, 1)(M_input)
    M_bias = Flatten()(M_bias)
    M_bias = Dropout(0.1)(M_bias)
    
    out = Dot(axes = 1)([U,M])
    out = add([out , U_bias , M_bias])
    model = Model([U_input,M_input],out)
    
    model.summary()
    return model


def rmse(y_true, y_pred): 
    return K.sqrt( K.mean((y_pred - y_true)**2) )

model = CFmodel(max_userid+1 , max_movieid+1 , 128)
model.compile(optimizer='adam', loss=rmse)

model.fit(X_train, Y_train, epochs=10, validation_split=.1 , batch_size = 256)
model.save('MF.h5')
