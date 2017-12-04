import numpy as np
import pandas as pd
import sys

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential , load_model
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.utils import np_utils
import keras.callbacks


import pickle

import time

def output(pred_list,text) :    
    out = [['id','label']]

    for i in range(len(pred_list)) :
        tmp = [str(i) , int(pred_list[i])]
        out.append(tmp)
    
    pd.DataFrame(out).to_csv(text , header = False , index = False) 


def predict(text , data) :    
    model = load_model(text)

    pred_list = model.predict(data)
    pred = np.argmax(pred_list , axis = 1) 

    return pred

def load_test(text) :
	test = pd.read_csv(text , sep = '\n').as_matrix()
	for i in range(test.shape[0]) :
		test[i] = test[i,0].split(',' , 1)[1]
	return test

def main() :
    X_test = load_test('data/testing_data.txt')
    with open('tokenizer_RNN_bow.pickle', 'rb') as handle:
	    T = pickle.load(handle)
    X_test = np.array(T.texts_to_sequences(X_test[:,0]))
    X_test = sequence.pad_sequences(X_test, maxlen=60)
    print('X_test shape:', X_test.shape)

    pred = predict('RNN_bow.h5' , X_test)
    output(pred.astype('int') , 'RNN_bow.csv')

if __name__ == "__main__":
    main()
