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

from gensim.models import word2vec
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

def word_embedding(model,text) :
    
    sentences = []
    for i in range(len(text)) :
        sentences.append(text[i,0].split())
    
    sen2vec = []
    most = 0
    for i in range(len(sentences)) :
        tmp = np.array([np.zeros(128)]*40)
        idx = 0
        for word in sentences[i] :
            try :
                tmp[idx] = model[word]
                idx += 1
            except Exception as e:
                continue
                #print( word , 'is not in dictionary')
        sen2vec.append(tmp)
    return np.array(sen2vec)

def main() :
    X_test = load_test('data/testing_data.txt')  
    print ('Load test done...')
    dict_model = word2vec.Word2Vec.load("wordDic128_semi.bin")
    print ('Load model done...')
    X_test_embedding = word_embedding(dict_model , X_test)
    print('X_test shape:', X_test_embedding.shape)

    pred = predict('RNN.h5' , X_test_embedding)
    print ('Predict done...')    
    output(pred.astype('int') , 'RNN.csv')

if __name__ == "__main__":
    main()