import numpy as np
import pandas as pd
import sys

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Convolution1D, Flatten, Dropout
from keras.layers import LSTM, Embedding, GRU, SimpleRNN
from keras.initializers import Orthogonal
from keras.utils import np_utils
import keras.callbacks

from gensim.models import word2vec
import pickle

import time

maxlen = 60

def load_train(text) :
    train_label = pd.read_csv(text , sep = '\n' , header = None)
    train = []
    for i in range(train_label.shape[0]) :
        train.append(train_label.iloc[i,0].split(' +++$+++ '))
    return np.array(train)

def load_nolabel(text) :
    train_no_label = pd.read_csv(text , sep = '\n' , header = None)
    return np.array(train_no_label)   

def word_embedding(model,text) :
    sentences = []
    for i in range(len(text)) :
        sentences.append(text[i].split())
        
    sen2vec = []
    for i in range(len(sentences)) :
        tmp = np.array([np.zeros(128)]*40)
        idx = 0
        for word in sentences[i] :
            try :
                tmp[idx] = model[word]
                idx += 1
            except Exception as e:
                continue
        sen2vec.append(tmp)
    return np.array(sen2vec)

def create_model() :
    model = Sequential()
    model.add(GRU(64, return_sequences=True , dropout=0.15, recurrent_dropout=0.15,input_shape = (40,128)))
    model.add(GRU(32, return_sequences=True , dropout=0.15, recurrent_dropout=0.15))
    model.add(GRU(32, return_sequences=False , dropout=0.15, recurrent_dropout=0.15))
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(2, activation='softmax'))

    return model

def fit(model , X_train , Y_train ,  val_split = 0.5 , epochs=5 , batch_size = 256) :
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train , Y_train , validation_split = val_split , epochs=epochs , batch_size=batch_size)
    return model

def main() :
    #### load data and reshape
    wordmodel = word2vec.Word2Vec.load("wordDic128_semi.bin")
    train_label = load_train('data/training_label.txt')
    X_train_no_label = load_nolabel('data/training_nolabel.txt')
    X_train_label = np.array(train_label[:,1])
    Y_train_label = np.array(train_label[:,0])
    
    #### build model
    model = create_model()
    model.summary()
    
    num_data = 200000
    
    for semi in range(5) :
        print ('X_train_label : ' , len(X_train_label) , ' / Y_train_label : ' , len(Y_train_label))    
        idx = np.random.choice(len(X_train_label), num_data , replace = False)
        X_train = []
        X_train = word_embedding(wordmodel,X_train_label[idx])
        Y_train = np.array(Y_train_label[idx])
        Y_train = np_utils.to_categorical(Y_train, 2) 
        
        model = fit(model , X_train , Y_train , epochs = (15 - 3*semi) , val_split = 0.05)
        
        aug_data = X_train_no_label[semi*num_data:(semi+1)*num_data,0]
        X_train = []
        X_train = word_embedding(wordmodel, aug_data)    
        pred_list = model.predict(X_train)
        pred = np.argmax(pred_list , axis = 1) 
        idx1 = np.where(pred_list[:,1] >= 0.85)[0]
        idx0 = np.where(pred_list[:,1] <= 0.15)[0]
        label_idx = np.concatenate((idx1,idx0),axis = 0)
        
        print ('add {} number'.format(len(label_idx)))
        X_train_label = np.concatenate((X_train_label,aug_data[label_idx]),axis = 0) 
        Y_train_label = np.concatenate((Y_train_label , pred[label_idx]),axis = 0)
    
    idx = np.random.choice(len(X_train_label), num_data , replace = False)
    X_train = []
    X_train = word_embedding(wordmodel,X_train_label[idx])
    Y_train = np.array(Y_train_label[idx])
    print('X_train shape:', X_train.shape)
    Y_train = np_utils.to_categorical(Y_train, 2)        
    score = model.evaluate(X_train, Y_train, verbose=0)
    print('Train score:', score[0])
    print('Train accuracy;', score[1])
    model.save('RNN_semi.h5') 
    

if __name__ == "__main__":
    main()