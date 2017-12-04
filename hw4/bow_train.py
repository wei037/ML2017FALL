import numpy as np
import pandas as pd
import sys

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Convolution1D, Flatten, Dropout
from keras.layers import LSTM, Embedding, GRU, SimpleRNN
from keras.utils import np_utils
import keras.callbacks

import pickle

import time

maxlen = 60

def load_train(text) :
	train_label = pd.read_csv(text , sep = '\n' , header = None)
	train = []
	for i in range(train_label.shape[0]) :
	    train.append(train_label.iloc[i,0].split(' +++$+++ '))
	return np.array(train)    

def create_model() :
  model = Sequential()
  model.add(Embedding(20000, 128, input_length=maxlen))
  model.add(GRU(64, return_sequences=True , dropout=0.15, recurrent_dropout=0.15,input_shape = (40,128)))
  model.add(GRU(32, return_sequences=True , dropout=0.15, recurrent_dropout=0.15))
  model.add(GRU(32, return_sequences=False , dropout=0.15, recurrent_dropout=0.15))
  model.add(Dense(128,activation='relu'))
  model.add(Dropout(0.25))
  model.add(Dense(64,activation='relu'))
  model.add(Dropout(0.25))
  model.add(Dense(2, activation='softmax'))

  model.add(Dense(2, activation='sigmoid'))
  model.summary()

  return model

def create_tokenizer(data):
	T = Tokenizer(num_words = 20000)
	T.fit_on_texts(data)
	X_train = np.array(T.texts_to_sequences(data))
	# saving
	with open('tokenizer_RNN_bow.pickle', 'wb') as handle:
	    pickle.dump(T, handle, protocol=pickle.HIGHEST_PROTOCOL)

	return X_train

def fit(model , X_train , Y_train ,  val_split = 0.5 , epochs=5 , batch_size = 256) :
	model.compile(loss='binary_crossentropy', optimizer='adam',	metrics=['accuracy'])
	model.fit(X_train , Y_train , validation_split = val_split , epochs=epochs , batch_size=batch_size)
	return model

def main() :
    #### load data and reshape
    train_label = load_train('data/training_label.txt')
    X_train = create_tokenizer(train_label[:,1])
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    Y_train = np.array(train_label[:,0])
    print('x_train shape:', X_train.shape)

    #### convert class vectors to binary class matrices (one hot vectors)
    Y_train = np_utils.to_categorical(Y_train, 2)

    #### build model
    model = create_model()
    model = fit(model , X_train , Y_train , epochs = 3 , val_split = 0.0)
    score = model.evaluate(X_train, Y_train, verbose=0)
    print('Train score:', score[0])
    print('Train accuracy;', score[1])
    model.save('RNN_bow.h5')

if __name__ == "__main__":
    main()