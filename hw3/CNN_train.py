import numpy as np
import pandas as pd
import sys

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation , Flatten
from keras.layers import Conv2D, MaxPooling2D , LeakyReLU , BatchNormalization
from keras import optimizers
from keras.utils import np_utils
import keras.callbacks
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator

import load_data
import time

def create_CNN_model() :
    depth = 32
    model = Sequential()
    model.add(Conv2D(depth*1, (5, 5), activation = 'relu' , padding = 'valid' , input_shape=(48,48,1)))
    model.add(Conv2D(depth*1, (5, 5), activation = 'relu' , padding = 'same'))       
    model.add(Conv2D(depth*1, (5, 5), activation = 'relu' , padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3) , strides = (2,2)))

    model.add(Conv2D(depth*2, (5, 5), activation = 'relu' , padding = 'same'))
    model.add(Conv2D(depth*2, (3, 3), activation = 'relu' , padding = 'same'))
    model.add(Conv2D(depth*2, (3, 3), activation = 'relu' , padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3) , strides = (2,2)))

    model.add(Conv2D(depth*3, (3, 3), activation = 'relu' , padding = 'same'))
    model.add(Conv2D(depth*3, (3, 3), activation = 'relu' , padding = 'same'))
    model.add(Conv2D(depth*3, (3, 3), activation = 'relu' , padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2) , strides = (2,2)))

    model.add(Conv2D(depth*4, (3, 3), activation = 'relu' , padding = 'same')) 
    model.add(Conv2D(depth*4, (3, 3), activation = 'relu' , padding = 'same'))
    model.add(Conv2D(depth*4, (3, 3), activation = 'relu' , padding = 'same'))  
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2) , strides = (2,2)))

    model.add(Flatten(name = 'Flatten'))
    model.add(Dense(1024 , activation = 'relu'))
    model.add(Dense(512 , activation = 'relu'))
    model.add(Dense(256 , activation = 'relu'))
    model.add(Dense(128 , activation = 'relu'))
    model.add(Dense(7, init='normal'))
    model.add(Activation('softmax', name='softmax1'))
    model.summary()
    return model

def fit(model , X_train , Y_train , augment = 0 , epochs = 20 , val_split = 0.0 , batch_size = 128) :
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam() , metrics=['accuracy'])
    if augment == 1 :
        tb_cb = keras.callbacks.TensorBoard(log_dir='./tmp/keras_log', histogram_freq=0 , write_graph = True , write_images = True)
        cbks = [tb_cb]

        #### generate data
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True)
        datagen.fit(X_train)

        #### validation data
        X_val , Y_val = X_train[:5000,:] , Y_train[:5000,:]    
        #X_train , Y_train = X_train[5000:,:] , Y_train[5000:,:]

        #'./augment_image'
        history = model.fit_generator(datagen.flow(X_train, Y_train,batch_size=batch_size , save_to_dir = None),
                            steps_per_epoch=X_train.shape[0] / batch_size,
                            validation_data=(X_val, Y_val),
                            epochs=epochs,
                            verbose=1, 
                            callbacks=None)
    else :
        if val_split == 0.0 :
            tb_cb = keras.callbacks.TensorBoard(log_dir='./tmp/keras_log', histogram_freq=0 , write_graph = True , write_images = True)
            cbks = [tb_cb]
        else :
            tb_cb = keras.callbacks.EarlyStopping(monitor ='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
            cbks = [tb_cb]

        history = model.fit(X_train, Y_train, batch_size=batch_size, epochs = epochs, validation_split=val_split, verbose=1, callbacks=None)

    return model

def main() :
    #### load data and reshapeimport sys

    X_train , Y_train = load_data.load(sys.argv[1] , 1)
    X_train = X_train.reshape(-1, 48, 48, 1)

    #### rescale
    X_train = X_train / 255

    print ('X_train shape : ' , X_train.shape)
    print ('Y_train shape : ' , Y_train.shape)

    #### convert class vectors to binary class matrices (one hot vectors)
    Y_train = np_utils.to_categorical(Y_train, 7)


    #### build model
    model = create_CNN_model()
    #model = fit(model , X_train , Y_train , epochs = 50 , val_split = 0.2)
    model = fit(model , X_train , Y_train , augment = 1 , epochs = 60 , val_split = 0.2)
    score = model.evaluate(X_train, Y_train, verbose=0)
    print('Train score:', score[0])
    print('Train accuracy;', score[1])
    model.save('65700.h5')

if __name__ == "__main__":
    main()
