import numpy as np
import pandas as pd

from keras.models import Model
from keras.layers import Dense, Input
from keras.callbacks import ModelCheckpoint
import keras.backend as K

from sklearn.cluster import KMeans

train = np.load('data/image.npy')
print (train.shape)
train = train.astype('float32') / 255.

def Models():
    encoding_dim = 32
    input_img = Input(shape=(784,))
    encode = Dense(384, activation='relu')(input_img)
    encoder_output = Dense(encoding_dim ,activation='relu')(encode)


    decode = Dense(384, activation='relu')(encoder_output)
    decoder_output = Dense(784, activation='sigmoid')(decode)

    auto_encoder = Model(input_img , decoder_output)
    encoder = Model(input_img , encoder_output)
    auto_encoder.summary()

    return auto_encoder , encoder

auto_encoder , encoder = Models()

cb = []
cb.append(ModelCheckpoint('model/model-loss_{val_loss:.5f}.h5', monitor='val_loss', save_best_only=True))

auto_encoder.compile(loss='binary_crossentropy', optimizer='adam')
auto_encoder.fit(train , train , epochs=500 , batch_size=512 , validation_split =0.05 , callbacks=cb)
