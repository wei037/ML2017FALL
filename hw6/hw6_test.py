import numpy as np
import pandas as pd
import sys

from keras.models import load_model
import keras.backend as K

from sklearn.cluster import KMeans

train = np.load(sys.argv[1])
test = pd.read_csv(sys.argv[2]).as_matrix()
train = train.astype('float32') / 255.

def output(pred_list,text) :    
    out = [['ID','Ans']]

    for i in range(len(pred_list)) :
        tmp = [str(i) , pred_list[i]]
        out.append(tmp)
    
    pd.DataFrame(out).to_csv(text , header = False , index = False) 

autoencoder = load_model('best.h5')
#autoencoder.summary()
encoder = K.function([autoencoder.get_layer('input_1').input], [autoencoder.get_layer('dense_1').output])

pred = encoder([train])

kmeans = KMeans(n_clusters=2, random_state=0).fit(pred[0])
labels = kmeans.labels_

print (sum(labels))

out = []
for i in range(1980000) :
    if labels[test[i][1]] == labels[test[i][2]] :
        out.append(1)
    else : 
        out.append(0)
out = np.array(out)

output(out , sys.argv[3])
