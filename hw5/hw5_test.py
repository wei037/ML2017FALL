import numpy as np
from keras.models import Sequential , load_model
from DataReader import *
import keras.backend as K
import sys

def output(pred_list,text) :    
    out = [['TestDataID','Rating']]

    for i in range(len(pred_list)) :
        tmp = [str(i+1) , pred_list[i,0]]
        out.append(tmp)
    
    pd.DataFrame(out).to_csv(text , header = False , index = False) 


def predict(text , data) :    
    model = load_model(text , custom_objects={'rmse': rmse})
    pred_list = model.predict(data)

    return pred_list

def rmse(y_true, y_pred): 
    return K.sqrt( K.mean((y_pred - y_true)**2) )

def main() :    
    Reader = DataReader()
    X_test = Reader.read_test(sys.argv[1])

    pred = predict('85535.h5' , X_test)
    std = 1.11689766115 
    mean = 3.58171208604 
    pred = pred*std + mean
    print (max(pred))
    print (min(pred))
    pred = np.clip(pred,1.0,5.0)
    print ('Predict done...')    
    output(pred , sys.argv[2])

if __name__ == "__main__":
    main()
