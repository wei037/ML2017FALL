import numpy as np
import pandas as pd
import sys
import load_data

from keras.models import load_model
from keras.utils.vis_utils import plot_model


def output(pred_list,text) :    
    out = [['id','label']]

    for i in range(len(pred_list)) :
        tmp = [str(i) , int(pred_list[i])]
        out.append(tmp)
    
    pd.DataFrame(out).to_csv(text , header = False , index = False) 


def predict(text , data) :    
    model = load_model(text)
    model_aug = load_model('64753.h5')

    pred_list = model.predict(data)
    pred_list = pred_list + model_aug.predict(data)
    pred = np.argmax(pred_list , axis = 1) 

    return pred

def main() :
    X_test = load_data.load(sys.argv[1])
    X_test = X_test.reshape(-1, 48, 48, 1)
    X_test = X_test / 255

    pred = predict('65700.h5' , X_test)
    output(pred.astype('int') , sys.argv[2])

if __name__ == "__main__":
    main()
