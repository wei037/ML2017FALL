import numpy as np
import pandas as pd
import time

def load(text):
    file = pd.read_csv(text).as_matrix()
    data = []
    for i in range(file.shape[0]) :
        data.extend(file[i,1].split())

    data = np.array(data).reshape(file.shape[0],2304).astype('float')
    
    if text == 'train.csv' :
        target = file[:,0]
        print ('train loaded done...')
        return  data , target

    print ('test loaded done...')
    return  data


def main():
    X_test = load('test.csv')
    X_train , Y_train = load('train.csv')

    pd.DataFrame(X_test.astype('int')).to_csv('X_test_1' , header = None, index = False) 
    pd.DataFrame(X_train.astype('int')).to_csv('X_train_1' , header = None, index = False) 
    pd.DataFrame(Y_train.astype('int')).to_csv('Y_train_1' , header = None, index = False) 
    
if __name__ == "__main__":
    main()