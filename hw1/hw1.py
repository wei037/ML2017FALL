import sys
import pandas as pd
import numpy as np
import time

def load_train(text):
    train_data = pd.read_csv(text , encoding = 'big5').as_matrix()
    train_data = train_data[:,3:]
    datalist = []
    for i in range(0,train_data.shape[0],18):
        tmp = train_data[i:i+18,:]
        datalist.append(tmp)
    
    data1 =  np.concatenate(datalist,axis=1).T
    data1[data1 == 'NR'] = 0
    data1 = data1.astype('float')
    attr = ['AMB_TEMP','CH4','CO','NMHC','NO','NO2','NOx','O3','PM10','PM2.5',
     'RAINFALL','RH','SO2','HC','WD_HR','WIND_DIREC','WIND_SPEED','WS_HR','qua_PM10','qua_PM2.5']
    qua_PM10 = data1[:,8:9]**2
    qua_PM25 = data1[:,9:10]**2
    data1 = np.concatenate((data1,qua_PM10,qua_PM25),axis=1)
    data1 = pd.DataFrame(data1)
    data1.columns = attr
    return data1

def load_test(text):
    test_data = pd.read_csv(text , header = None).as_matrix()
    test_data = test_data[:,2:]
    datalist = []
    for i in range(0,test_data.shape[0],18):
        tmp = test_data[i:i+18,:]
        datalist.append(tmp)
    
    data1 =  np.concatenate(datalist,axis=1).T
    data1[data1 == 'NR'] = 0
    data1 = data1.astype('float')
    attr = ['AMB_TEMP','CH4','CO','NMHC','NO','NO2','NOx','O3','PM10','PM2.5',
     'RAINFALL','RH','SO2','HC','WD_HR','WIND_DIREC','WIND_SPEED','WS_HR','qua_PM10','qua_PM2.5']
    qua_PM10 = data1[:,8:9]**2
    qua_PM25 = data1[:,9:10]**2
    data1 = np.concatenate((data1,qua_PM10,qua_PM25),axis=1)
    data1 = pd.DataFrame(data1)
    data1.columns = attr
    return data1

def feature_select(train_df,test_df,f_list) :
    train_df = train_df[f_list]
    target = train_df['PM2.5']
    data = np.array(train_df)
    x = []
    y = []
    for i in range(12):
        for j in range(20*24-9):
            x.append(data[i*480+j : i*480+j+9,:].flatten())
            y.append(target.iloc[i*480+j+9])
    test_df = test_df[f_list]
    data = np.array(test_df)
    test_x = []
    for i in range(0,data.shape[0],9) :
        test_x.append(data[i:i+9,:].flatten())
            
    return np.array(x),np.array(y),np.array(test_x)

def linear_regression(X,Y,iteration = 20000) :
    start_time = time.time()

    f_num = len(X[0])
    # ydata = b + w * xdata 
    b = 0.0 # initial b
    w = np.ones(f_num) # initial w
    
    lr = 1.0 # learning rate
    b_lr = 0.0
    w_lr = np.zeros(f_num)

    for i in range(iteration):
        error = Y - b - np.dot(X,w)
        b_grad = -2.0*(np.sum(error))*1.0 
        w_grad = -2.0*(np.dot(X.T,error)) 
        
        b_lr = b_lr + b_grad**2
        w_lr = w_lr + w_grad**2
                       
        b = b - lr/np.sqrt(b_lr) * b_grad                        
        w = w - lr/np.sqrt(w_lr) * w_grad

        if (i+1)%1000 == 0 :
            error = Y - b - np.dot(X,w)
            print ('Error = ',np.sqrt(np.mean(error**2)))
    
    print ('Elapsed : ',time.time() - start_time,'second')
    return b,w

def pred(b,w,x) :
    return np.dot(x,w) + b

def output(pred_list,text) :    
    out = [['id','value']]
    for i in range(240) :
        tmp = ["id_"+str(i) , pred_list[i]]
        out.append(tmp)
    
    pd.DataFrame(out).to_csv(text , header = False , index = False) 

train_df = load_train('train.csv')
test_df = load_test(sys.argv[1])
f_list = ['CO','O3','PM10','PM2.5',
     'RAINFALL','SO2','WD_HR','WIND_DIREC','WIND_SPEED','WS_HR','qua_PM10','qua_PM2.5']
train_X , train_Y , test_X = feature_select(train_df,test_df,f_list) 
b,w = linear_regression(train_X,train_Y,300000)

output(pred(b,w,test_X),sys.argv[2])


