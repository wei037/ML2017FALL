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
train_X = np.concatenate((train_X[:2826,], train_X[3297:,]))
train_Y = np.concatenate((train_Y[:2826,], train_Y[3297:,]))
#b,w = linear_regression(train_X,train_Y,300000)

b = -0.640998380274
w = [0.23212480085646367, -0.005166890812423059, -0.0023511712449797617, 0.015812663404493596, 0.019240616598598794, -0.16494190844019224, -0.0019245036892864657, -0.001774886103409932, -0.10812725973742332, -0.1368653990122893, 3.2476908664638845e-05, -0.000563302418796304, -0.3266885033635034, 0.023687640809546686, -0.0033374391357325252, 0.019776615815488324, -0.013218534204111614, 0.13938147038808377, 0.0027403806910393, -0.000717956729702384, -0.018283580968849925, 0.10319090123248761, 6.965812613667992e-05, -0.0003640103884167955, 0.07571105166229478, -0.01343426062020999, -0.004019724893633926, 0.004040598526946951, -0.057404273541760156, 0.03752341030908132, 0.00014978287178830495, 0.0006599157582539886, 0.185561346058433, 0.010501258852476613, -2.6561649377133462e-05, 0.0021769861099877395, -0.4861697906751362, -0.006651543270487102, 0.015699607535255216, -0.09219005500563367, -0.0026567765650769215, -0.1629954951204742, 0.0022132594205200675, -0.0017854071567142303, 0.05009903349523773, -0.09164064978390969, 4.1638065126671556e-05, -0.0013760602207337404, 0.28368581338439486, 0.00023725552588158973, 0.017279262827163536, 0.0631661801826821, -0.043224018910400064, -0.009250972375368082, -0.00011210817913392679, 0.0008096208957109243, -0.006030175159842445, -0.13202010001192357, -0.0003602392188379375, -0.0008538620772872862, -0.052197079919998945, -0.019664943557401234, -0.004111079805190976, 0.20370663032610078, 0.02604333812956818, 0.06993720529007832, 0.0019449684659069025, -0.0005448468286998538, 0.051797634900799826, 0.1890380686054784, 9.108685582324193e-06, 0.003093363629958625, 0.027991176790527185, -0.006961717138342797, -0.007506335184899176, -0.3337820234493808, 0.019851852928865533, -0.20953314564364178, -0.001658964778593375, 0.0009198892277135132, 0.03185297034557725, 0.022324915088625007, 0.00027885521248651336, -0.0025509893044200517, 0.5836038752448605, -0.014786995676192927, 0.031027021578185706, 0.10782617623734746, -0.039161154028756945, -0.04544696104251007, 0.001395870920171871, -0.0013371187284868327, -0.09250485320803138, -0.31952716714850227, -0.0005257283049060813, -0.001763336620202881, 1.4410521358870398, 0.07254496454926704, 0.04919890205267133, 0.7675254299096629, -0.07008543084950566, 0.544179908491821, 0.0008279218241592327, 0.0011574231354290524, -0.058170422922710616, -0.17257385947536144, 0.0001727669560576928, 0.0033169615189839415]
output(pred(b,w,test_X),sys.argv[2])


