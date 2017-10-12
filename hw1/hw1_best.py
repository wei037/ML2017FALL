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
#b,w = linear_regression(train_X,train_Y,300000)

b = -0.635614996799
w = [0.030635458407937084, -0.008200179507507736, -0.002760506077351969, 0.026602759440893187, 0.018975847807704196, -0.2462881904136309, -0.001278491512632653, -0.0020294923775047167, -0.07615990815980762, -0.05520019780504333, 6.137052696100863e-05, -0.000846438542755817, -0.07222137506005613, 0.017455532086341735, -0.0029908594030940952, -0.0038049815029286096, -0.01155961280836185, 0.2079112569431778, 0.0030755506497631845, -0.0007849235689046723, -0.00744512874662323, 0.06941418645224048, 3.714089883942277e-05, 0.00019744624203300813, -0.01981556195297696, -0.007440018973054335, -0.0036372752662358955, 0.05190433207666753, -0.05991478813023103, -0.007308503347912939, -0.0006245728837620516, 0.0008534164486210785, 0.16885462427051004, -0.02711112897148147, -2.387275935568688e-05, 0.0014931578352759152, -0.30473481905122246, -0.006915618054288658, 0.021676251844881356, -0.10622796851429359, -0.008620709904204505, -0.09600590916235145, 0.0021695507687923086, -0.0020848641432759684, 0.048738623440220466, -0.10178458079908063, 2.6370418442045672e-05, -0.001270263621294023, 0.28835098726544506, -0.0009645847491074178, 0.01634421123408834, 0.029486032103850646, -0.04916708352834255, -0.0015865662460308825, 0.0006559513122704994, 0.0006945647873383344, -0.027894622354077657, -0.11014685890146184, -0.0004001018368718498, -0.000246250478086909, -0.0728236332657258, -0.017049231480150175, -0.012770802792902429, 0.21661421190736804, 0.032176924910259294, 0.02733760332446342, 0.0015225158130865064, -3.020957600572e-05, 0.04530938680851459, 0.18458714548803593, 7.327373388006914e-05, 0.0028332795824410483, -0.17806583265386308, -0.0124406937770799, 0.0049725839849593945, -0.3088831344584508, 0.009481396284350395, -0.17362624565915602, -0.0023901165553858444, 0.00029091002614251986, 0.018439182685257474, -0.004372032417938337, 0.0001627703811663039, -0.003047187063500416, 0.567675442891866, -0.009414638874989967, 0.025172739806750673, 0.05638139842417937, -0.03418269512467868, 0.05006628723187681, 0.0018712651093737462, -0.0012733521824479194, -0.12278807535346724, -0.2831663296043042, -0.0004637028607204313, -0.00019482344509047885, 1.6173536622977394, 0.07172788143844208, 0.04177996160622597, 0.8089683928360266, -0.0834046036572372, 0.4394449885928648, 0.0012557306471826808, 0.0013598961284275958, -0.03193594700050219, -0.15260165942559348, 0.00027839860399230193, 0.0018489598901723797]

output(pred(b,w,test_X),sys.argv[2])


