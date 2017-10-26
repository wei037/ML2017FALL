import numpy as np
import pandas as pd
import sys
import time

def logistic_regression(X,Y,iteration = 200) :
    start_time = time.time()

    f_num = len(X[0])
    b = 0.0 # initial b
    w = np.zeros(f_num) # initial w
    
    lr = 0.1 # learning rate
    b_lr = 0.0
    w_lr = np.zeros(f_num)
    
    X = scale(X)
    ld = 0.0
    for i in range(iteration):
        loss =  Y - sigmoid(b+np.dot(X,w))
        
        b_grad = -(np.sum(loss))
        w_grad = -(np.dot(X.T,loss) + ld*w) 
         
        b_lr = b_lr + b_grad**2
        w_lr = w_lr + w_grad**2
                       
        b = b - lr/np.sqrt(b_lr) * b_grad                        
        w = w - lr/np.sqrt(w_lr) * w_grad

        if (i+1)%500 == 0 :
            C_entropy = -(Y*np.log(sigmoid(b+np.dot(X,w)) + 1e-10) + (1-Y)*np.log(1-sigmoid(b+np.dot(X,w)) + 1e-10 ))
            print ('Loss = ',np.mean(C_entropy) , ' Accu = ' , valid(b,w,X,Y))
    
    print ('Elapsed : ',time.time() - start_time,'second')
    return b,w

def sigmoid(z) :
    return 1 / (1 + np.exp(-z))

def scale(X):
    return (X - np.mean(X , axis = 0))/(np.std(X , axis = 0) + 1e-10)

def predict(b,w,X):
    pred = sigmoid(b+np.dot(X,w))
    pred[pred < 0.5] = 0.0
    pred[pred >= 0.5] = 1.0
    return pred

def valid(b,w,X,Y):
    pred = predict(b,w,X)
    return (np.mean(1-np.abs(pred - Y)))

def output(pred_list,text) :    
    out = [['id','label']]
    for i in range(len(pred_list)) :
        tmp = [str(i+1) , int(pred_list[i])]
        out.append(tmp)
    
    pd.DataFrame(out).to_csv(text , header = False , index = False) 

qua = [0,1,3,5]   #age,fnlwgt,capital_gain,capital_loss,hours_per_week
tri = [0,3]       #2 for sex
workclass          =  np.arange(9)  + 6 
education          =  np.arange(16) + 15
marital_status     =  np.arange(7)  + 31
occupation         =  np.arange(15) + 38
relationship       =  np.arange(6)  + 53
race               =  np.arange(5)  + 59
native_country     =  np.arange(42) + 64
features = np.concatenate([[0,1,2,3,4,5] , workclass , education , occupation , relationship ] )

train_X = pd.read_csv(sys.argv[1]).as_matrix().astype('float')
train_Y = pd.read_csv(sys.argv[2]).as_matrix().astype('float').reshape(-1,)
test_X = pd.read_csv(sys.argv[3]).as_matrix().astype('float')
train_X = train_X[:,features]
test_X = test_X[:,features]
train_X = np.concatenate([train_X , train_X[:,qua]**(2) ,  train_X[:,tri]**(3)] , axis = 1)
test_X = np.concatenate([test_X , test_X[:,qua]**(2) ,  test_X[:,tri]**(3)] , axis = 1)
#b, w = logistic_regression(train_X,train_Y,50000)
b = -2.29643979822 
w = [4.581099272458971, 0.11291178656426686, 0.38743201169782987, 2.4238562340501404, 0.2596689216802788, 0.9141675569283402, 0.08621544209318731, -0.03235173348398744, -0.12158353678249628, 0.05203434707146411, 0.058126829673021005, -0.0850084655658994, -0.04715659490369769, -0.19684121894346826, -0.035717858490454495, -0.20718468959618827, -0.21760275764277637, -0.09680013408434522, -0.14400950785410715, -0.1793835655029549, -0.24077732350269537, -0.18872634467978278, -0.0004723097936832124, 0.0014861797701884144, 0.22409381575557724, 0.18448893040978323, -0.23071597187701678, 0.19958810386047646, -0.5551354962486406, 0.19540646072098722, -0.06141985560272927, -0.048535900944758785, -0.020260541209327458, -0.0405004113426845, 0.21077311896373008, -0.17082285388290083, -0.1662186906730598, -0.12315712795916577, -0.2945366829800433, -0.2457512796940215, 0.12471512993931072, 0.0735238157894263, 0.05296378236769395, 0.08398844693579499, -0.05641162981408907, -0.03685026102693441, 0.4424547589646427, -0.38643519068215315, -0.16353892465135234, -0.6171346828329838, -0.31900210250056477, 0.48594935876254947, -5.381936720046831, -0.05735325424762903, -0.14619468043738326, -0.5501952450937081, 1.4309250900013288, -1.5298525294334167]

test_X = scale(test_X)
output(predict(b,w,test_X),sys.argv[4])



