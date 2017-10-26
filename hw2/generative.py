import numpy as np
import pandas as pd
from numpy.linalg import inv
import sys
import time

class generative_model :
    def covariance(self,x1,x2):
        sigma1 = np.zeros((106,106))
        sigma2 = np.zeros((106,106))
        for i in range(int(self.N1)) :
            sigma1 += np.dot(np.transpose([x1[i] - self.u1]), [(x1[i] - self.u1)])
        for i in range(int(self.N2)) :
            sigma2 += np.dot(np.transpose([x2[i] - self.u2]), [(x2[i] - self.u2)])
        sigma1 /= self.N1
        sigma2 /= self.N2
        cov = ( self.N1*sigma1 + self.N2*sigma2 ) / (self.N1 + self.N2)
        return cov
    def mean(self,X):
        return np.mean(X , axis = 0)
    def fit(self,X,Y):
        X = self.scale(X)
        self.class_0, self.class_1 = np.where(Y == 0) , np.where(Y == 1)
        self.u1, self.u2 = self.mean(X[self.class_1]),self.mean(X[self.class_0])
        self.N1, self.N2 = float(len(self.class_1[0])) , float(len(self.class_0[0]))  
        self.cov = self.covariance(X[self.class_1] , X[self.class_0])
        pred = self.predict(X)
        print (np.mean(1-np.abs(pred - Y)))
               
    def predict(self,X):
        w = np.dot((self.u1-self.u2) , inv(self.cov))
        wx = np.dot(w , X.T)        
        b = -(1/2)*(np.dot(np.dot(self.u1,inv(self.cov)),self.u1))
        b += (1/2)*(np.dot(np.dot(self.u2,inv(self.cov)),self.u2))
        b += np.log(self.N1/self.N2)
        
        result = self.sigmoid(wx+b)
        result[result < 0.5] = 0
        result[result >= 0.5] = 1
        return result
               
    def scale(self,X):
        return (X - np.mean(X , axis = 0))/(np.std(X , axis = 0) + 1e-10)
    
    def sigmoid(self,z) :
        #z = w*x + b
        return 1 / (1 + np.exp(-z))
    
def output(pred_list,text) :    
    out = [['id','label']]
    for i in range(len(pred_list)) :
        tmp = [str(i+1) , int(pred_list[i])]
        out.append(tmp)
    
    pd.DataFrame(out).to_csv(text , header = False , index = False) 


def main():
    train_X = pd.read_csv(sys.argv[1]).as_matrix().astype('float')
    train_Y = pd.read_csv(sys.argv[2]).as_matrix().astype('float').reshape(-1,)
    test_X = pd.read_csv(sys.argv[3]).as_matrix().astype('float')
    model = generative_model()
    model.fit(train_X,train_Y)
    pred = model.predict(test_X)
    output(pred,sys.argv[4])
    
if __name__ == "__main__":
    main()

