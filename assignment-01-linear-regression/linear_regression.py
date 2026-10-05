import sklearn
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import model_selection
from sklearn import linear_model
from sklearn import metrics
import numpy as np

class CustomError(Exception):
    pass


class LinearRegression:
    def __init__(self):
        self.w=None
        self.b=None



    def fit(self,X,y):
        if not isinstance(X, np.ndarray) and not isinstance (y,np.ndarray):
            raise ValueError("Not array objects")

        if X.shape[0]!=y.shape[0]:
             raise ValueError("Size not match!")

        ones = np.ones((X.shape[0],1)) #create vector p*1
        X = np.hstack((X,ones)) #create array Ν*(p+1) #concatenate the 2 arrays horrizontally

        X_T=np.transpose(X) #(p+1)*N
        XTX=np.dot(X_T,X) #(p+1)*(p+1)
        XTX_inv=np.linalg.inv(XTX) #(p+1)*(p+1)
        theta_a=np.dot(X_T,y) #(p+1)*1
        theta=np.dot(XTX_inv,theta_a) #(p+1)*1

        self.w=theta[:-1] #p*1
        print(f'w:{self.w}')
        self.b=theta[-1]
        print(f'b:{self.b}')

    def predict(self,X):
        #yˆ = Xw + b
        return np.dot(X,self.w)+self.b

    def evaluate(self, X, y):
        if self.w is None and self.b is None:
            raise CustomError('Not trained')
        y_hat= self.predict(X)
        #MSE =1/N(ˆy − y)^T(ˆy − y)
        diff=y_hat-y
        MSE=np.mean(diff**2)
        return y_hat,MSE

