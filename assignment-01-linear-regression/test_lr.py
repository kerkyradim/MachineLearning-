import sklearn
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import model_selection
from sklearn import linear_model
from sklearn import metrics
import numpy as np
import linear_regression as LR
import random
from math import sqrt


dataset=sklearn.datasets.fetch_california_housing(data_home=None, download_if_missing=True, return_X_y=False, as_frame=False)


print(type(dataset.data)) #print type of the dataset
print(f'Target is {dataset.target}')

print(F'shape:{dataset.data.shape}')#print shape of the dataset


#checking my custom error(if not trained)
#lr=LR.LinearRegression()
#lr.evaluate([],[])

#split data to train-set and test-set
X_train, X_test, y_train, y_test = train_test_split( dataset.data, dataset.target, test_size=0.3, random_state=42)

lr=LR.LinearRegression()
lr.fit(X_train,y_train)

#predict
y_hat=lr.predict(X_test)
print(f'y_hat:{y_hat}\n')

#evaluate
y_hat,MSE=lr.evaluate(X_test,y_test)
r_MSE=sqrt(MSE)
#print(f'y_hat:{y_hat}')
print(f'MSE:{MSE}')
print(f'root MSE:{r_MSE}\n')

#run for 20 iterations
rMSE=[]
for i in range (20):
  X_train, X_test, y_train, y_test = train_test_split( dataset.data, dataset.target, test_size=0.3, random_state=random.randint(0, 2**32 - 1))
  y_hat=lr.predict(X_test)
  y_hat,MSE=lr.evaluate(X_test,y_test)
  r_MSE=sqrt(MSE)
  rMSE.append(r_MSE)

rMSEmean=np.mean(rMSE)
stdMSE=np.std(rMSE)
#print(f'rmse:{rMSE}')
print("With 20 iterations")
print(f'rmse mean:{rMSEmean}')
print(f'std rmse:{stdMSE}')
print("\n")

print("--------------- Using Linear Regression class from scikit-learn ---------------\n")

model =linear_model.LinearRegression()

model.fit(X_train,y_train)#fit

#predict
y_hat=model.predict(X_test)

MSE=np.mean((y_hat - y_test)**2)
r_MSE=sqrt(MSE)
print(f'MSE:{MSE}')
print(f'root MSE:{r_MSE}\n')

#run for 20 iterations
rMSE=[]
for i in range(20):
   X_train, X_test, y_train, y_test = train_test_split( dataset.data, dataset.target, test_size=0.3, random_state=random.randint(0, 2**32 - 1))
   #fit
   model.fit(X_train,y_train)
   #train
   y_hat= model.predict(X_test)
   MSE=np.mean((y_hat - y_test)**2) #mse sklearn
   r_MSE=sqrt(MSE)
   rMSE.append(r_MSE)

rMSEmean=np.mean(rMSE)
stdMSE=np.std(rMSE)
#print(f'root MSE:{rMSE}\n')
print("With 20 iterations")
print(f'rmse mean:{rMSEmean}')
print(f'std rmse:{stdMSE}')


#Παρατηρούμε ότι το μοτέλο μας έχει εκπαιδευτεί καλά και δεν έχουμε υποεκπαίδευση ή υπερεκπαίδευση.MSE~=0,5
#και rMSE~=0,72(η ποσότητα λάθους σε μονάδες 100,000)
#Αυτό σημαίνει ότι οι προβλέψεις μας και οι πραγματικές τιμές διαφέρουν ελάχιστα.
# Επίσης βλέπουμε ότι η τυπική απόκλιση του rMSE~=0,0088(για τις 20 επαναλήψεις) που σημαίνει ότι το μοντέλο μας μπορεί να προσαρμοστεί καλά στα δεδομένα μας και ότι είναι αξιόπιστο

#Όταν προσπαθούμε να προβλέψουμε τις τιμές χρησιμοποιώντας την κλάση Linear Regression του sickit-learn
#έχουμε rMse mean~=0,73 και std rmse=0,0094(για τις 20 επαναλήψεις) που σημαίνει πάλι κοντινές προβλεπομένες τιμές με πραγματικές και
#καλή προσαρμογή του μοντέλου στα δεδομένα μας

#Επίσης παρατηρούμε ότι το custom model έχει καλύτερες επιδόσεις και μικρότερο ποσοστό λάθους από το έτοιμο μοντέλο
# Linear Regression του sklearn