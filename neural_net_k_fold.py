from cProfile import label
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import random
from sklearn.preprocessing import scale, StandardScaler, MinMaxScaler
# linear regression
from sklearn.linear_model import LinearRegression
# neural net
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import pickle 
from sklearn import tree


'''
this file is for demonstrating the method of kfold cross validation 
It is demonstrated for decision trees. 
The root mean squared error for the different variables is plotted in a bar plot. this is to see if the errors for the different folds vary a lot, which, as we will find out, they do not. 
'''


# what trainingset do we want to use 
training_set_one = False


if training_set_one == True:
    x_train = np.loadtxt('x_train_1')
    y_train = np.loadtxt('y_train_1')
    x_val   = np.loadtxt('x_val_1')
    y_val   = np.loadtxt('y_val_1')
    x_test  = np.loadtxt('x_test_1')
    y_test  = np.loadtxt('y_test_1')
else: 
    x_train = np.loadtxt('x_train_2')
    y_train = np.loadtxt('y_train_2')
    x_val   = np.loadtxt('x_val_2')
    y_val   = np.loadtxt('y_val_2')
    x_test  = np.loadtxt('x_test_2')
    y_test  = np.loadtxt('y_test_2')


# set to the perfect depths we found earlier 
# very close for both datasets, so we leave it like this 
temperatureTree = tree.DecisionTreeRegressor(max_depth=12)
windTree = tree.DecisionTreeRegressor(max_depth=12)
precipTree = tree.DecisionTreeRegressor(max_depth=8)


rmse = {"test_temp":[], "test_wind":[], "test_precip":[]}

nn_reg = MLPRegressor(solver='adam', alpha= 1e-5, hidden_layer_sizes=(7,6), random_state=0)
scores=[]
kFold=KFold(n_splits=5,random_state=0,shuffle=True)
for train_index, val_index in kFold.split(x_train):
    print("Train Index: ", train_index, "\n")
    print("Validation Index: ", val_index)
    
    x_train_k, x_val_k, y_train_k, y_val_k = x_train[train_index], x_train[val_index], y_train[train_index], y_train[val_index]

    temperatureTree.fit(x_train, y_train[:, 0])
    windTree.fit(x_train, y_train[:, 1])
    precipTree.fit(x_train, y_train[:, 2])

    rmse["test_temp"].append((np.sqrt(np.mean((temperatureTree.predict(x_test)-y_test[:, 0])**2))))
    rmse["test_wind"].append((np.sqrt(np.mean((windTree.predict(x_test)-y_test[:, 1])**2))))
    rmse["test_precip"].append((np.sqrt(np.mean((precipTree.predict(x_test)-y_test[:, 2])**2))))
    

plt.figure()
plt.bar(["1", "2", "3", "4", "5"],rmse["test_temp"])
plt.xlabel("K")
plt.ylabel("RMSE")
if training_set_one == True:
    plt.title("Temperature, training set one")
    plt.savefig('trees/kfold_temp_1.png')
else: 
    plt.title("Temperature, training set two")
    plt.savefig("trees/kfold_temp_2.png")


plt.figure()
plt.bar(["1", "2", "3", "4", "5"],rmse["test_wind"])
plt.xlabel("K")
plt.ylabel("RMSE")
if training_set_one == True:
    plt.title("Wind, training set one")
    plt.savefig('trees/kfold_wind_1.png')
else: 
    plt.title("Wind, training set two")
    plt.savefig("trees/kfold_wind_2.png")


plt.figure()
plt.bar(["1", "2", "3", "4", "5"],rmse["test_precip"])
plt.xlabel("K")
plt.ylabel("RMSE")
if training_set_one == True:
    plt.title("Precipitation, training set one")
    plt.savefig('trees/kfold_prec_1.png')
else: 
    plt.title("Precipitation, training set two")
    plt.savefig("trees/kfold_prec_2.png")

