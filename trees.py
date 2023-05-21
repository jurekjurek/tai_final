#!/usr/bin/python3
import numpy as np
import pandas as pd 
from sklearn import tree
from sklearn.model_selection import train_test_split
from ast import literal_eval
from sklearn.preprocessing import scale, StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import pickle


'''
this file is training trees of varying depth on the data and plotting the losscurves 
'''

# what trainingset do we want to use 
training_set_one = True


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

rmse = {"train_temp":[], "train_wind":[], "train_precip":[], "val_temp":[], "val_wind":[], "val_precip":[]}
minrmse_t = 100
minrmse_w = 100
minrmse_p = 100
bestmodel_t = 0
bestmodel_w = 0
bestmodel_p = 0
for i in range(1,15):
    temperatureTree = tree.DecisionTreeRegressor(max_depth=i)
    windTree = tree.DecisionTreeRegressor(max_depth=i)
    precipTree = tree.DecisionTreeRegressor(max_depth=i)
    temperatureTree.fit(x_train, y_train[:, 0])
    windTree.fit(x_train, y_train[:, 1])
    precipTree.fit(x_train, y_train[:, 2])
    if np.sqrt(np.mean((temperatureTree.predict(x_val)-y_val[:, 0])**2)) < minrmse_t:
        bestmodel_t = temperatureTree
        minrmse_t = np.sqrt(np.mean((temperatureTree.predict(x_val)-y_val[:, 0])**2))
        print("Temperature, depth: ", i)
    if np.sqrt(np.mean((windTree.predict(x_val)-y_val[:, 1])**2)) < minrmse_w:
        bestmodel_w = windTree
        minrmse_w = np.sqrt(np.mean((windTree.predict(x_val)-y_val[:,1])**2))
        print("Wind, depth: ", i)
    if np.sqrt(np.mean((precipTree.predict(x_val)-y_val[:, 2])**2)) < minrmse_p:
        bestmodel_p = precipTree
        minrmse_p = np.sqrt(np.mean((precipTree.predict(x_val)-y_val[:, 2])**2))
        print("Precipitation, depth: ", i)
    rmse["train_temp"].append((np.sqrt(np.mean((temperatureTree.predict(x_train)-y_train[:, 0])**2))))
    rmse["val_temp"].append((np.sqrt(np.mean((temperatureTree.predict(x_val)-y_val[:, 0])**2))))
    rmse["train_wind"].append((np.sqrt(np.mean((windTree.predict(x_train)-y_train[:, 1])**2))))
    rmse["val_wind"].append((np.sqrt(np.mean((windTree.predict(x_val)-y_val[:, 1])**2))))
    rmse["train_precip"].append((np.sqrt(np.mean((precipTree.predict(x_train)-y_train[:, 2])**2))))
    rmse["val_precip"].append((np.sqrt(np.mean((precipTree.predict(x_val)-y_val[:, 2])**2))))
    # best trees
    # temp: 11
    # wind: 13
    # precip: 7
if training_set_one == True:
    pickle.dump(bestmodel_t, open("trees/tree_temp_1", "wb"))
    pickle.dump(bestmodel_w, open("trees/tree_wind_1", "wb"))
    pickle.dump(bestmodel_p, open("trees/tree_prec_1", "wb"))
else:
    pickle.dump(bestmodel_t, open("trees/tree_temp_2", "wb"))
    pickle.dump(bestmodel_w, open("trees/tree_wind_2", "wb"))
    pickle.dump(bestmodel_p, open("trees/tree_prec_2", "wb"))


plt.figure()
plt.plot(rmse["train_temp"], label='training')
plt.plot(rmse["val_temp"], label='validation')
plt.legend()
plt.xlabel("Depth")
plt.ylabel("RMSE")
if training_set_one == True:
    plt.title("Temperature, training set one")
    plt.savefig('trees/temptree_1.png')
else:
    plt.title("Temperature, training set two")
    plt.savefig('trees/temptree_2.png')
    
plt.figure()
plt.plot(rmse["train_wind"], label='training')
plt.plot(rmse["val_wind"], label='validation')
plt.legend()
plt.ylabel("RMSE")
plt.xlabel("Depth")
if training_set_one == True:
    plt.title("Wind, training set one")
    plt.savefig('trees/windtree_1.png')
else:
    plt.title("Wind, training set two")
    plt.savefig('trees/windtree_2.png')

plt.figure()
plt.plot(rmse["train_precip"], label='training')
plt.plot(rmse["val_precip"], label='validation')
plt.legend()
plt.ylabel("RMSE")
plt.xlabel("Depth")
if training_set_one == True:
    plt.title("Precipitation, training set one")
    plt.savefig('trees/prectree_1.png')
else:
    plt.title("Precipitation, training set two")
    plt.savefig('trees/prectree_2.png')

# print("The final loss value is: ", np.mean((clf.predict(x_test) - y_test) ** 2))
print("For Temperature: ", np.sqrt(np.mean((bestmodel_t.predict(x_test) - y_test[:,0]) ** 2)))
print("For Windspeed: ", np.sqrt(np.mean((bestmodel_w.predict(x_test) - y_test[:,1]) ** 2)))
print("For Precipitation: ", np.sqrt(np.mean((bestmodel_p.predict(x_test) - y_test[:,2]) ** 2)))




