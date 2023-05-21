#!/usr/bin/python3
import numpy as np
import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale, StandardScaler, MinMaxScaler
from ast import literal_eval
import matplotlib.pyplot as plt
import pickle

'''
this file is fitting randomforests with varying depth to the data and fitting the loss curves 
'''

# what trainingset do we want to use 
training_set_one = True

if training_set_one:
    scaler_target   = pickle.load(open("scaler_target_1", 'rb'))
else: 
    scaler_target   = pickle.load(open("scaler_target_2", 'rb'))

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


# create one random forest for each variable 
loss_values = {"train_temp":[], "train_wind":[], "train_precip":[], "val_temp":[], "val_wind":[], "val_precip":[]}
minrmse_t = 100
minrmse_w = 100
minrmse_p = 100
bestmodel_t = 0
bestmodel_w = 0
bestmodel_p = 0
for depth in range(1,25):
    rfr_temp = RandomForestRegressor(max_depth= depth, random_state= 0)
    rfr_wind = RandomForestRegressor(max_depth= depth, random_state= 0)
    rfr_precip = RandomForestRegressor(max_depth= depth, random_state= 0)
    rfr_temp.fit(x_train, y_train[:, 0])
    rfr_wind.fit(x_train, y_train[:, 1])
    rfr_precip.fit(x_train, y_train[:, 2])
    if np.sqrt(np.mean((rfr_temp.predict(x_val)-y_val[:, 0])**2)) < minrmse_t:
        bestmodel_t = rfr_temp
        minrmse_t = np.sqrt(np.mean((rfr_temp.predict(x_val)-y_val[:, 0])**2))
        print("Temp:", depth)
    if np.sqrt(np.mean((rfr_wind.predict(x_val)-y_val[:, 1])**2)) < minrmse_w:
        bestmodel_w = rfr_wind
        minrmse_w = np.sqrt(np.mean((rfr_wind.predict(x_val)-y_val[:, 1])**2))
        print("Wind:", depth)
    if np.sqrt(np.mean((rfr_precip.predict(x_val)-y_val[:, 2])**2)) < minrmse_p:
        bestmodel_p = rfr_precip
        minrmse_p = np.sqrt(np.mean((rfr_precip.predict(x_val)-y_val[:, 2])**2))
        print("Precip:", depth)
    loss_values["train_temp"].append((np.sqrt(np.mean((rfr_temp.predict(x_train)-y_train[:, 0])**2))))
    loss_values["val_temp"].append((np.sqrt(np.mean((rfr_temp.predict(x_val)-y_val[:, 0])**2))))
    loss_values["train_wind"].append((np.sqrt(np.mean((rfr_wind.predict(x_train)-y_train[:, 1])**2))))
    loss_values["val_wind"].append((np.sqrt(np.mean((rfr_wind.predict(x_val)-y_val[:, 1])**2))))
    loss_values["train_precip"].append((np.sqrt(np.mean((rfr_precip.predict(x_train)-y_train[:, 2])**2))))
    loss_values["val_precip"].append((np.sqrt(np.mean((rfr_precip.predict(x_val)-y_val[:, 2])**2))))

if training_set_one == True:
    pickle.dump(bestmodel_t, open("randomforest/forest_temp_1", "wb"))
    pickle.dump(bestmodel_w, open("randomforest/forest_wind_1", "wb"))
    pickle.dump(bestmodel_p, open("randomforest/forest_prec_1", "wb"))
else:
    pickle.dump(bestmodel_t, open("randomforest/forest_temp_2", "wb"))
    pickle.dump(bestmodel_w, open("randomforest/forest_wind_2", "wb"))
    pickle.dump(bestmodel_p, open("randomforest/forest_prec_2", "wb"))



plt.figure()
plt.plot(loss_values["train_temp"], label='training')
plt.plot(loss_values["val_temp"], label='validation')
plt.legend()
plt.xlabel("Depth")
plt.ylabel("MSE")
if training_set_one == True:
    plt.title("Temperature, training set one")
    plt.savefig('randomforest/tempforest_1.png')
else:
    plt.title("Temperature, training set two")
    plt.savefig('randomforest/tempforest_2.png')

plt.figure()
plt.plot(loss_values["train_wind"], label='training')
plt.plot(loss_values["val_wind"], label='validation')
plt.legend()
plt.ylabel("MSE")
plt.xlabel("Depth")
if training_set_one == True:
    plt.title("Wind, training set one")
    plt.savefig('randomforest/windforest_1.png')
else:
    plt.title("Wind, training set two")
    plt.savefig('randomforest/windforest_2.png')

plt.figure()
plt.plot(loss_values["train_precip"], label='training')
plt.plot(loss_values["val_precip"], label='validation')
plt.legend()
plt.ylabel("MSE")
plt.xlabel("Depth")
if training_set_one == True:
    plt.title("Precipitation, training set one")
    plt.savefig('randomforest/precforest_1.png')
else:
    plt.title("Precipitation, training set two")
    plt.savefig('randomforest/precforest_2.png')


print("For Temperature: ", np.sqrt(np.mean((bestmodel_t.predict(x_test) - y_test[:,0]) ** 2)))
print("For Windspeed: ", np.sqrt(np.mean((bestmodel_w.predict(x_test) - y_test[:,1]) ** 2)))
print("For Precipitation: ", np.sqrt(np.mean((bestmodel_p.predict(x_test) - y_test[:,2]) ** 2)))
