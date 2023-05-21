#!/usr/bin/python3
import numpy as np
import pandas as pd 
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from ast import literal_eval
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import scale, StandardScaler, MinMaxScaler

'''
this file trains the bayesian ridge model
'''

# what trainingset do we want to use 
training_set_one = False

if training_set_one:
    scaler_target   = pickle.load(open("scaler_target_1", 'rb'))
else: 
    scaler_target   = pickle.load(open("scaler_target_2", 'rb'))

if training_set_one == True:
    x_train = np.loadtxt('x_train_1')
    y_train_unscaled = np.loadtxt('y_train_1')
    y_train = scaler_target.transform( np.loadtxt('y_train_1'))
    x_val   = np.loadtxt('x_val_1')
    y_val   = np.loadtxt('y_val_1')
    x_test  = np.loadtxt('x_test_1')
    y_test  = np.loadtxt('y_test_1')
else: 
    x_train = np.loadtxt('x_train_2')
    y_train_unscaled = np.loadtxt('y_train_2')
    y_train = scaler_target.transform(np.loadtxt('y_train_2'))
    x_val   = np.loadtxt('x_val_2')
    y_val   = np.loadtxt('y_val_2')
    x_test  = np.loadtxt('x_test_2')
    y_test  = np.loadtxt('y_test_2')




rmse = {"train_temp":[], "train_wind":[], "train_precip":[], "val_temp":[], "val_wind":[], "val_precip":[]}

alpha_list = [0.001, 0.01, 0.1, 1.0, 10.0]

for i in alpha_list:
    # if i%10 == 0:  
        BayReg_temp = linear_model.BayesianRidge(alpha_1 = i)
        BayReg_wind = linear_model.BayesianRidge(alpha_1 = i)
        BayReg_precip = linear_model.BayesianRidge(alpha_1 = i)

        BayReg_temp.fit(x_train, y_train[:, 0])
        BayReg_wind.fit(x_train, y_train[:, 1])
        BayReg_precip.fit(x_train, y_train[:, 2])


        y_pred_train = scaler_target.inverse_transform( np.transpose( np.vstack ((BayReg_temp.predict(x_train), BayReg_wind.predict(x_train), BayReg_precip.predict(x_train)) ) ))
        y_pred_val = scaler_target.inverse_transform( np.transpose(  np.vstack ((BayReg_temp.predict(x_val), BayReg_wind.predict(x_val), BayReg_precip.predict(x_val)) ) ))


        rmse["train_temp"].append((np.mean((y_pred_train[:,0]-y_train[:, 0])**2)))
        rmse["val_temp"].append((np.mean((y_pred_val[:,0]-y_val[:, 0])**2)))
        rmse["train_wind"].append((np.mean((y_pred_train[:,1]-y_train[:, 1])**2)))
        rmse["val_wind"].append((np.mean((y_pred_val[:,1]-y_val[:, 1])**2)))
        rmse["train_precip"].append((np.mean((y_pred_train[:,2]-y_train[:, 2])**2)))
        rmse["val_precip"].append((np.mean((y_pred_val[:,2]-y_val[:, 2])**2)))


if training_set_one == True:
    pickle.dump(BayReg_temp, open("bayesian/temp_1", "wb"))
    pickle.dump(BayReg_wind, open("bayesian/wind_1", "wb"))
    pickle.dump(BayReg_precip, open("bayesian/prec_1", "wb"))
else: 
    pickle.dump(BayReg_temp, open("bayesian/temp_2", "wb"))
    pickle.dump(BayReg_wind, open("bayesian/wind_2", "wb"))
    pickle.dump(BayReg_precip, open("bayesian/prec_2", "wb"))


plt.figure()
plt.plot(rmse["train_temp"], label='training')
plt.plot(rmse["val_temp"], label='validation')
plt.legend()
plt.title("Temperature")
plt.xlabel("Depth")
plt.ylabel("MSE")
if training_set_one == True:
    plt.savefig('bayesian/bay_temp_1.png')
else: 
    plt.savefig("bayesian/bay_temp_2.png")

plt.figure()
plt.plot(rmse["train_wind"], label='training')
plt.plot(rmse["val_wind"], label='validation')
plt.legend()
plt.title("Wind speed")
plt.ylabel("MSE")
plt.xlabel("Depth")
if training_set_one == True:
    plt.savefig('bayesian/bay_wind_1.png')
else: 
    plt.savefig("bayesian/bay_wind_2.png")

plt.figure()
plt.plot(rmse["train_precip"], label='training')
plt.plot(rmse["val_precip"], label='validation')
plt.legend()
plt.title("Precipitation")
plt.ylabel("MSE")
plt.xlabel("Depth")
if training_set_one == True:
    plt.savefig('bayesian/bay_prec_1.png')
else: 
    plt.savefig("bayesian/bay_prec_2.png")



result = scaler_target.inverse_transform( np.transpose( np.vstack ((BayReg_temp.predict(x_test), BayReg_wind.predict(x_test), BayReg_precip.predict(x_test)) )))

print("For Temperature: ", np.sqrt(np.mean((result[:,0] - y_test[:,0]) ** 2)))
print("For Windspeed: ", np.sqrt(np.mean((result[:,1] - y_test[:,1]) ** 2)))
print("For Precipitation: ", np.sqrt(np.mean((result[:,2] - y_test[:,2]) ** 2)))
