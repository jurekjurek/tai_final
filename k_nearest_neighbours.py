import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.model_selection import train_test_split, cross_val_score
import random
from sklearn.preprocessing import scale, StandardScaler, MinMaxScaler
# k nearest neighbours
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import pickle

'''
We are trying out the machine learning with the k nearest neighbours method 
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

# remember: all x are scaled, all y are not scaled 

# number of neighbours 
no_nb = 10

loss_values = {'loss_train_temp': [],'loss_train_wind': [], 'loss_train_prec': [], 'loss_val_temp': [], 'loss_val_wind': [], 'loss_val_prec': []}


for i in range(1, no_nb):
    print("Neighbour: ", i)
    # train model for different number of neighbours 
    temp_model = KNeighborsRegressor(n_neighbors=i)
    wind_model = KNeighborsRegressor(n_neighbors=i)
    precip_model = KNeighborsRegressor(n_neighbors=i)
    temp_model.fit(x_train, y_train[:,0])
    wind_model.fit(x_train, y_train[:,1])
    precip_model.fit(x_train, y_train[:,2])
    y_pred_train = scaler_target.inverse_transform( np.transpose( np.vstack ((temp_model.predict(x_train), wind_model.predict(x_train), precip_model.predict(x_train)) ) ))
    y_pred_val = scaler_target.inverse_transform( np.transpose(  np.vstack ((temp_model.predict(x_val), wind_model.predict(x_val), precip_model.predict(x_val)) ) ))
    # y_pred_val_wind   = scaler_target.inverse_transform( wind_model.predict(x_val) )
    # y_pred_train_precip = scaler_target.inverse_transform( precip_model.predict(x_train) ) 
    # y_pred_val_precip   = scaler_target.inverse_transform( precip_model.predict(x_val) )


    rmse_train_temp = np.sqrt(np.mean((y_pred_train[:,0] - y_train_unscaled[:,0]) ** 2))
    rmse_train_wind = np.sqrt(np.mean((y_pred_train[:,1] - y_train_unscaled[:,1]) ** 2))
    rmse_train_prec = np.sqrt(np.mean((y_pred_train[:,2] - y_train_unscaled[:,2]) ** 2))

    rmse_val_temp   = np.sqrt(np.mean((y_pred_val[:,0] - y_val[:,0])**2))
    rmse_val_wind   = np.sqrt(np.mean((y_pred_val[:,1] - y_val[:,1])**2))
    rmse_val_prec   = np.sqrt(np.mean((y_pred_val[:,2] - y_val[:,2])**2))


    # rmse_train_temp = np.sqrt(np.mean((y_pred_train_temp - y_train_unscaled[:,0]) ** 2))
    # rmse_train_wind = np.sqrt(np.mean((y_pred_train_wind - y_train_unscaled[:,1]) ** 2))
    # rmse_train_prec = np.sqrt(np.mean((y_pred_train_precip - y_train_unscaled[:,2]) ** 2))

    # rmse_val_temp   = np.sqrt(np.mean((y_pred_val_temp - y_val[:,0])**2))
    # rmse_val_wind   = np.sqrt(np.mean((y_pred_val_wind - y_val[:,1])**2))
    # rmse_val_prec   = np.sqrt(np.mean((y_pred_val_precip - y_val[:,2])**2))

    loss_values["loss_train_temp"].append(rmse_train_temp)
    loss_values["loss_train_wind"].append(rmse_train_wind)
    loss_values["loss_train_prec"].append(rmse_train_prec)

    loss_values["loss_val_temp"].append(rmse_val_temp)
    loss_values["loss_val_wind"].append(rmse_val_wind)
    loss_values["loss_val_prec"].append(rmse_val_prec)

if training_set_one == True:
    pickle.dump(temp_model, open("knn/temp_1", "wb"))
    pickle.dump(wind_model, open("knn/wind_1", "wb"))
    pickle.dump(precip_model, open("knn/precip_1", "wb"))
else: 
    pickle.dump(temp_model, open("knn/temp_2", "wb"))
    pickle.dump(wind_model, open("knn/wind_2", "wb"))
    pickle.dump(precip_model, open("knn/precip_2", "wb"))

plt.figure()
plt.plot(loss_values["loss_train_temp"], label="training")
plt.plot(loss_values["loss_val_temp"], label="validation")
plt.legend()
plt.xlabel("Number of neighbours")
plt.ylabel("RMSE")
if training_set_one == True:
    plt.title("Temperature, training set one ")
    plt.savefig('knn/temp_1.png')
else:
    plt.title("Temperature, training set two")
    plt.savefig('knn/temp_2.png')

plt.figure()
plt.plot(loss_values["loss_train_wind"], label="training")
plt.plot(loss_values["loss_val_wind"], label="validation")
plt.legend()
plt.xlabel("Number of neighbours")
plt.ylabel("RMSE")
if training_set_one == True:
    plt.title("Windspeed, training set one")
    plt.savefig('knn/wind_1.png')
else:
    plt.title("Windspeed, training set two")
    plt.savefig('knn/wind_2.png')

plt.figure()
plt.plot(loss_values["loss_train_prec"], label="training")
plt.plot(loss_values["loss_val_prec"], label="validation")
plt.legend()
plt.xlabel("Number of neighbours")
plt.ylabel("RMSE")
if training_set_one == True:
    plt.title("Precipitation, trainig set one")
    plt.savefig('knn/prec_1.png')
else:
    plt.title("Precipitation, trainig set two")
    plt.savefig('knn/prec_2.png')

result = scaler_target.inverse_transform( np.transpose( np.vstack ((temp_model.predict(x_test), wind_model.predict(x_test), precip_model.predict(x_test)) )))

print("For Temperature: ", np.sqrt(np.mean((result[:,0] - y_test[:,0]) ** 2)))
print("For Windspeed: ", np.sqrt(np.mean((result[:,1] - y_test[:,1]) ** 2)))
print("For Precipitation: ", np.sqrt(np.mean((result[:,2] - y_test[:,2]) ** 2)))
