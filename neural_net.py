import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.model_selection import train_test_split, cross_val_score
import random
from sklearn.preprocessing import scale, StandardScaler, MinMaxScaler
# linear regression
from sklearn.linear_model import LinearRegression
# neural net
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import pickle

'''
this file is fitting a neural net to the data, varying the number of epochs used for training and plotting the loss curves 
'''

# what trainingset do we want to use 
training_set_one = True

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



clf = MLPRegressor(solver='adam', alpha= 1e-5, hidden_layer_sizes=(7,6), random_state=0)

loss_values = {'loss_train_temp': [],'loss_train_wind': [], 'loss_train_prec': [], 'loss_val_temp': [], 'loss_val_wind': [], 'loss_val_prec': []}

max_number_epochs = 20

for i in range(1, max_number_epochs):
    # clf = MLPRegressor(solver='adam', alpha= 1e-5, hidden_layer_sizes=(10,10,10), random_state=0, max_iter = i)
    # partial fit only works for alan, bc apparently, it's stochastic
    clf.partial_fit(x_train, y_train)

    y_pred_train = scaler_target.inverse_transform( clf.predict(x_train) )
    y_pred_val   = scaler_target.inverse_transform( clf.predict(x_val) )   

    rmse_train_temp = np.sqrt(np.mean((y_pred_train[:,0] - y_train_unscaled[:,0]) ** 2))
    rmse_train_wind = np.sqrt(np.mean((y_pred_train[:,1] - y_train_unscaled[:,1]) ** 2))
    rmse_train_prec = np.sqrt(np.mean((y_pred_train[:,2] - y_train_unscaled[:,2]) ** 2))

    rmse_val_temp   = np.sqrt(np.mean((y_pred_val[:,0] - y_val[:,0])**2))
    rmse_val_wind   = np.sqrt(np.mean((y_pred_val[:,1] - y_val[:,1])**2))
    rmse_val_prec   = np.sqrt(np.mean((y_pred_val[:,2] - y_val[:,2])**2))

    loss_values['loss_train_temp'].append(rmse_train_temp)
    loss_values['loss_train_wind'].append(rmse_train_wind)
    loss_values['loss_train_prec'].append(rmse_train_prec)

    loss_values['loss_val_temp'].append(rmse_val_temp)
    loss_values['loss_val_wind'].append(rmse_val_wind)
    loss_values['loss_val_prec'].append(rmse_val_prec)


if training_set_one == True:
    pickle.dump(clf, open("neural_net/net_1", "wb"))
else: 
    pickle.dump(clf, open("neural_net/net_2", "wb"))

plt.figure()
plt.title("Temperature")
plt.plot(loss_values['loss_train_temp'], label="training")
plt.plot(loss_values['loss_val_temp'], label="validation")
plt.legend()
plt.xlabel("Number of epochs")
plt.ylabel("RMSE")
if training_set_one == True:
    plt.title("Temperature, training set one")
    plt.savefig('neural_net/temp_1.png')
else:
    plt.title("Temperature, training set two")
    plt.savefig('neural_net/temp_2.png')

plt.figure()
plt.title("Wind")
plt.plot(loss_values['loss_train_wind'], label="training")
plt.plot(loss_values['loss_val_wind'], label="validation")
plt.legend()
plt.xlabel("Number of epochs")
plt.ylabel("RMSE")
if training_set_one == True:
    plt.title("Wind, training set one")
    plt.savefig('neural_net/wind_1.png')
else:
    plt.title("Wind, training set two")
    plt.savefig('neural_net/wind_2.png')


plt.figure()
plt.plot(loss_values['loss_train_prec'], label="training")
plt.plot(loss_values['loss_val_prec'], label="validation")
plt.legend()
plt.xlabel("Number of epochs")
plt.ylabel("RMSE")
if training_set_one == True:
    plt.title("Precipitation, training set one")
    plt.savefig('neural_net/prec_1.png')
else:
    plt.title("Precipitation, training set two")
    plt.savefig('neural_net/prec_2.png')



total_pred = scaler_target.inverse_transform( clf.predict(x_test) )

print("The final total loss value is: ", np.sqrt(np.mean((total_pred - y_test) ** 2)))
print("For Temperature: ", np.sqrt(np.mean((total_pred[:,0] - y_test[:,0]) ** 2)))
print("For Windspeed: ", np.sqrt(np.mean((total_pred[:,1] - y_test[:,1]) ** 2)))
print("For Precipitation: ", np.sqrt(np.mean((total_pred[:,2] - y_test[:,2]) ** 2)))






