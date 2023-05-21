from pp_functions import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import scale, StandardScaler, MinMaxScaler
import pickle
from sklearn.model_selection import train_test_split, cross_val_score

'''
In this file, the two datasets for actually training the machine learning model are being produced. 

What we want to do now is create an input dataset for the training. The first attempt looks as follows: 
We are going to, for each set of 20 values (5 days forecast for each day) import the value for the temperature, precipitation, windspeed and -angle for the timesteps t, t-1 and t+1 
where t-1 and t+1 correspond to the values at times - or + 6h. Where the first of the 20 is t,t,t+1 and the last one t-1,t
We are going to create a 12 x (len(input)) - dimensional dataset with the rows consisting of T(t-1) P(t-1) WD(t-1) WS(t-1) T(t) ... 
and the respective target dataset looks as follows: 
T(t) P(t) WD(t) WS(t)
T(t+1) P(t+1) WD(t+1) WS(t+1)
T(t+2) P(t+2) WD(t+2) WS(t+2)
.
.
.
T(t+19) P(t+19) WD(t+19) WS(t+19)
T(t+1) P(t+1) WD(t+1) WS(t+1)           // here the five day forecast for the second day starts 
and so on

second attempt: we create the target data and input data for the second approach, just feeding in the data

'''

# load in prepared data for preprocessing 
df_true = pd.read_csv('data/true_data.csv')
df_input = pd.read_csv('data/forecastdata.csv')


# turn them into np.arrays, drop columns that are not interesting for training 
input_arr = np.array(df_input.drop(columns=['datetime', 'number']))
target_arr = np.array(df_true.drop(columns=['datetime', 'local_datetime', 'wind_direction']))


# this function creates the input dataset, it takes numpy arrays in, gives numpy arrays out
training_input, training_target = makeTrainingSetOne(input_arr, target_arr)

# create pandas dataframes out of np arrays 
df_input_1 = pd.DataFrame(training_input, columns=["T(t-1)", "WD(t-1)", "WS(t-1)", "P(t-1)", "T(t)", "WD(t)", "WS(t)", "P(t)", "T(t+1)", "WD(t+1)", "WS(t+1)", "P(t+1)"])
df_target_1 = pd.DataFrame(training_target, columns=["T(t)", "WS(t)", "P(t)"])


df_target_2 = df_target_1
df_input_2  = df_input 

# we are not interested in the winddirection in the targetdataset, drop it. 
# df_target = df_target.drop(columns=['WD(t)'])



# in order to get rid of the NaN values, fuse them together, use dropna() and separate them again. This way, we get 2 arrays of the same shape back. 
# We have to do this because there are no NaN values in target, but some in input, and the respective lines in target would not be dropped

# for unprocessed data
df_input_2[['temp_true', 'wind_speed_true', 'precip_quantity_6hour_true']] = df_target_2[['T(t)', 'WS(t)', 'P(t)']]
# drop the NaN values
df_input_2 = df_input_2.dropna() 
# create new dataframe
df_target_2 = pd.DataFrame()
df_target_2[['T(t)', 'WS(t)', 'P(t)']] = df_input_2[['temp_true', 'wind_speed_true', 'precip_quantity_6hour_true']]
df_input_2 = df_input_2.drop(columns=['temp_true', 'wind_speed_true', 'precip_quantity_6hour_true', 'datetime', 'number'])


# for timestep data
df_input_1[['temp_true', 'wind_speed_true', 'precip_quantity_6hour_true']] = df_target_1[['T(t)', 'WS(t)', 'P(t)']]
# drop the NaN values
df_input_1 = df_input_1.dropna() 
# create new dataframe
df_target_1 = pd.DataFrame()
df_target_1[['T(t)', 'WS(t)', 'P(t)']] = df_input_1[['temp_true', 'wind_speed_true', 'precip_quantity_6hour_true']]
df_input_1 = df_input_1.drop(columns=['temp_true', 'wind_speed_true', 'precip_quantity_6hour_true'])


print("Before minmaxscaling, so after all the preprocessing the data looks like this. Shape input_1: ", np.shape(df_input_1), 
      ", Shape input_2: ", np.shape(df_input_2), ", Shape target_1: ", np.shape(df_target_1), ", Shape target_2: ", np.shape(df_target_2))

# now, minmaxscale the data
arr_target_1 = np.array(df_target_1)
arr_input_1  = np.array(df_input_1)

arr_target_2 = np.array(df_target_2)
arr_input_2  = np.array(df_input_2)



#####

# splitting into 80, 10, 10
x_train_1, x_valtest_1, y_train_1, y_valtest_1 = train_test_split(arr_input_1, arr_target_1, test_size = 0.2, random_state = 0, shuffle=True)
x_val_1, x_test_1, y_val_1, y_test_1 = train_test_split(x_valtest_1, y_valtest_1, test_size= 0.5, random_state = 0, shuffle=True)


x_train_2, x_valtest_2, y_train_2, y_valtest_2 = train_test_split(arr_input_2, arr_target_2, test_size = 0.2, random_state = 0, shuffle=True)
x_val_2, x_test_2, y_val_2, y_test_2 = train_test_split(x_valtest_2, y_valtest_2, test_size= 0.5, random_state = 0, shuffle=True)


# scale input and target so that they have values between 0 and 1 
# but ONLY FIT THE TRAINING DATA, NOT THE TARGET OR THE VALIDATION DATA!!!!
scaler_features_1 = MinMaxScaler((0,1))
x_train_1  = scaler_features_1.fit_transform(x_train_1)
x_val_1    = scaler_features_1.transform(x_val_1)
x_test_1   = scaler_features_1.transform(x_test_1)

scaler_target_1 = MinMaxScaler((0,1))
scaler_target_1.fit(y_train_1)
# y_train_1 = scaler_target_1.inverse_transform(y_train_1)
# y_val_1    = scaler_target_1.transform(y_val_1)
# y_test_1   = scaler_target_1.transform(y_test_1)

scaler_features_2 = MinMaxScaler((0,1))
x_train_2  = scaler_features_2.fit_transform(x_train_2)
x_val_2    = scaler_features_2.transform(x_val_2)
x_test_2   = scaler_features_2.transform(x_test_2)

scaler_target_2 = MinMaxScaler((0,1))
scaler_target_2.fit(y_train_2)
# y_train_2 = scaler_target_2.inverse_transform(y_train_2)
# y_val_2    = scaler_target_2.transform(y_val_2)
# y_test_2   = scaler_target_2.transform(y_test_2)


pickle.dump(scaler_features_1, open("scaler_features_1", 'wb'))
pickle.dump(scaler_target_1, open("scaler_target_1", 'wb'))

pickle.dump(scaler_features_2, open("scaler_features_2", 'wb'))
pickle.dump(scaler_target_2, open("scaler_target_2", 'wb'))



np.savetxt('x_train_1', x_train_1, fmt='%f')
np.savetxt('y_train_1', y_train_1, fmt='%f')
np.savetxt('x_val_1', x_val_1, fmt='%f')
np.savetxt('y_val_1', y_val_1, fmt='%f')
np.savetxt('x_test_1', x_test_1, fmt='%f')
np.savetxt('y_test_1', y_test_1, fmt='%f')



np.savetxt('x_train_2', x_train_2, fmt='%f')
np.savetxt('y_train_2', y_train_2, fmt='%f')
np.savetxt('x_val_2', x_val_2, fmt='%f')
np.savetxt('y_val_2', y_val_2, fmt='%f')
np.savetxt('x_test_2', x_test_2, fmt='%f')
np.savetxt('y_test_2', y_test_2, fmt='%f')


# target 1 and 2 are the same 
# finally, save preprocessed data 
df_input_1.to_csv("data/input_data_1.csv")
df_target_1.to_csv("data/target_data.csv")

df_input_2.to_csv("data/input_data_2.csv")


print("check")