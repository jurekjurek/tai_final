
import numpy as np
import pandas as pd
import math 
# from sklearn import tree
# from sklearn.model_selection import train_test_split
# import pickle
import matplotlib.pyplot as plt


def wind_direction(long, lat):
    '''
    this function actually returns the wind direction to be put in a machine learning model
    it calls the vectorize function in order to be able to calculate out of the respective u and v columns in our dataframe the winddirection 
    for every element in the column 
    '''
    vectorAngle = np.vectorize(Angle)
    wind_direction = vectorAngle(long, lat)
    return wind_direction

def Angle(long, lat):
    '''
    this function calculates the wind direction for a given longitude long and latitude lat which correspond to u and v
    depending on the magnitude and also on the sign of the respective components of the wind, the winddirection is calculated using the arctan function 
    '''
    if long > 0 and lat > 0:
        long = abs(long)
        lat = abs(lat)
        return np.arctan(long / lat)* 180/np.pi + 180

    if long < 0 and lat < 0:
        long = abs(long)
        lat = abs(lat)
        return np.arctan(long / lat)* 180/np.pi
    
    if long < 0 and lat > 0:
        long = abs(long)
        lat = abs(lat)
        return 180 - np.arctan(long / lat)* 180/np.pi
    
    if long > 0 and lat < 0:
        long = abs(long)
        lat = abs(lat)
        return 360 - np.arctan(long / lat)* 180/np.pi



def quantiles(prediction):
    '''
    this function accepts a numpy array of shape (1000, 3). In this array, the predictions for the weather forecast are stored 
    (:, 0) : temperature 
    (:, 1) : wind 
    (: ,2) : precipitation
    and produces a np.array of shape (60, 5). In total, we have 20 values for each variable (t,w,p) and 5 quantiles for every variable.
    This numpy array then gets converted into a pd.DataFrame where strings are attached to make it better understandable 
    The structure of the input array is : 
    # t w p 
    1
    1
    .
    .
    .
    1
    2
    2
    .
    .
    .
    50
    50

    in total 20*50 elements. The quantiles for the ensemble are calculated by simply taking the first 50 lines for the respective variables, then the next 50 and so on
    '''
    quantiles_arr = np.zeros((60, 5))
    for i in range(20):
        # temperature
        quantiles_arr[i, :] = np.array([np.quantile(prediction[i*50:i*50+50,0],0.025), np.quantile(prediction[i*50:i*50+50,0],0.25), 
                                        np.quantile(prediction[i*50:i*50+50,0],0.5), np.quantile(prediction[i*50:i*50+50,0],0.75), 
                                        np.quantile(prediction[i*50:i*50+50,0],0.975)]) 
        
        # wind
        quantiles_arr[i+20, :] = np.array([np.quantile(prediction[i*50:i*50+50,1],0.025), np.quantile(prediction[i*50:i*50+50,1],0.25), 
                                        np.quantile(prediction[i*50:i*50+50,1],0.5), np.quantile(prediction[i*50:i*50+50,1],0.75), 
                                        np.quantile(prediction[i*50:i*50+50,1],0.975)])
        
        # precipitation 
        quantiles_arr[i+40, :] = np.array([np.quantile(prediction[i*50:i*50+50,2],0.025), np.quantile(prediction[i*50:i*50+50,2],0.25), 
                                        np.quantile(prediction[i*50:i*50+50,2],0.5), np.quantile(prediction[i*50:i*50+50,2],0.75), 
                                        np.quantile(prediction[i*50:i*50+50,2],0.975)]) 
    
    # convert np.array to pd.DataFrame
    df_quantiles = pd.DataFrame(quantiles_arr, columns=['q0.025', 'q0.25', 'q0.5', 'q0.75', 'q0.975'])
    date_list    = []
    target_list  = []
    horizon_list = []
    for i in range(60):
        date_list.append("2023-05-06")
        if i < 20: 
            target_list.append("t2m")
            horizon_list.append(str(6*(i+1)) + ' hour')
        elif i < 40: 
            target_list.append('wind')
            horizon_list.append(str(6+6*((i)%20)) + ' hour')
        else: 
            target_list.append('precip')   
            horizon_list.append(str(6+6*((i)%20)) + ' hour') 
    df_quantiles = df_quantiles.assign(forecast_date = date_list)
    df_quantiles = df_quantiles.assign(horizon = horizon_list)
    df_quantiles = df_quantiles.assign(target = target_list)
    df_quantiles = df_quantiles[['forecast_date', 'target', 'horizon', 'q0.025', 'q0.25', 'q0.5', 'q0.75', 'q0.975']]

    # return the pd.DataFrame
    return df_quantiles


def precip_trafo(precip_data):
    '''
    in order to have the target precipitation data also in timesteps of 6hrs, sum up the first 6 values of precip (each corresponding to the total precip in this one hour, respectively)
    '''
    new_precip = pd.Series(precip_data.rolling(6).sum())
    return new_precip


def makeTrainingSetOne(Input_arr, Target_arr):
    """
    this function creates the first trainingset as described in the paper. 
    """
    training_input = np.zeros((len(Input_arr), 12))
    training_target = np.zeros((len(Input_arr), 3))
    for i in range(len(Input_arr)):
        if i%10000 == 0:
            print("done= ", i/len(Input_arr))
        if i%20 == 0:
            # if we start with a new five day forecast series for a new day, provide the value at time t for the value at time t-1
            training_input[i, 0:4] = Input_arr[i, :]
            training_input[i, 4:8] = Input_arr[i, :]
            training_input[i, 8:12] = Input_arr[i+1, :]
        elif i%20 == 19: 
            # if we just end with such a five day forecast serier, provide the value at time t for the value at time t+1
            training_input[i, 0:4] = Input_arr[i-1, :]
            training_input[i, 4:8] = Input_arr[i, :]
            training_input[i, 8:12] = Input_arr[i, :]
        else:
            # if none of the above cases apply, we are inside a five day forecast serier and just provide the values at t-1, t and t+1 for the respective variables 
            training_input[i, 0:4] = Input_arr[i-1, :]
            training_input[i, 4:8] = Input_arr[i, :]
            training_input[i, 8:12] = Input_arr[i+1, :]
        # after each five days, we reset to the next day
        # after one ensemble is over, we start over 

        # for the targetdata: 
        training_target[i, :] = Target_arr[int(i%20+np.floor(i/20))%(244), :]
    return training_input, training_target












