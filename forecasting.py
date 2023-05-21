import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
# from neural_net import * 
from pp_functions import *

'''
this file puts the forecastdata in the right format in order to make predictions with the machine learning models. 
There are two different shapes the data can be put in. Different for first and second preprocessing method. 
'''

df_forecast = pd.read_csv('forecast/Forecast_13_05_2023.csv')

df_forecast['wind_direction'] = wind_direction(df_forecast['u10'], df_forecast['v10'])
df_forecast['wind_speed'] = np.sqrt(df_forecast['u10']**2 + df_forecast['v10']**2)
df_forecast['t2m'] = df_forecast['t2m']-273.15

df_forecast = df_forecast[['t2m', 'wind_direction', 'wind_speed', 'tp']]

# drop first 50 rows, because we have these for some reason... 
# df_forecast = df_forecast.tail(df_forecast.shape[0] - 50)

print(np.shape(df_forecast))
arr_forecast = np.array(df_forecast)
forecast_input_1 = np.zeros((1000,4))


for i in range(len(df_forecast)):
    print(i)
    forecast_input_1[i, 0:4] = arr_forecast[int((i%20)*50+np.floor(i/20)), :]




input_arr, no_array = makeTrainingSetOne(forecast_input_1, np.zeros((len(forecast_input_1, 3))))

df_one = pd.DataFrame(input_arr, columns=["T(t-1)", "WD(t-1)", "WS(t-1)", "P(t-1)", "T(t)", "WD(t)", "WS(t)", "P(t)", "T(t+1)", "WD(t+1)", "WS(t+1)", "P(t+1)"])

df_one.to_csv('forecast/training_input_1.csv')
df_forecast.to_csv('forecast/training_input_2.csv')


