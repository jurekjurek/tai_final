import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import scale, StandardScaler, MinMaxScaler
from pp_functions import *

'''
this file calculates the quantiles for a given forecast. 
'''


df_prediction  = pd.read_csv('forecast/prediction.csv')
df_prediction = df_prediction.drop(columns=df_prediction.columns[0])


'''
Loading models and scalers
'''
model = ""
trainingsetone = True

if trainingsetone:
    scaler_features = pickle.load(open("scaler_features_1", 'rb'))
    scaler_target   = pickle.load(open("scaler_target_1", 'rb'))
else: 
    scaler_features = pickle.load(open("scaler_features_2", 'rb'))
    scaler_target   = pickle.load(open("scaler_target_2", 'rb'))

if model == "Bayes":
    if trainingsetone == True:
        temp_model = pickle.load(open("bayesian/temp_1", 'rb'))
        wind_model = pickle.load(open("bayesian/wind_1", 'rb'))
        prec_model = pickle.load(open("bayesian/prec_1", 'rb'))
    else: 
        temp_model = pickle.load(open("bayesian/temp_2", 'rb'))
        wind_model = pickle.load(open("bayesian/wind_2", 'rb'))
        prec_model = pickle.load(open("bayesian/prec_2", 'rb'))
elif model == "Neural Net":
    if trainingsetone == True:
        model = pickle.load(open("neural_net/nn_1", 'rb'))
    else: 
        model = pickle.load(open("neural_net/nn_2", 'rb'))
elif model == "Tree":
    if trainingsetone == True:
        temp_model = pickle.load(open("trees/tree_temp_1", 'rb'))
        wind_model = pickle.load(open("trees/tree_wind_1", 'rb'))
        prec_model = pickle.load(open("trees/tree_prec_1", 'rb'))
    else: 
        temp_model = pickle.load(open("trees/tree_temp_2", 'rb'))
        wind_model = pickle.load(open("trees/tree_wind_2", 'rb'))
        prec_model = pickle.load(open("trees/tree_prec_2", 'rb'))
elif model == "Randomforest":
    if trainingsetone == True:
        temp_model = pickle.load(open("randomforest/forest_temp_1", 'rb'))
        wind_model = pickle.load(open("randomforest/forest_wind_1", 'rb'))
        prec_model = pickle.load(open("randomforest/forest_prec_1", 'rb'))
    else: 
        temp_model = pickle.load(open("randomforest/forest_temp_2", 'rb'))
        wind_model = pickle.load(open("randomforest/forest_wind_2", 'rb'))
        prec_model = pickle.load(open("randomforest/forest_prec_2", 'rb'))
elif model == "K Nearest Neighbours": 
    if trainingsetone == True:
        temp_model = pickle.load(open("knn/temp_1", 'rb'))
        wind_model = pickle.load(open("knn/wind_1", 'rb'))
        prec_model = pickle.load(open("knn/prec_1", 'rb'))
    else: 
        temp_model = pickle.load(open("knn/temp_2", 'rb'))
        wind_model = pickle.load(open("knn/wind_2", 'rb'))
        prec_model = pickle.load(open("knn/prec_2", 'rb'))

'''
INPUT SET ONE
'''


if trainingsetone:
    df_forecast = pd.read_csv('forecast/training_input_1.csv')
    df_forecast  = df_forecast.drop(columns = df_forecast.columns[0])
    arr_forecast = np.array(df_forecast)
else: 
    df_forecast = pd.read_csv('forecast/training_input_2.csv')
    df_forecast  = df_forecast.drop(columns = df_forecast.columns[0])
    arr_forecast = np.array(df_forecast)

# scale according to MinMax
arr_forecast = scaler_features.transform(arr_forecast)

if model == "Neural Net": 
    total_pred = model.predict(arr_forecast)
else:
    t_pred = temp_model.predict(arr_forecast)
    w_pred = wind_model.predict(arr_forecast)
    p_pred = prec_model.predict(arr_forecast)

    total_pred = np.zeros((1000,3))
    total_pred[:,0] = t_pred
    total_pred[:,1] = w_pred
    total_pred[:,2] = p_pred

total_pred = scaler_target.inverse_transform(total_pred)

# FOR THE FIRST DATASET: 
# convert back into usual format; first 20 values, next 20 values and so on and save prediction 
if trainingsetone:
    new_arr = np.zeros((1000,3))
    for i in range(len(total_pred)):
        new_arr[i, :] = total_pred[int((i%20)*20 + np.floor(i/20))]
    prediction_ = quantiles(new_arr)
else: 
    prediction_ = quantiles(total_pred) 

prediction_.to_csv('forecast/20230508_PaoloConte.csv')












