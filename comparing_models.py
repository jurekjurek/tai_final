import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import scale, StandardScaler, MinMaxScaler
from pp_functions import *

'''
this file compares the RMSEs of all the different machine learning models that were used to each other. 
It creates a bar plot and so on ...
'''

df_prediction  = pd.read_csv('forecast/prediction.csv')
df_prediction = df_prediction.drop(columns=df_prediction.columns[0])


'''
Loading models and scalers
'''
# model_ = "Neural Net"
# training_set_one = True

def compare(model_, training_set_one):
    if training_set_one:
        scaler_features = pickle.load(open("scaler_features_1", 'rb'))
        scaler_target   = pickle.load(open("scaler_target_1", 'rb'))
    else: 
        scaler_features = pickle.load(open("scaler_features_2", 'rb'))
        scaler_target   = pickle.load(open("scaler_target_2", 'rb'))

    if model_ == "Bayes":
        if training_set_one == True:
            temp_model = pickle.load(open("bayesian/temp_1", 'rb'))
            wind_model = pickle.load(open("bayesian/wind_1", 'rb'))
            prec_model = pickle.load(open("bayesian/prec_1", 'rb'))
        else: 
            temp_model = pickle.load(open("bayesian/temp_2", 'rb'))
            wind_model = pickle.load(open("bayesian/wind_2", 'rb'))
            prec_model = pickle.load(open("bayesian/prec_2", 'rb'))
    elif model_ == "Neural Net":
        if training_set_one == True:
            model = pickle.load(open("neural_net/net_1", 'rb'))
        else: 
            model = pickle.load(open("neural_net/net_2", 'rb'))
    elif model_ == "Tree":
        if training_set_one == True:
            temp_model = pickle.load(open("trees/tree_temp_1", 'rb'))
            wind_model = pickle.load(open("trees/tree_wind_1", 'rb'))
            prec_model = pickle.load(open("trees/tree_prec_1", 'rb'))
        else: 
            temp_model = pickle.load(open("trees/tree_temp_2", 'rb'))
            wind_model = pickle.load(open("trees/tree_wind_2", 'rb'))
            prec_model = pickle.load(open("trees/tree_prec_2", 'rb'))
    elif model_ == "Randomforest":
        if training_set_one == True:
            temp_model = pickle.load(open("randomforest/forest_temp_1", 'rb'))
            wind_model = pickle.load(open("randomforest/forest_wind_1", 'rb'))
            prec_model = pickle.load(open("randomforest/forest_prec_1", 'rb'))
        else: 
            temp_model = pickle.load(open("randomforest/forest_temp_2", 'rb'))
            wind_model = pickle.load(open("randomforest/forest_wind_2", 'rb'))
            prec_model = pickle.load(open("randomforest/forest_prec_2", 'rb'))
    elif model_ == "K Nearest Neighbours": 
        if training_set_one == True:
            temp_model = pickle.load(open("knn/temp_1", 'rb'))
            wind_model = pickle.load(open("knn/wind_1", 'rb'))
            prec_model = pickle.load(open("knn/precip_1", 'rb'))
        else: 
            temp_model = pickle.load(open("knn/temp_2", 'rb'))
            wind_model = pickle.load(open("knn/wind_2", 'rb'))
            prec_model = pickle.load(open("knn/precip_2", 'rb'))


    if training_set_one == True:
        x_test  = np.loadtxt('x_test_1')
        y_test  = np.loadtxt('y_test_1')
    else: 
        x_test  = np.loadtxt('x_test_2')
        y_test  = np.loadtxt('y_test_2')


    if model_ == "K Nearest Neighbours" or model_ == "Bayes":
        result = scaler_target.inverse_transform( np.transpose( np.vstack ((temp_model.predict(x_test), wind_model.predict(x_test), prec_model.predict(x_test)) )))

        return [np.sqrt(np.mean((result[:,0] - y_test[:,0]) ** 2)), np.sqrt(np.mean((result[:,1] - y_test[:,1]) ** 2)), np.sqrt(np.mean((result[:,2] - y_test[:,2]) ** 2))]

    elif model_ == "Neural Net":
        result = scaler_target.inverse_transform( model.predict(x_test) )

        return [np.sqrt(np.mean((result[:,0] - y_test[:,0]) ** 2)), np.sqrt(np.mean((result[:,1] - y_test[:,1]) ** 2)), np.sqrt(np.mean((result[:,2] - y_test[:,2]) ** 2))]

    elif model_ == "Tree" or model_ == "Randomforest": 

        return [ np.sqrt(np.mean((temp_model.predict(x_test) - y_test[:,0]) ** 2)) ,np.sqrt(np.mean((wind_model.predict(x_test) - y_test[:,1]) ** 2)) ,np.sqrt(np.mean((prec_model.predict(x_test) - y_test[:,2]) ** 2)) ]

Neural_Net_1 = compare("Neural Net", True)
Neural_Net_2 = compare("Neural Net", False)

Trees_1 = compare("Tree", True)
Trees_2 = compare("Tree", False)

Randomforest_1 = compare("Randomforest", True)
Randomforest_2 = compare("Randomforest", False)

Bayes_1 = compare("Bayes", True)
Bayes_2 = compare("Bayes", False)

K_Nearest_Neighbours_1 = compare("K Nearest Neighbours", True)
K_Nearest_Neighbours_2 = compare("K Nearest Neighbours", False)

temp_1 = [Neural_Net_1[0], Trees_1[0], Randomforest_1[0], Bayes_1[0], K_Nearest_Neighbours_1[0]]
temp_2 = [Neural_Net_2[0], Trees_2[0], Randomforest_2[0], Bayes_2[0], K_Nearest_Neighbours_2[0]]


wind_1 = [Neural_Net_1[1], Trees_1[1], Randomforest_1[1], Bayes_1[1], K_Nearest_Neighbours_1[1]]
wind_2 = [Neural_Net_2[1], Trees_2[1], Randomforest_2[1], Bayes_2[1], K_Nearest_Neighbours_2[1]]

prec_1 = [Neural_Net_1[2], Trees_1[2], Randomforest_1[2], Bayes_1[2], K_Nearest_Neighbours_1[2]]
prec_2 = [Neural_Net_2[2], Trees_2[2], Randomforest_2[2], Bayes_2[2], K_Nearest_Neighbours_2[2]]


plt.figure()
plt.bar(["Neural Net", "Tree", "Randomforest", "Bayes", "K N N"],temp_1)
plt.xlabel("Model")
plt.ylabel("RMSE")
plt.title("Comparison, training set one")
plt.savefig('final/compare_temp_1.png')

plt.figure()
plt.bar(["Neural Net", "Tree", "Randomforest", "Bayes", "K N N"],wind_1)
plt.xlabel("Model")
plt.ylabel("RMSE")
plt.title("Comparison, training set one")
plt.savefig('final/compare_wind_1.png')

plt.figure()
plt.bar(["Neural Net", "Tree", "Randomforest", "Bayes", "K N N"],prec_1)
plt.xlabel("Model")
plt.ylabel("RMSE")
plt.title("Comparison, training set one")
plt.savefig('final/compare_prec_1.png')



plt.figure()
plt.bar(["Neural Net", "Tree", "Randomforest", "Bayes", "K N N"],temp_2)
plt.xlabel("Model")
plt.ylabel("RMSE")
plt.title("Comparison, training set two")
plt.savefig('final/compare_temp_2.png')

plt.figure()
plt.bar(["Neural Net", "Tree", "Randomforest", "Bayes", "K N N"],wind_2)
plt.xlabel("Model")
plt.ylabel("RMSE")
plt.title("Comparison, training set two")
plt.savefig('final/compare_wind_2.png')

plt.figure()
plt.bar(["Neural Net", "Tree", "Randomforest", "Bayes", "K N N"],prec_2)
plt.xlabel("Model")
plt.ylabel("RMSE")
plt.title("Comparison, training set two")
plt.savefig('final/compare_prec_2.png')

