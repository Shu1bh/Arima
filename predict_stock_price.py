# -*- coding: utf-8 -*-

import pandas as pd
import pickle
import numpy as np
from sklearn import metrics


# modelop.init
def begin():
    
    global arima_model
    
    # load pickled logistic regression model
    arima_model = pickle.load(open("Arima_model.pickle", "rb"))
    
# modelop.score
def action(sample):
    global df_forecast, df_conf
    
    forecast,conf_int = arima_model.predict(n_periods=30,return_conf_int=True)
    df_forecast = pd.DataFrame(forecast,columns=['close_pred'])
    df_conf = pd.DataFrame(conf_int,columns= ['Upper_bound','Lower_bound'])
    
    yield [df_forecast.to_dict(orient="records"),df_conf.to_dict(orient="records")]

  # modelop.metrics
def metric(test):
    #global df_forecast
    forecast,conf_int = arima_model.predict(n_periods=30,return_conf_int=True)
    df_forecast = pd.DataFrame(forecast,columns=['close_pred'])
    # Turn data into DataFrame
    data = pd.DataFrame(test)
    
    y_true = data['Close']
    y_pred = df_forecast
    def mean_absolute_percentage_error(y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print('Evaluation metric results:-')
    print(f'MSE is : {metrics.mean_squared_error(y_true, y_pred)}')
    print(f'MAE is : {metrics.mean_absolute_error(y_true, y_pred)}')
    print(f'RMSE is : {np.sqrt(metrics.mean_squared_error(y_true, y_pred))}')
    print(f'MAPE is : {mean_absolute_percentage_error(y_true, y_pred)}')
    print(f'R2 is : {metrics.r2_score(y_true, y_pred)}',end='\n\n')
    result = {'MSE':metrics.mean_squared_error(y_true, y_pred),'MAE': metrics.mean_absolute_error(y_true, y_pred),
           'RMSE':np.sqrt(metrics.mean_squared_error(y_true, y_pred)), 'MAPE':mean_absolute_percentage_error(y_true, y_pred),
           'R2':metrics.r2_score(y_true, y_pred)}
    yield result
