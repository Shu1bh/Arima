# -*- coding: utf-8 -*-

import pandas as pd
import pickle
import numpy as np
from sklearn import metrics

# modelop.init
def begin():
    
    global arima_model
    
    # load pickled logistic regression model
    arima_model = pickle.load(open("Arima_model.sav", "rb"))

def action(data):
    
    # Turn data into DataFrame
    data = pd.DataFrame([data])
    
    data["forecasted"] = arima_model.predict(n_periods=30)
    
    yield data.to_dict(orient="records")

def metrics(data):
    # Turn data into DataFrame
    data = pd.DataFrame([data])
    y_true = data['Close']
    y_pred = ['forecasted']
    def mean_absolute_percentage_error(y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print('Evaluation metric results:-')
    print(f'MSE is : {metrics.mean_squared_error(y_true, y_pred)}')
    print(f'MAE is : {metrics.mean_absolute_error(y_true, y_pred)}')
    print(f'RMSE is : {np.sqrt(metrics.mean_squared_error(y_true, y_pred))}')
    print(f'MAPE is : {mean_absolute_percentage_error(y_true, y_pred)}')
    print(f'R2 is : {metrics.r2_score(y_true, y_pred)}',end='\n\n')
    yield {'MSE':metrics.mean_squared_error(y_true, y_pred),'MAE': metrics.mean_absolute_error(y_true, y_pred),
           'RMSE':np.sqrt(metrics.mean_squared_error(y_true, y_pred)), 'MAPE':mean_absolute_percentage_error(y_true, y_pred),
           'R2':metrics.r2_score(y_true, y_pred)}

