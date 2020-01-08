# -*- coding: utf-8 -*-
"""
Created on Wed Jan 08 18:00:00 2020
@author: BRAJA GOPAL SAHOO
"""

### Provide directory location ###
import os
os.chdir("C:/Projects/..")

### Import required library ###
from sklearn.svm import SVR
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import copy 
from math import pi
import warnings

### Upload data ###
df = pd.read_csv ("Volume Data for Model.csv")

### Create Fourier components as one of feature(s) to the model ###
## Definition to identify period 
# Daily data - period weekly = 7, monthly = 365.25/12, quarterly = 365.25/4, yearly = 365.25
# Weekly data - period weekly = 1, monthly = (365.25/7)/12, quarterly = (365.25/7)/4, yearly = (365.25/7)
# Monthly data -  monthly = 1, quarterly = 12/4, yearly = 12
# Similar for other types
class FourierComponentWarning(UserWarning):
    pass

def fourier_components(data,no_of_components,period,historical_data_len,forecast_len,seasonality_type):
    output = pd.DataFrame()
    if 2*no_of_components > period:
        warnings.warn("No of components should be less or eqaul to half of period. Used default value",FourierComponentWarning)
        no_of_components = int((period/2))
    else:
        no_of_components = no_of_components
    
    tmp_data = copy.deepcopy(data)
    if forecast_len == 0:
        tmp_data['row_number'] = np.arange(historical_data_len)+1
    else:
        tmp_data['row_number'] = np.arange(forecast_len)+1+historical_data_len
        
    for i in range(1,no_of_components+1):
        output[str(seasonality_type)+"_sin_component_"+str(i)+"period_"+str(period)] = np.sin(2*pi*i*tmp_data['row_number']/period) 
        output[str(seasonality_type)+"_cos_component_"+str(i)+"period_"+str(period)] = np.cos(2*pi*i*tmp_data['row_number']/period)
    return output

# test_fourierFunct = fourier_components(data = df,no_of_components=1,period=30.25)

### Partition data into train and test ###
def split_train_test(X_data,y_data,split_type,test_size):
    if split_type == "Timeseries":
        if isinstance(test_size,float):
            test_size = int(X_data.shape[0]*test_size)
        if isinstance(test_size,int): 
            test_size = test_size 
        X_train = X_data.iloc[:-test_size,:]
        X_test = X_data.iloc[-test_size:,:]
        y_train = y_data[:-test_size]
        y_test = y_data[-test_size:] 
    if split_type == "random":
        if isinstance(test_size,float):
            test_size = test_size
        if isinstance(test_size,int):
            test_size = float(test_size/X_data.shape[0])
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=test_size) 
    return X_train, X_test, y_train, y_test


### Parameter range creation ###
def para_range(minimum,maximum,base):
    r = list(range(minimum,maximum))
    para_list = []
    for i,val in enumerate(r):
        para_list.append(pow(base,r[i]))
    return para_list

### Transform target variable ###
def transformation_fn (tran_type, data):
    if tran_type == "log":
        data = np.log(data)
    if tran_type == "exp":
        data = np.exp(data)
    if tran_type == "inv_exp":
        data = np.exp(1/data)
    if tran_type == "no_tran":
        data = data
    return data

### Inverse Transformed target variable ###
def inverse_transformation_fn (tran_type, data):
    if tran_type == "log":
        data = np.exp(data)
    if tran_type == "exp":
        data = np.log(data)
    if tran_type == "inv_exp":
        data = 1/np.log(data)
    if tran_type == "no_tran":
        data = data
    return data

### Accuracy mesure metrics for model evaluation ###
def accuracy_metric(metric, actual, pred, tran_type):
    if metric == "MAPE":
        result = round(100*np.mean(np.abs(inverse_transformation_fn (tran_type = tran_type, data = actual) - inverse_transformation_fn (tran_type = tran_type, data = pred))/inverse_transformation_fn (tran_type = tran_type, data = actual)),2)
    return result

### Model hyper parameter set-up  ###
Volume_data1 = copy.deepcopy(df)
max_lag_period = 5
gam = para_range(-5,1,2)
cost = para_range(-1,8,2)
ker = ["rbf","sigmoid"] #,"poly"
no_forecast_components = list(range(0,5,1))
result_MAPE = []
Total_run = (max_lag_period*len(gam)*len(cost)*len(ker)*len(no_forecast_components))
s_type = "weekly" #Seasonality Type
p = 7 #period
tran_type = "log"
split_type = "Timeseries"
test_size = 30
forecast_lenght = 40

### Parameter tunning by grid-search method ###
ind = 1

for fc in no_forecast_components:
    if fc*2>p:
        warnings.warn("STOPPED: No of fourier components exceed half of period",FourierComponentWarning)
        break
    Volume_data2 = copy.deepcopy(Volume_data1)
    if fc==0:
        Volume_data = Volume_data2
    else:
        quarter_fc = fourier_components(data = Volume_data2 ,no_of_components = fc,period = p,historical_data_len = len(Volume_data2),forecast_len = 0,seasonality_type = str(s_type))
        Volume_data = pd.concat([Volume_data2,quarter_fc], axis=1)
        
    for l in range (1,max_lag_period+1):
        for il in range (1,l+1):
            Volume_data["Lag"+str(il)] = transformation_fn(tran_type = tran_type,data = Volume_data["Total_Daily_Trnx"].shift(il))
        data_after_lag = Volume_data[l:]
        X_data = data_after_lag.iloc[:,2:]
        y_data = transformation_fn(tran_type = tran_type, data = data_after_lag["Total_Daily_Trnx"])
        X_train, X_test, y_train, y_test = split_train_test(X_data = X_data,y_data = y_data,split_type = split_type ,test_size = test_size)
        for k in ker:
            for g in gam:
                for costi in cost:
                    svmFit = SVR ( 
                          kernel= str(k), 
                          gamma = g,
                          C = costi,
                          verbose = False
                          )
                    svmFit.fit(X_train, y_train)
                    MAPE = accuracy_metric(metric = "MAPE", actual = y_test , pred = svmFit.predict(X_test), tran_type = tran_type)
                    result_MAPE.append([s_type,fc,l,svmFit.kernel,svmFit.gamma,svmFit.C,svmFit.epsilon,svmFit.tol,svmFit.degree,MAPE])
                    print(str(ind)+"/"+str(Total_run),[s_type,fc,l,svmFit.kernel,svmFit.gamma,svmFit.C,svmFit.epsilon,svmFit.tol,svmFit.degree,MAPE])
                    ind = ind+1
                

### Find best model and its parameters ###
Results = pd.DataFrame(result_MAPE, columns = ["Seasonality_Type","No_of_Four_comp","Max_Lag","Kernel","Gamma","C","Epsilon","Tol","Degree","MAPE"])
BestSVmParameters = Results[Results.MAPE == min(Results.MAPE)]

###  Create model object using best model parameters ###
(BestSVmParameters)
Volume_data = copy.deepcopy(df)

if int(BestSVmParameters["No_of_Four_comp"])==0:
    Volume_data = Volume_data
else:
    quarter_fc = fourier_components(data = Volume_data ,no_of_components = int(BestSVmParameters["No_of_Four_comp"]),period = p,historical_data_len = len(Volume_data),forecast_len = 0,seasonality_type = list(BestSVmParameters["Seasonality_Type"])[0])
    Volume_data = pd.concat([Volume_data,quarter_fc], axis=1)
    
for il in range (1,int(BestSVmParameters["Max_Lag"])+1):
    Volume_data["Lag"+str(il)] = transformation_fn(tran_type = tran_type,data = Volume_data["Total_Daily_Trnx"].shift(il))
data_after_lag = Volume_data[int(BestSVmParameters["Max_Lag"]):]
X_data = data_after_lag.iloc[:,2:]
y_data = transformation_fn(tran_type = tran_type, data = data_after_lag["Total_Daily_Trnx"])
X_train, X_test, y_train, y_test = split_train_test(X_data = X_data,y_data = y_data,split_type = split_type ,test_size = test_size)
  
BestsvmFit = SVR(epsilon = float(BestSVmParameters["Epsilon"]),
                      kernel= list(BestSVmParameters["Kernel"])[0], 
                      gamma = float(BestSVmParameters["Gamma"]),
                      C = float(BestSVmParameters["C"]),
                      tol = float(BestSVmParameters["Tol"]),
                      degree = float(BestSVmParameters["Degree"]),
                      verbose = False)
BestsvmFit.fit(X_train,y_train)

### Predict train & test using best model
pd.set_option('display.float_format', lambda x: '%.2f' % x) #to display result in numeric
Final_test_predict = pd.DataFrame()
Final_test_predict["Actual"] = inverse_transformation_fn (tran_type = tran_type, data = y_test)
Final_test_predict["Predicted"] = inverse_transformation_fn (tran_type = tran_type,data = BestsvmFit.predict(X_test))

Final_train_predict = pd.DataFrame()
Final_train_predict["Actual"] = inverse_transformation_fn (tran_type = tran_type, data = y_train)
Final_train_predict["Predicted"] = inverse_transformation_fn (tran_type = tran_type,data = BestsvmFit.predict(X_train))

### Write train and test result  ###
Final_test_predict.to_csv("Test_Period_Data.csv")
Final_train_predict.to_csv("Train_Period_Data.csv")

### Forecast Using Bestmodel object ###
## Prepare data if require ##
# Required library
from datetime import timedelta 

# Forecast parameters - period length        
last_date_position = data_after_lag["creation_date"].shape[0]+BestSVmParameters["Max_Lag"]-1


#Generate date charecteristics variables
start_date = pd.to_datetime(data_after_lag["creation_date"][last_date_position])+timedelta(1)
#start_date = dt.date.today()
last_date = start_date+timedelta(forecast_lenght)
toForecast = pd.DataFrame()
toForecast["creation_date"] = pd.date_range(list(start_date)[0], list(last_date)[0], freq='D')
toForecast["Day_of_week"] = pd.to_datetime(toForecast["creation_date"]).dt.dayofweek+2
toForecast.loc[toForecast["Day_of_week"]==8,"Day_of_week"] = 1
toForecast["First_Day_month"] = pd.to_datetime(toForecast["creation_date"]).dt.day
toForecast.loc[toForecast["First_Day_month"]!=1,"First_Day_month"] = 0
toForecast["Last_Day_of_month"] = toForecast["First_Day_month"].shift(-1)
toForecast["Last_Day_of_month"] = toForecast["Last_Day_of_month"].fillna(0)
toForecast["Week_of_the_month"] = np.ceil(pd.DatetimeIndex(toForecast["creation_date"]).day/7)
toForecast["Month_of_the_Year"] = pd.DatetimeIndex(toForecast["creation_date"]).month

# Add holiday component to the data

listofholiday = pd.read_csv("Holiday List upto Dec19.csv")
listofholiday["Date"] = pd.to_datetime(listofholiday["Date"])
listofholiday = listofholiday.rename(columns={"Date":"Date","Gazetted Holiday":"Gazetted","Muslim, Common local holiday":"Muslim_Common_local","Observance":"Observance","Restricted Holiday":"Regional"})

#toForecast = toForecast.set_index('creation_date').join(listofholiday.set_index('Date'))
toForecast = pd.merge(toForecast,listofholiday,how = 'left',left_on='creation_date', right_on='Date',)
toForecast = toForecast.drop(['Date'],axis=1)
toForecast = toForecast.fillna(0)

if int(BestSVmParameters["No_of_Four_comp"])==0:
    toForecast = toForecast
else:
    quarter_fc = fourier_components(data = toForecast ,no_of_components = int(BestSVmParameters["No_of_Four_comp"]),period = p,historical_data_len = len(Volume_data),forecast_len = len(toForecast),seasonality_type = list(BestSVmParameters["Seasonality_Type"])[0])
    toForecast = pd.concat([toForecast,quarter_fc], axis=1)

# Create place holder variable for lag variable
for il in range (1,int(BestSVmParameters["Max_Lag"])+1):
    toForecast["Lag"+str(il)] = float(0)

# Iterate forecast for entire period through single step prediction
Forecast_vec_test = list()
Forecast_vec_test.append(list(y_test.iloc[-int(BestSVmParameters["Max_Lag"]):]))
Forecast_vec =  Forecast_vec_test[0]                         #[item for sublist in Forecast_vec_test for item in sublist]

for f in range (0,forecast_lenght+1):
    for i in range(1,int(BestSVmParameters["Max_Lag"])+1):
        toForecast["Lag"+str(i)].loc[f] = Forecast_vec[-i]                                  
    tmp_data_for_forecast = np.array(toForecast.iloc[f,1:])    
    Forecast_vec.append(list(BestsvmFit.predict(tmp_data_for_forecast.reshape(1,-1)))[0])
    
toForecast["Forecasted_Value"] =  inverse_transformation_fn (tran_type = tran_type, data = Forecast_vec[int(BestSVmParameters["Max_Lag"]):])
toForecast.to_csv("Forecasted_Volume.csv")

