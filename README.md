# Time Series Models With SVM
To build model for time series data in ARIMA set up, we use features like auto regressive, stationarity, moving average, seasonality  components of the original series itself along with additional exogenous variables.

In similar fashion, here I have integrated auto regressive and seasonality features of original series along with additional exogenous variables in SVM set up to build model for Time Series Data. Also there is a option for data transformation. Seasonality feature implemented through Fourier components.
	
Components of the code - 
1. Option for data transformation 
2. Selection for type of seasonality(limitation only one type of seasonality at a time) 
3. SVM Hyper-parameter and auto-regressive period optimization 
4. One-step ahead forecast.
