---
layout: post
title: "Time Series Forecasting with Python"
author: sole
description: "Find out how to implement time series forecasting in Python, from statistical models, to machine learning and deep learning."
excerpt: "Find out how to implement time series forecasting in Python, from statistical models, to machine learning and deep learning."
categories: [Time Series Forecasting]
image: assets/images/posts/time-series-forecasting-python/time-series-forecasting-python.png
---

In data science, predicting future values is a common task. To do that, we can implement time series forecasting models with Python.

Time series forecasting models are designed to predict future values of a time series dataset by analyzing historical data. These models include classical forecasting methods such as ARIMA and Exponential Smoothing (ETS), as well as machine learning approaches that utilize supervised learning algorithms to automatically detect patterns within the data.

![]({{ site.baseurl }}/assets/images/posts/time-series-forecasting-python/02-forecasting-overview-1024x576.png)

In this tutorial, we will explore both traditional forecasting models, such as ETS and ARIMA, and machine learning approaches to forecasting. We’ll discuss the workings of these widely adopted time series models and demonstrate how to utilize various Python libraries for time series forecasting.

Let’s get started!

> Become an all-rounded forecaster with our [forecasting specialization](https://www.trainindata.com/p/forecasting-specialization) at Train in Data.

## **Traditional Forecasting Models**

Traditional forecasting models, such as ARIMA and Exponential Smoothing (ETS), rely on statistical techniques to analyze historical data patterns and make predictions for future values in time series datasets. These models capture different components like trend, seasonality, and autoregression to forecast accurately.

Traditional or statistical forecasting models date back to the early 20th century, and have profoundly shaped the field of time series analysis.

Before moving on to the details about some of the most widely used traditional forecasting models, let’s review some time series characteristics. This will help us understand the ARIMA and ETS models we will cover later.

### **Time Series Characteristics**

**Stationarity**: A time series is considered stationary when its data points exhibit independence from time. Therefore, time series containing [trends or seasonal patterns](https://www.blog.trainindata.com/seasonal-time-series-forecasting/) are non-stationary.

**Differencing**: Differencing involves calculating the differences between consecutive points in a non-stationary time series to transform it into a stationary one.

### **Classical Forecasting Models**

There are various statistical models used in time series forecasting. Among these AR, MA, and ARIMA models are the most widely used. Let’s break them down.

#### **Autoregressive Models (AR)**

AR models forecasts future values of the time series based on a linear combination of its past values. An AR model that uses the past p values is referred to as AR(p).

#### **Moving Average Models (MA)**

Instead of using past data points in forecasting, this approach uses past forecast errors. For instance, a MA(2) model uses the error terms from the previous two data points.

#### **Autoregressive Moving Average (ARMA)**

ARMA is a linear combination of AR and MA models.

#### **Autoregressive integrated moving average (ARIMA)**

ARIMA models have an additional I(d) term, where d is the number of differencing operations used to make the time series stationary.

Note that ARIMA transforms the time series into a stationary one, thereby eliminating seasonal effects. The following 2 models also incorporate seasonal and external parameters into our methods.

#### **SARIMA**

This predictive modeling approach usestwo distinct ARIMA models: one for the seasonal component and another for the non-seasonal component of the data.

#### **SARIMAX**

This approach complements SARIMA in that it adds exogenous variables to the time series data, thus achieving a more inclusive time series analysis.

#### **Exponential Time Smoothing (ETS)**

We utilize historical data points to predict future values in a time series. Yet, it’s often more reasonable to assign greater importance to recent observations compared to those from the dataset‘s inception. This concept underlies simple exponential smoothing. Here, forecasts are generated using weighted averages, where the weights exponentially amplify as observations draw closer to the present.

Just like ARIMA models, simple exponential smoothing can also accommodate seasonality and trends. These additive models feature separate smoothing parameters for the level, trend, and seasonal components

### **Implementing statistical forecasting models with Python**

Let’s implement the ARIMA and SARIMAX models in Python using statsmodels. We’ll use the airline passenger dataset for this demo.

Let’s import the libraries:

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
```

Let’s load the data and split it into a training data set and a test set:

```
url = "https://raw.githubusercontent.com/facebook/prophet/main/examples/example_air_passengers.csv"
data = pd.read_csv(url, header=0, parse_dates=[0], index_col=0,)
train = data.iloc[:-12]
test = data.iloc[-12:]
```

Let’s fit an ARIMA model:

```
arima_model = ARIMA(train, order=(5,1,0))
arima_result = arima_model.fit()
```

Now we forecasts the next 12 months of airline passengers:

```
arima_forecast = arima_result.forecast(steps=12)
```

```

```

Let’s fit a SARIMA model and make a forecast for the same time period: the future 12 months:

```
sarimax_model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12))
sarimax_result = sarimax_model.fit()
sarimax_forecast = sarimax_result.forecast(steps=12)

```

Now, let’s plot the forecasts of the 2 models against the actuals for visualization:

```
plt.figure(figsize=(10, 5))
plt.plot(train, label='Train')
plt.plot(test, label='Test')
plt.plot(arima_forecast, label='ARIMA Forecast')
plt.plot(sarimax_forecast, label='SARIMAX Forecast')
plt.legend()
plt.title('ARIMA and SARIMAX Forecasting')
plt.show()
```

We see that SARIMAX (red) is a better model for this forecasting tasks:

![time series forecasting with python]({{ site.baseurl }}/assets/images/posts/time-series-forecasting-python/forecasting-with-python.png)

Let’s now calculate MAE and RMSE metrics to determine both statistical models’ performance:

```
#MAE
arima_mae = mean_absolute_error(test, arima_forecast)
sarimax_mae = mean_absolute_error(test, sarimax_forecast)

#RMSE
arima_mse = mean_squared_error(test, arima_forecast)
arima_rmse = np.sqrt(arima_mse)
sarimax_mse = mean_squared_error(test, sarimax_forecast)
sarimax_rmse = np.sqrt(sarimax_mse)

print(f"ARIMA MAE: {arima_mae:.2f}, SARIMAX MAE: {sarimax_mae:.2f}" )
print(f"ARIMA RMSE: {arima_rmse:.2f}, SARIMAX RMSE: {sarimax_rmse:.2f}")

```

From the metrics values, we see that SARIMAX achieves overall better scores on the test data:

```
ARIMA MAE: 67.39, SARIMAX MAE: 16.32
ARIMA RMSE: 86.81, SARIMAX RMSE: 21.19

```

## **Machine Learning Models**

The traditional statistical models we’ve covered thus far might yield accurate forecasts for linear trends, but their performance may decline when dealing with non-linear patterns and interactions between multiple variables.

In contrast, [machine learning models](https://www.blog.trainindata.com/algorithms-of-machine-learning/) excel at adapting to diverse and complex scenarios and datasets through their adjustable hyperparameters and capacity to learn from data. Let’s take a moment to explore three straightforward machine learning algorithms.

### **Linear Regression**

Linear regression is a predictive modeling method used to model the relationship between a dependent variable and one or more independent variables by fitting a linear equation to observed data points. It assumes that there is a linear relationship between the variables and aims to find the best-fitting line that minimizes the sum of squared differences between the observed and predicted values.

### **Ensemble Learning**

A technique where multiple models, often referred to as ‘weak learners’, are strategically combined to improve prediction accuracy and robustness over individual models. This approach prioritizes the strength of a collective system to achieve better performance and usually reduces variance and bias. Random Forests are the classical example of ensemble learning. But we can combine any model in principle.

### **Random Forest**

An ensemble method, creates a ‘forest’ of decision trees, each trained on a random subset of the data and features. By averaging the predictions of these diverse trees, Random Forest enhances predictive accuracy and controls overfitting, making it effective for a wide range of data types and problems.

### **Support Vector Regression**

It works by finding the best-fit line (or curve for more complex data) through the data points. SVR is special because it focuses not just on getting close to each data point but also on maintaining a balanced fit overall. This makes it really good at dealing with complicated data where you don’t want a few unusual points to throw off your predictions. It’s particularly strong when your data doesn’t follow a simple pattern and is great at handling data with many features.

## **Forecasting with machine learning**

To forecast with traditional machine learning models, we need to create a table of features and a target variable, from our time series.

We can create features by lagging past values of the time series; these are called lag features. We can also apply functions over past windows of data; the so called window features.

We can also enrich the dataset with external data sources.

> Master feature extraction to forecast with traditional machine learning models with our course [Feature Engineering for Time Series Forecasting](https://www.trainindata.com/p/feature-engineering-for-forecasting).

[![Feature engineering for time series forecasting course]({{ site.baseurl }}/assets/images/posts/time-series-forecasting-python/feature-engineering-forecasting-course-1024x576.png)](https://www.trainindata.com/p/feature-engineering-for-forecasting)

### **Forecasting multiple steps**

Traditional machine learning models produce single outputs. Yet, when we forecast, we normally want to predict multiple steps in the future. Hence, to perform multistep forecasting with traditional machine learning, we need to implement multistep forecasting strategies. The most commonly used are recursive forecasting and direct forecasting.

Recursive forecasting involves generating predictions for future time points iteratively, where each prediction is based on previously forecasted values. In recursive forecasting, we train one machine learning model, to predict the entire horizon.

In direct forecasting, each step of the forecasting horizon is predicted independently. Hence, in direct forecasting, we train several models, one for each step of the horizon.

> Master the use of machine learning models to forecast one or multiple time series with our course [Forecasting with Machine Learning.](https://www.trainindata.com/p/forecasting-with-machine-learning)

[![]({{ site.baseurl }}/assets/images/posts/time-series-forecasting-python/forecasting-with-machine-learning-1024x576.png)](https://www.trainindata.com/p/forecasting-with-machine-learning)

### **Forecasting with machine learning models in Python**

In this demo, we will implement recursive forecasting to forecast a univariate time series using the Python library [skforecast](https://skforecast.org).

Let’s start with the imports:

```
import pandas as pd
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
```

Let’s load the data and split it into a training data set and a test set:

```

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
data = pd.read_csv(url, header=0, parse_dates=[0], index_col=0)
data = data.squeeze("columns")
train = data[:-12]
test = data[-12:]
```

Let’s set up 3 forecasting models: one with lasso, one with random forests and one with support vector regression.

```
# Model 1: Linear Regression
forecaster_lr = ForecasterAutoreg(regressor=Lasso(random_state=10), lags=12)
forecaster_lr.fit(y=train)
predictions_lr = forecaster_lr.predict(steps=12)

# Model 2: Random Forest
forecaster_rf = ForecasterAutoreg(regressor=RandomForestRegressor(n_estimators=100, random_state=42), lags=12)
forecaster_rf.fit(y=train)
predictions_rf = forecaster_rf.predict(steps=12)

# Model 3: Support Vector Machine (SVR)
forecaster_svr = ForecasterAutoreg(regressor=SVR(kernel='linear'), lags=12)
forecaster_svr.fit(y=train)
predictions_svr = forecaster_svr.predict(steps=12)
```

Let’s compare the model performance:

```
error_rmse = mean_squared_error(
                y_true = test.head(12),
                y_pred = predictions_lr,
                squared=False,
            )

print(f"Lasso rmse: {error_rmse}")

error_rmse = mean_squared_error(
                y_true = test.head(12),
                y_pred = predictions_rf,
                squared=False,
            )

print(f"Random forests rmse: {error_rmse}")

error_rmse = mean_squared_error(
                y_true = test.head(12),
                y_pred = predictions_svr,
                squared=False,
            )

print(f"SVR rmse: {error_rmse}")
```

We see the performance of each model below:

```

Lasso rmse: 17.43917405292811
Random forests rmse: 38.452050443810315
SVR rmse: 18.44807192448605
```

In this case, Lasso returned the bests forecasts.

### LightGBM

Gradient Boosting Models (GBM) have also gained popularity in forecasting due to their ability to capture non-linear information and exogenous variables.

We will again use the [skforecast](https://skforecast.org) API to fetch a built in multivariate dataset and forecast using the LightGBM. In addition, we’re going to implement backtesting to evaluate the performance of our model.

Let’s being with the imports:

```
import numpy as np
import pandas as pd
from skforecast.datasets import fetch_dataset
from lightgbm import LGBMRegressor
from skforecast.model_selection import backtesting_forecaster
from skforecast.ForecasterAutoreg import ForecasterAutoreg
```

Now, we load the data:

```
data = fetch_dataset('bike_sharing', raw=True)
data = data[['date_time', 'users']]
data['date_time'] = pd.to_datetime(data['date_time'], format='%Y-%m-%d %H:%M:%S')
data = data.set_index('date_time')
data = data.asfreq('H')
data = data.sort_index()

```

Let’s split the data into training and test set:

```
end_train = '2012-03-31 23:59:00'
end_validation = '2012-08-31 23:59:00'
data_train = data.loc[: end_train, :]
data_val = data.loc[end_train:end_validation, :]
data_test = data.loc[end_validation:, :]
```

Next, we set up a recursive forecaster to forecast using lightGBM:

```
forecaster = ForecasterAutoreg(
    regressor = LGBMRegressor(random_state=15926, verbose=-1),
    lags = 24
)
```

Let’s implement backtesting to evaluate the performance of this model:

```
metric, predictions = backtesting_forecaster(
    forecaster = forecaster,
    y = data['users'],
    steps = 36,
    metric = 'mean_absolute_error',
    initial_train_size = len(data[:end_validation]),
    refit = False,
    n_jobs = 'auto',
    verbose = True,
    show_progress = True
)
```

Let’s print out the evaluation metric:

```
print(f'Backtest error (MAE): {metric}')
```

We see the error metric below:

```
Backtest error (MAE): 76.25868057062392

```

To master forecasting with traditional machine learning models, check out our [Forecasting specialization](https://www.trainindata.com/p/forecasting-specialization).

[![]({{ site.baseurl }}/assets/images/posts/time-series-forecasting-python/forecasting-specialization-1024x576.png)](https://www.trainindata.com/p/forecasting-specialization)

## Deep Learning

Deep learning leverages artificial neural networks (NN) that consist of layers of interconnected nodes, or ‘neurons’, which hierarchically process input data. This layered structure allows deep learning models to learn and extract high-level features from large volumes of data, a capability that sets them apart from traditional machine learning approaches.

### LSTM

Long Short-Term Memory (LSTM) networks are a type of recurrent neural network specially designed to overcome the challenge of learning long-term dependencies in sequence data. Traditional NNs struggle to retain information from earlier points in a sequence as they process more data, a problem known as vanishing gradients.

LSTM networks address this by incorporating memory cells that maintain information over longer periods, and gates that regulate the flow of information into and out of these cells. This structure makes LSTMs adapt to tasks like time series forecasting, language modeling, and text generation, where understanding the context and dependencies over extended sequences is crucial.

We will now build a pretty simple LSTM model to forecast the “users” column from the previous dataset. Note that we opted for Tensorflow although you can use other libraries like Prophet as well.

Let’s now jump into the next demo. We’ll begin with the imports:

```
from math import sqrt
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from skforecast.datasets import fetch_dataset
```

Let’s now load and prepare the data:

```
data = fetch_dataset('bike_sharing', raw=True)
dataset = data['users'].values.reshape(-1, 1)

# scale down the values between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# train-test split
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
```

Now we train an LSTM and evaluate its performance:

```
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

look_back = 1
# separate the ylabel
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=3, batch_size=1, verbose=2)

# get the final rmse after prediction
testPredict = model.predict(testX)
testScore = sqrt(mean_squared_error(testY, testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
```

Below we see the performance of the LSTM:

```
Test Score: 0.13 RMSE
```

## Conclusion

In this tutorial, we reviewed both traditional and machine learning approaches to time series forecasting problems. We learned the pros and cons of both methods and implemented various models using statsmodels, skforecast, and Tensorflow libraries. Time series forecasting is a crucial [skill any data scientist](https://www.blog.trainindata.com/data-science-skillset/) should have. Forecasting is used in many data science domains and now you are ready to dive into the lucrative word of time series!
