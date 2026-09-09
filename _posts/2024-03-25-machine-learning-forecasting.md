---
layout: post
title: "Machine Learning Forecasting of Time Series"
author: ruben
description: "Discover how to implement machine learning forecasting of time series data with Python, by using recursive and direct forecasting."
excerpt: "Discover how to implement machine learning forecasting of time series data with Python, by using recursive and direct forecasting."
categories: [Time Series Forecasting]
image: assets/images/posts/machine-learning-forecasting/machine-learning-forecasting.png
---

Time series forecasting is an important topic in data science, given its widespread application across various industries. From healthcare to finance, many sectors rely on time series forecasting to inform strategic decisions and drive their businesses forward.

If you’re keen to explore how machine learning forecasting can be leveraged for predicting future values of time series data, then you’re in the right place.

In this article, we’ll begin by discussing different types of time series data. Following that, we’ll provide an overview of available methods for conducting time series forecasting. Finally, we’ll learn the concept of time series forecasting with machine learning, complete with example code.

But before we delve into machine learning forecasting, it’s crucial to understand the concept of forecasting first. So, without further ado, let’s start there!

## **What is Forecasting**

Forecasting involves predicting an unknown value. In the context of data science, we typically use the term ‘forecasting‘ when predicting unknown values within time series data.

Now the question is, what is time series data?

Time series data consists of a sequence of data points measured over successive time intervals and arranged chronologically. One notable characteristic of time series data is that each data point is usually associated with a specific timestamp, resembling something like this:

**![time series forecasting diagram]({{ site.baseurl }}/assets/images/posts/machine-learning-forecasting/image_1-1024x511.png)**

And you’ve likely encountered these types of data in your daily life, such as:

- Daily stock price fluctuations over a one-month period.
- Daily count of new COVID-19 cases within a week.
- Weekly revenue generated from online sales over the past six months.

When we forecast time series data, we predict the future value of a time-dependent variable. As an example, we might want to forecast the price of a stock three days from now or the number of new COVID-19 cases a week from now.

But now another question is: how do we forecast the future values of a time-dependent variable?

As with any data science use case, we need a collection of past data when we want to forecast a future value of a variable. We then use the past data to predict the value of a variable in the future. For example, to forecast tomorrow’s stock price, we need the historical data of stock price spanning several weeks, months, or even years prior to the forecasting point.

In today’s world, many profit and non-profit organizations from different sectors use time series forecasting to solve their business problems. Below are just a few examples:

#### **Finance and Investment**

Financial institutions utilize time series forecasting to predict stock prices, exchange rates, and interest rates. This leads to better investment decisions, risk management, and portfolio optimization.

#### **Supply Chain Management**

Companies rely on time series forecasting to optimize inventory management, production planning, and logistics operations. By forecasting demand patterns and lead times, companies can streamline their supply chains, minimize stock outs, and reduce inventory holding costs. This improves operational efficiency and customer satisfaction.

#### **Healthcare**

Healthcare providers use time series forecasting to anticipate patient admission rates, disease outbreaks, and resource utilization levels. By forecasting healthcare demand, hospitals and healthcare facilities can allocate resources efficiently, optimize staff scheduling, and improve patient care delivery. This improves healthcare outcomes and reduces costs.

Now that you know how relevant time series forecasting is, let’s go deeper into this topic!

## **Types of Time Series Data**

In general, time series data can be divided into two categories: univariate and multivariate time series data. Let’s dissect the difference between these two categories.

### **Univariate Time Series**

Univariate time series show only one time-dependent variable. That is, a univariate time series is a collection of data points recorded for one variable over successive intervals of time.

As an example, let’s say you want to track the daily weather temperature in your hometown over one year. The univariate time series data for this case would look something like this.

**![univariate time series example]({{ site.baseurl }}/assets/images/posts/machine-learning-forecasting/image_2-1024x511.png)**

Based on this time series, we could, for example, forecast tomorrow’s weather temperature.

Forecasting univariate time series data involves examining trends and seasonality of the variable. There are various methods or models that we can use to forecast univariate time series, and we will explore them in the next section.

### **Multivariate Time Series**

If univariate time series data consists of only one time-dependent variable, then multivariate time series data consists of several time-dependent variables. This means there are two or more variables recorded over successive intervals of time. These variables may or may not have some dependency on each other, and they’re usually recorded at the same time intervals.

To understand it better, let’s revisit the use case from the previous univariate time series example. Instead of just tracking temperature, now you have other variables such as humidity and wind speed, each recorded daily.

**![multivariate time series example]({{ site.baseurl }}/assets/images/posts/machine-learning-forecasting/image_3-1024x511.png)**

Now, instead of only using the temperature variable, we can also utilize humidity and wind speed variables as additional information (also called predictors) to forecast tomorrow’s temperature.

One important thing to keep in mind is that we need to be careful when using time-dependent variables as predictors for forecasting future values. This is because you may not always have information about their values at the time of forecasting.

As example, if you want to forecast tomorrow’s weather temperature and you use humidity as one of the predictors, you may not have information about tomorrow’s humidity at the time you need to make the forecast.

Therefore, as a rule of thumb, always ask yourself this before using a time-dependent variable as one of the predictors: *“Will I have the information regarding the value of this predictor by the time I need to make the forecast?”*

## **Input data for forecasting**

To forecast any time series, there are typically two types of input variables: variables that we derive from the time series itself, and variables that we can obtain from additional data sources. Let’s break it down:

#### **Endogenous data**

Endogenous data are variables that are directly related to the variable we want to forecast. In the previous weather temperature example, endogenous data refers to the daily temperature and the many features that we can extract from it.

There are various methods or models we can use for time series forecasting, and we’ll also delve into these models in detail in the next section. Some of these methods extract the features automatically. For machine learning forecasting, we need to create the features ourselves.

#### **Exogenous data**

Exogenous data are variables that are not directly related to the variable we want to forecast. In the weather temperature example, exogenous variables include humidity and wind speed. It could also be factors like the location of each observation, days of the week, whether a day is a holiday, etc.

## **Overview of Forecasting Models**

There are several time series models that we can use for forecasting, which can generally be categorized into three different categories:

- **Classical models**, such as ARIMA, SARIMA, Moving Averages, Exponential Smoothing, or Vector Autoregression.
- **Machine Learning models**, such as Linear Regression, Random Forest, and Gradient Boosting algorithms, like XGBoost and LightGBM, or any other regression algorithm.
- **Deep Learning models**, such as RNN, LSTM, or Transformers-based models.

**![time series forecasting models]({{ site.baseurl }}/assets/images/posts/machine-learning-forecasting/image_4-1024x701.png)**

Let’s now take a closer look at each category of these forecasting models.

### **Classical Forecasting Models**

The models under this category are referred to as classical models because they use traditional statistical methods to forecast time-series data. These models are simple yet very effective in capturing simple patterns or trends from our time-series data. Here are some of the most well-known models that fall under this category:

#### **Moving Averages**

It computes the average of a fixed number of past observations to forecast future values. It’s a good method to smooth out random fluctuations and highlight underlying trends in the data.

#### **Exponential Smoothing**

ES predicts future values of the time series based on averages of past values. But ot assigns a particular weight to each past observation, which exponentially decreases from newest to oldest observations. The newer the observation, the higher the weight will be, and vice versa. In other words, the closer the observation is to the point we want to forecast, the strongest its influence in the prediction. ES is great for capturing short-term trends and irregular patterns in time series data.

#### **Autoregressive Models (AR)**

AR models forecasts future values of the time series based on a linear combination of its past values. An AR model that uses the past p values is referred to as AR(p).

#### **Autoregressive Integrated Moving Average (ARIMA)**

It combines AR, differencing, and MA concepts to capture trends in a time series. Differencing consists in computing the differences between consecutive observations of the time series. These makes ARIMA capable of forecasting more complext trends and patters in time series data.

#### **Seasonal ARIMA (SARIMA)**

It is the extension of ARIMA models to include seasonal components. This allows the model to capture the recurring seasonal patterns of the data.

### **Characteristics of classical forecasting models.**

Classical forecasting methods have been around since 1920, and they are great solutions for forecasting simple time series, or those with clear seasonalities and trends. And they also show some limitations that may make them not suitable for more complex time series data.

#### **Limited capacity to capture complex patterns**

Classical models may struggle to model nonlinear trends, seasonality, or irregular patterns in the data effectively.

#### **Lack of ability to handle high-dimensional data**

Classical models might not perform optimally when dealing with high-dimensional data, where we have lots of variables that we want to use as predictors.

#### **Limited ability to include exogenous data**

Classical models often focus solely on historical time-series data and most of them can’t handle additional exogenous data that could improve forecasting accuracy.

#### **Assumption of linearity**

Many classical models such as autoregressive (AR) and moving average (MA) models are based on the assumption of linear relationships between a time series and its past values. This assumption may not hold true for all types of data, particularly when dealing with nonlinear or complex relationships, leading to inaccurate predictions.

So, what can we do when we have complex time-series data that classical models most likely won’t be able to handle? This is where machine learning models come into play.

### **Machine Learning Models**

Different from classical models, machine learning models use advanced artificial intelligence algorithms to learn patterns and relationships directly from data. These properties make them useful in cases where we have high-dimensional data with complex patterns. Here are some of the most well-known machine learning forecasting models:

#### **Linear Regression**

Linear regression models the relationship between the dependent variable and one or more predictors by fitting a linear equation to the observed data. It’s simple and interpretable but assumes a linear relationship, which may not capture complex patterns.

#### **Random Forest**

It’s an ensemble learning method based on a collection of decision trees. Random forest is very effective for time-series forecasting tasks with nonlinear relationships and high-dimensional data.

#### **Gradient Boosting Machine (GBM)**

Similar to random forest, it’s also an ensemble learning method. It combines multiple weak models (which can be a decision tree or a linear regression) to create a strong model. They are effective for time series forecasting tasks with nonlinear relationships and complex patterns, offering high predictive accuracy.

### **Advantages of machine learning forecasting**

Machine learning models are better options than classical forecasting models when handling complex time-series data. There are several advantages to using machine learning models for time-series forecasting:

#### **Interpretability**

Normally, we use regression or tree based models in time-series forecasting. This means that the results obtained from our machine learning model can be interpreted or explained. This aspect is important for understanding model predictions and learning more about our data.

#### **Scalability**

Machine learning models can handle large volumes of data. This means that they can be used even when there are a lot of predictors and/or observations. They are scalable to big data environments and can process massive datasets.

#### **Use of exogenous data**

Machine learning models can easily incorporate exogenous data into the forecasting process. This allows them to use additional information from external sources to improve forecasting accuracy.

#### **Ability to capture complex patterns**

Machine learning models can capture complex, nonlinear relationships in the data, which may be difficult for classical models to handle.

#### **Ensemble methods**

We can use several machine learning models and ensemble them into one model to improve forecasting accuracy. This is a good method to exploit the strengths and mitigate weaknesses of each individual model, resulting in more robust and accurate predictions.

### **Deep Learning Models**

Deep learning models gained popularity in the time-series domain over the past few years, especially since the inception of Transformers. If we have a huge dataset with complex patterns, neural networks might provide more accurate forecast compared to standard ML models.

Recurrent Neural networks (RNN) and Long Short-Term Memory (LSTM) are the most common deep learning models used for time-series forecasting. This is due to their architecture, which is specially designed to capture long-term dependencies in sequential data.

However, using deep learning models for time-series tasks comes with several challenges:

#### **Interpretability**

Deep learning models are highly complex, which means that they’re often treated as black boxes. This makes it difficult to interpret their predictions, and interpretability is crucial in time-series forecasting tasks.

#### **Computational resources**

While deep learning models may offer slightly better performance compared to machine learning models, they are computationally intensive. Training a deep learning model may require specialized hardware such as GPUs or TPUs.

#### **Data requirements**

Deep learning models typically require a large amount of data to train effectively. This can be a limitation for time-series forecasting tasks, especially when data is limited or difficult to obtain.

Therefore, in this article, we’ll focus on the application of time-series forecasting with machine learning methods instead.

## **Machine Learning Forecasting**

Data preprocessing is the first step in using any machine learning model. And this is not an exception in time-series forecasting.

Machine learning models expect the data to be structured in tabular form: each column represents a predictor variable, and each row represents an observation. To put this into example, let’s consider our daily weather temperature data once again:

**![univariate time series data]({{ site.baseurl }}/assets/images/posts/machine-learning-forecasting/image_5-1024x511.png)**

To ensure our machine learning model can capture complex patterns and trends in our time-series, we need to create a table of predictors first. One common approach is to introduce lag features into our data.

### **Feature Engineering with Lag Features**

Lag features transform our time-series data into tabular format for supervised learning. The concept is simple: we take a value of a variable at the previous time step, and then include it as a feature. This means that we shift the data forward by a certain number of time steps; and we refer the number of time steps as the lag value. As an example, if we set the lag values to be 1, 2, and 3, our daily weather temperature data will look like this:

![lag features for machine learning forecasting]({{ site.baseurl }}/assets/images/posts/machine-learning-forecasting/image_6-1024x511.png)

In this context, t-3 represents the weather temperature 3 days before the current temperature, t-2 represents the temperature 2 days before, and so on so forth.

This data transformation is crucial for our machine learning models to capture the patterns and trends in our time-series data. The number of lags doesn’t have to be 3; it’s a hyperparameter that we need to tune based on the data complexity.

### **Feature Engineering with Window Features**

In addition to lag features, we can transform our time series data into tabular format by including summary statistics of the observation values from previous time steps. Using a sliding window, we can include statistics as features for our machine learning model. The rolling mean, which calculates the mean of the previous few observation values, is a popular choice.

Let’s use our weather temperature data as an example and say we set the sliding window to 3. This means we predict the temperature at the current time step using the mean temperature of the last 3 days:

**![window features for machine learning forecasting]({{ site.baseurl }}/assets/images/posts/machine-learning-forecasting/image_7-1024x511.png)**

These features capture local trends, fluctuations, and overall behavior in the time series data. Thus, they can be particularly useful for noisy data or non-stationary time series.

### **Feature Engineering with Exogenous Features**

An advantage of using machine learning models is the ability to incorporate exogenous data to enhance the forecasting performance. This includes features with static values and features with known or unknown future values.

For instance, suppose we want to add exogenous data to our weather temperature data. To improve model performance, we add three features: location, humidity, and day of the week:

**![exogenous data for machine learning forecasting]({{ site.baseurl }}/assets/images/posts/machine-learning-forecasting/image_8-1024x511.png)**

In this example:

- Location is a static feature, as it has identical values across all observations.
- Humidity is a feature with unknown future values, as we don’t know its value at the time of predicting temperature.
- Day of the week is a feature with known future values, as we know its value when predicting temperature for a particular day.

If you want to learn more about these and other methods to extract and create features for time series forecasting, check out our course [Feature Engineering for Time Series Forecasting](https://www.trainindata.com/p/feature-engineering-for-forecasting).

[![Feature engineering for time series forecasting course]({{ site.baseurl }}/assets/images/posts/time-series-forecasting-python/feature-engineering-forecasting-course-1024x576.png)](https://www.trainindata.com/p/feature-engineering-for-forecasting)

## **Machine Learning Forecasting**

Once we prepared the data, we can now train our machine learning model to predict the weather temperature of a particular day.

In time-series forecasting, we want the machine learning model to predict not only the next value of the observation (e.g., tomorrow’s temperature) but also the entire future interval (e.g., the temperature of three consecutive days from today).

However, a machine learning model can only output one prediction at a time. In our example, this means that our machine learning model can forecast only tomorrow’s temperature or the temperature 10 days from now, not a sequence of temperatures for the next three consecutive days.

To address this, two different methods are commonly applied in machine learning forecasting: recursive and direct forecasting.

### **Recursive Forecasting**

As the name suggests, in recursive forecasting, we recursively predict future values with our machine learning model. The model initially predicts one time step ahead (t+1) and then uses that prediction as an input to forecast the value in the next time step (t+2). This process is repeated recursively for n steps (t+n), where n is the step value that we defined in advance.

**![recursive forecasting]({{ site.baseurl }}/assets/images/posts/machine-learning-forecasting/image_9-1024x511.png)**

With this method, we use one machine learning model to predict the future values from t+1 up to a far point in time t+n.

### **Direct Forecasting**

To understand the concept of direct forecasting, we first need to address the concept of a “horizon”. A horizon refers to the number of time steps into the future for which predictions are made.

For example, if we are analyzing daily weather temperature data and want to predict the temperature for the next three consecutive days, then the horizon for our forecasting task is three days.

**![Direct forecasting]({{ site.baseurl }}/assets/images/posts/machine-learning-forecasting/image_10-1024x511.png)**

In direct forecasting, we train several machine learning models separately according to the horizon of our forecasting task. If the horizon is three, then we need to train three different machine learning models, each trained with a different horizon ranging from one to three.

After all models have been trained, we use each model to predict one of the steps in the horizon. For example, the model trained with horizon step one will predict the value one time step ahead; the model trained with the second step in the horizon will predict the value two time steps ahead, and so on.

For more details about direct and recursive machine learning forecasting, check out our course [Forecasting with Machine Learning](https://www.trainindata.com/p/forecasting-with-machine-learning).

[![]({{ site.baseurl }}/assets/images/posts/time-series-forecasting-python/forecasting-with-machine-learning-1024x576.png)](https://www.trainindata.com/p/forecasting-with-machine-learning)

## **Machine Learning Forecasting with Python**

In this section, we’ll implement both recursive and direct forecasting with Python. To achieve this, we’ll use a combination of these highly useful libraries:

- **Scikit-learn**: A Python library that covers a wide range of machine learning algorithms and concepts, covering both supervised and unsupervised learning.
- [**Skforecast**](https://skforecast.org): A Python library that wraps several machine learning models from scikit-learn, transforming them into easy-to-use models suitable for time-series forecasting purposes.
- [**Feature-engine**](https://feature-engine.trainindata.com): A Python library designed to create and select features for wide range of machine learning models, including ML forecasting models.

Now, let’s import all the necessary libraries.

```

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import Lasso
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline as ScikitPipeline

from skforecast.recursive import ForecasterRecursive

from feature_engine.timeseries.forecasting import (
    LagFeatures,
    WindowFeatures,
)

from feature_engine.imputation import DropMissingData
from feature_engine.datetime import DatetimeFeatures
from feature_engine.pipeline import Pipeline
```

Before we delve deeper into the implementation, let’s examine the dataset we’ll be working with.

We’ll use an electricity demand dataset, which you can obtain [here](https://github.com/tidyverts/tsibbledata/tree/master/data-raw/vic_elec/VIC2015). This dataset consists of the following variables:

- `demand`: Electricity demand in Victoria, Australia, recorded at 30-minute intervals over a 12-year period, from 2002 to early 2015.
- `temperature`: Temperature data for Melbourne, also recorded at 30-minute intervals.
- `holiday`: An indicator denoting whether a day is a public holiday.

Since our objective is to build machine learning models to forecast electricity demand, we’ll focus solely on the `demand` variable from the dataset. Additionally, we’ll perform feature engineering as needed using the [Feature-engine](https://feature-engine.trainindata.com) library.

Let’s proceed by loading the data and resampling it to hourly intervals instead of 30 minutes.

```

url = "https://raw.githubusercontent.com/tidyverts/tsibbledata/master/data-raw/vic_elec/VIC2015/demand.csv"
df = pd.read_csv(url)

df.drop(columns=["Industrial"], inplace=True)

# Convert the integer Date to an actual date with datetime type
df["date"] = df["Date"].apply(
    lambda x: pd.Timestamp("1899-12-30") + pd.Timedelta(x, unit="days")
)

# Create a timestamp from the integer Period representing 30 minute intervals
df["date_time"] = df["date"] + \
    pd.to_timedelta((df["Period"] - 1) * 30, unit="m")

df.dropna(inplace=True)

# Rename columns
df = df[["date_time", "OperationalLessIndustrial"]]

df.columns = ["date_time", "demand"]

# Resample to hourly
df = (
    df.set_index("date_time")
    .resample("H")
    .agg({"demand": "sum"})
)

df.head()
```

**![Electricity demand data set that we want to forecast with machine learning models]({{ site.baseurl }}/assets/images/posts/machine-learning-forecasting/image_11-1024x212.png)**

And now we’re ready to implement machine learning for a time series forecasting task with recursive and direct forecasting.

### **Recursive Machine Learning Forecasting with Python**

Before we train any predictive model, we need to split the data into training and test subset. For this purpose, we’ll use the data up until December 31, 2014 as the training data, and leave the rest for the test set.

```

end_train = '2014-12-31 23:59:59'
X_train = df.loc[:end_train]
X_test  = df.loc[end_train:]
```

If we take the last 500 observations from the training data and the first 500 observations from the test data, our time series data looks like the following:

**![electricity demand time series data set]({{ site.baseurl }}/assets/images/posts/machine-learning-forecasting/image_12-1024x545.png)**

An advantage of using machine learning for time series forecasting is that we can enhance its performance by adding exogenous data into the model.

In the following example, we’ll add three features with known future values to our data: month, day of the week, and hour of each observation.

We can achieve this using `DatetimeFeatures` class from the Feature-engine library. It expects a datetime column or index, and we can specify the datetime features that we want to extract.

```

datetime_f = DatetimeFeatures(
    features_to_extract = ["month", "day_of_week", "hour"],
    drop_original=True,
)

# the input to the datetime features

datetime_df = pd.DataFrame(
    X_train.index,
    index=X_train.index,
)

datetime_f.fit_transform(datetime_df)
```

**![input features to forecast energy demand with machine learning]({{ site.baseurl }}/assets/images/posts/machine-learning-forecasting/image_13-1024x212.png)**

Now that we have exogenous data consisting of the month, day of the week, and the hour of each observation, let’s proceed to build our machine learning model. In the following example, we’ll use a Lasso regression model, which we can easily implement using the scikit-learn library.

```
model = ScikitPipeline([
    ("scaler", MinMaxScaler()),
    ("lasso", Lasso(random_state=9, alpha=10))
])

```

At this point, we’ve built our model, and we already have the exogenous data that can enhance its performance during training. However, what we haven’t done is transform our electricity demand data into lag features.

Transforming our data into lag features can be time-consuming and complex, but we can abstract the underlying process using the `ForecasterAutoReg` class from [Skforecast](https://skforecast.org). All we need to do is specify the number of lags that we want to create.

In the following example, we will extract lag features from the previous 1, 24, and 144 hours of each observation.

```
forecaster = ForecasterRecursive(
    estimator=model,              # the machine learning model
    lags=[1, 24, 6*24],           # the lag features to create
    transformer_exog=datetime_f,  # to get the datetime features
    forecaster_id="recursive"
)

X, y = forecaster.create_train_X_y(
    y=X_train["demand"],
    exog=datetime_df,
)

X, y
```

**![predictor table and target variables for direct forecasting with machine learning models]({{ site.baseurl }}/assets/images/posts/machine-learning-forecasting/image_14-1024x355.png)**

Now that we have features in the correct format to train our regression model, let’s proceed to train the model.

```
forecaster.fit(
    y=X_train["demand"],  # the series for the lags
    exog=datetime_df,    # the datetime for the datetime features
)
```

With the model trained, let’s use it to predict future values.

Let’s say we want to predict the electricity demand for the first 24 hours of our test data. We can achieve this simply by calling the `predict` method as follows:

```

datetime_df_test = pd.DataFrame(
    X_test.head(24).index,
    index=X_test.head(24).index,
)

predictions = forecaster.predict(
    steps=24,
    exog=datetime_df_test,
)
```

If we plot the predictions over the actuals, we’ll see the following:

**![forecasts made with machine learning models]({{ site.baseurl }}/assets/images/posts/machine-learning-forecasting/image_15-1024x412.png)**As you can see, with the recursive forecasting method, we only need to train one machine learning model. The model initially predicts one time step ahead and then uses that prediction as an input to forecast the value in the next time step. The model does this recursively until it reaches the predefined steps.

If you’d like to compute the errors between the predicted values and the ground-truth values, then you can use common metrics applied in any regression problems, such as Mean Squared Error (MSE) or Root Mean Squared Error (RMSE).

### **Direct Machine Learning Forecasting with Python**

The direct forecasting method involves training one or more machine learning models, depending on the forecasting horizon. In the following example, we’ll use a 24-hour forecasting horizon, which means that we’ll train 24 distinct models.

Firstly, we must generate the labels or ground-truth values for each model. As an example, the ground-truth value for the 10th model is the electricity demand 10 hours after the actual observation. These ground-truth columns can be created using Pandas.

```
# The forecasting horizon.
horizon = 24

# Create an empty dataframe for the targets.
y = pd.DataFrame(index=df.index)

# Add each one of the steps ahead.
for h in range(horizon):
    y[f"h_{h}"] = df.shift(periods=-h, freq="H")

# Remove nan
y.dropna(inplace=True)
```

And below is the visualization of what the labels look like for the training process. For simplicity, we only show you the first three and the last three labels of the forecasting horizon.

**![table with predictors and target for direct forecasting]({{ site.baseurl }}/assets/images/posts/machine-learning-forecasting/image_16-1024x374.png)**

Next, let’s split the data into train and test set. As a note, we need to adjust one small thing in our test data. We need to take into account the last few observations from the training data in order to accomodate the lag features later on.

```
# align data to available target values

df = df.loc[y.index]

# We leave 2015 in the test set

end_train = '2014-12-31 23:59:59'
X_train = df.loc[:end_train]
y_train = y.loc[:end_train]

# add the last 6 hours of observations from training set
# into the test set to accomodate lag features later on

begin_test = '2014-12-31 17:59:59'
X_test  = df.loc[begin_test:]
y_test = y.loc[begin_test:]
```

Now, we can transform our electricity demand data into a proper tabular format with two types of features: lag features and window features.

In the following example, we’ll set the range of lag values from 1 to 6, resulting in a total of 6 lag values. Meanwhile, for window features, we aim to compute the mean and standard deviation of the observations from the previous 3 and 24 hours. Both features can be generated simply using [`LagFeatures`](https://feature-engine.trainindata.com/en/latest/api_doc/timeseries/forecasting/LagFeatures.html) and [`WindowFeatures`](https://feature-engine.trainindata.com/en/latest/api_doc/timeseries/forecasting/WindowFeatures.html) from the Feature-engine library.

As you might have noticed already, the application of both lag features and window features will introduce missing values. Therefore, we also need to drop the rows with missing values.

Finally, we encapsulate everything easily in a [`Pipeline`](https://feature-engine.trainindata.com/en/latest/api_doc/pipeline/Pipeline.html) class from Feature-engine.

```
# Lag features.
lagf = LagFeatures(
    variables=["demand"],  # the input variables
    periods=[1, 2, 3, 4, 5, 6],  # 6 lags
    missing_values="ignore",
)

# window features
winf = WindowFeatures(
    variables=["demand"],  # the input variables
    window=["3H", "24H"],  # average of 3 and 24 previous hours
    freq="1H",  # move 1 hr forward
    functions=["mean", "std"],
    missing_values="ignore",
    drop_original=True,    # drop demand when i don't need it any more
)

# Drop missing data
dropna = DropMissingData()

# transformation pipeline
pipe = Pipeline(
    [
        ("lagf", lagf),
        ("winf", winf),
        ("dropna", dropna),
    ]
).set_output(transform="pandas")

pipe.fit(X_train, y_train)
```

Below is the visualization of the final data that we’ll shortly use to train our regression models:

**![]({{ site.baseurl }}/assets/images/posts/machine-learning-forecasting/image_17-1024x344.png)**

Now, it’s time for us to build the model. The model will be similar to the one used in the recursive method, which is Lasso regression.

As mentioned earlier, in direct forecasting, we need to train one or more models separately depending on the forecasting horizon. Since the forecasting horizon in our example is 24, we need to build and train 24 different models.

Now the question is: how can we instantiate 24 different Lasso regression models?

We can achieve this by using the `MultiOutputRegressor` class from scikit-learn. This class simplifies the process of fitting multiple regressors depending on the number of target variables.

```

lasso = MultiOutputRegressor(Lasso(random_state=0, max_iter=10))
```

In fact, we can integrate our Lasso models into the Feature-engine pipeline by adding them into the end of the pipeline. Then, by simply calling the `fit` method, the whole pipeline can be executed; starting from the generation of lag features and window features until the training process of 24 distinct Lasso models.

```
pipe = Pipeline(
    [
        ("lagf", lagf),
        ("winf", winf),
        ("dropna", dropna),
        ("lasso", lasso),
    ]
)

# this step trains 24 lasso models
pipe.fit(X_train, y_train)
```

The forecasting of future values with trained model can be done by as simple as calling `predict` method from Feature-engine pipeline.

```

forecast = pipe.predict(X_test)

forecasts = pd.DataFrame(
    pipe.predict(X_test),
    index=X_test.loc[end_train:].index,
    columns=[f"step_{i+1}" for i in range(24)]

)
```

**![]({{ site.baseurl }}/assets/images/posts/machine-learning-forecasting/image_18-1024x405.png)**

In the visualization above, we only display the first 5 and the last 5 forecasting horizons for simplicity.

Using the `predict` method, we obtain the next 24 hours of electricity demand forecast for each observation. The forecasted value for each hour originates from a specific Lasso regression model.

Next, we can visualize the next 24-hour forecasted values of the first observation in the test data alongside the corresponding ground-truth values, as follows:

```

f = forecasts.iloc[0]
f.index = X_test.loc[end_train:].head(24).index

fig, ax = plt.subplots(figsize=(10, 4))
X_test.loc[end_train:].head(24).plot(ax=ax, label='train')
f.plot(ax=ax, label='predictions')
plt.title("Lasso forecasting")
ax.legend(bbox_to_anchor=(1.3, 1.0));
```

**![]({{ site.baseurl }}/assets/images/posts/machine-learning-forecasting/image_19-1024x424.png)**

And we have done it! Direct forecasting is computationally more expensive than the recursive method since it requires training of several models. However, in certain use cases, it achieves better results compared to the recursive forecasting method.

If you’d like to learn more about time series forecasting in a deeper level and working on more practical hands-on tutorials, then we have got you covered: we have launched a [Forecasting specialization](https://www.trainindata.com/p/forecasting-specialization) on Train in Data!

[![Forecasting specialization]({{ site.baseurl }}/assets/images/posts/time-series-forecasting-python/forecasting-specialization-1024x576.png)](https://www.trainindata.com/p/forecasting-specialization)

There you can enroll in all of our specialized forecasting courses in one go to make sure that you have a complete understanding of the whole time series forecasting workflow.

## **Conclusion**

In this article, we have learned how to utilize machine learning for forecasting tasks. Firstly, we explored the characteristics that define time series data, and then we delved into an overview of commonly applied methods for time series forecasting tasks. Finally, we explored how we can leverage machine learning for time series forecasting and implemented it in code.

I hope that this article is useful for you to mastering time series forecasting with machine learning. Time series forecasting is widely used in every business sector. Therefore, regardless of whether you are a data scientist, analyst, business intelligence professional, or machine learning engineer, the ability to forecast future trends and patterns using machine learning is an invaluable **skill set** to possess.
