---
layout: post
title: "Feature scaling in machine learning: Standardization, MinMaxScaling and more..."
author: sole
description: Discover why and how we scale variables in Python for machine learning.
excerpt: Discover why and how we scale variables in Python for machine learning.
categories: [ Feature engineering, Python, Machine learning]
image: assets/images/posts/scaling/cover.gif
---

You’ve probably heard that feature scaling is a common data preprocessing step when training machine learning models. But 
why do we rescale features in our data science projects? Do we need to scale features for all machine learning algorithms? 
And which feature scaling methods should we use?

These are the questions that we are going to address throughout this article. Let’s get started!


## When should we rescale our variables?

Many machine learning algorithms are sensitive to the feature scale. You probably know that the coefficients of linear 
regression models vary with the scale of the numeric features; that is, changing the feature scale will change the coefficient’s 
value.

Let’s illustrate this with a simple example: we want to predict house price with the size of the room. If the price is 100.000 
dollars and the room size is 10 square meters, then the coefficient is 10.000: 10.000 x 10 = 100.000.

If we use a different metric for the room size, say square centimeters, now the room size is 1000 square centimeters, and 
hence, the coefficient will be 100: 100 x 1000 = 100.000.

In this simple example, you see how changing the feature scale, simply by taking a measurement on another scale, changes 
the value of the coefficient by one order of magnitude or more. As in linear and logistic regression, the coefficients are 
used to determine the importance of the feature towards the prediction; having the features on similar scales allows us to 
compare feature importance more fairly.

For support vector machines (SVMs) and neural networks, gradient descent shows faster convergence if the features have 
similar scales.

Machine learning algorithms based on distance calculations, like the Euclidean distance, are also sensitive to the feature 
scale. Among them are clustering, for example, with k-means, principal component analysis (PCA) and K-nearest neighbors (KNNs).

Features with bigger value ranges tend to dominate over features with smaller ranges. Therefore, having features on a similar 
scale allows us to get a better view of data points that are more similar to each other when using distance metrics.


![machine learning algorithms sensitive to feature magnitude]({{ site.baseurl }}/assets/images/posts/scaling/algos.png)   


In summary, having features on a similar scale lets us compare feature importance, allows us to train better models, and also 
helps algorithms converge faster, thus improving model performance and training times.

**Then, which models do not require feature scaling?**

The models that do not require feature scaling are the tree-based algorithms like, decision trees, random forests, and 
gradient boosting machines.


## What is feature scaling?

Feature scaling is the process of setting the variables on a similar scale. This is usually done using normalization, 
standardization, or scaling to the minimum and maximum values.

The goal is to have all variables with similar value ranges. The value range is the difference between the maximum and 
minimum values.

In general, scaling techniques divide the variables by some constant scaling factor. Therefore, most scaling methods do 
not change the shape of the variable distribution. If you want to change the distribution shape, you need to apply 
[variance stabilizing transformations](https://www.blog.trainindata.com/variance-stabilizing-transformations-in-machine-learning/).

In the rest of the article, we will describe the following feature scaling techniques and then show you how to implement 
them in Python:

- Standardization

- Scaling to the maximum and minimum values

- Scaling with the median and quantiles, otherwise known as robust scaling.


## Standardization

Standardization is the process of centering the variable at 0 (zero mean) and standardizing the variance to 1 (unit variance), 
and it is suitable for variables with a Gaussian distribution. After the standard scaling, the standard deviation will also 
be 1 (unit standard deviation). Remember that standard deviation = sqrt(variance), where sqrt is the square root.

To standardize features, we subtract the mean from each observation and then divide the result by the standard deviation:

X_scaled = (X - X_mean)/ std(X)

The result of this transformation is called the z-score which indicates how many standard deviations a given observation 
deviates from the mean. Hence, standardization is also called z-score normalization.

This transformation is well suited for variables with a normal distribution. If your variables are skewed, it is perhaps 
better to carry out MinMaxScaling.


### Standardization with Python and sklearn.

Let’s implement standardization with scikit-learn. To begin, we will import the required packages, load the dataset, and 
prepare the training and test sets:

```
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
```

Let’s load the California housing dataset into a dataframe containing the independent variables and a series containing the 
dependent variable:

```
X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X.drop(labels=["Latitude", "Longitude"], axis=1, inplace=True)
```

Now, let’s divide the dataset into a training set and a test set:

```
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)
```

Let’s display the first rows of the data:

```
X_train.head()
```

![california housing dataframe]({{ site.baseurl }}/assets/images/posts/scaling/dataframe.png)   

As we can see in the table above, the continuous variables have different scales. Hence, feature scaling is a necessary 
pre-processing step to predict house prices with a linear regression model.

Let’s set up a standard scaler with its default parameters and fit it to the training data set so that it learns each 
variable’s mean and standard deviation:

```
scaler = StandardScaler(with_mean=True, with_std=True)
scaler.fit(X_train)
```

Now, let’s standardize the training data and also the test set with the trained scaler:

```
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```


The `StandardScaler()` stores the mean and standard deviation learned from the training set. Let’s visualize the learned 
parameters:

```
scaler.mean_
```

```
array([3.86666741e+00, 2.86187016e+01, 5.42340368e+00,
1.09477484e+00,1.42515732e+03, 3.04051776e+00])
```

Now, let’s print the standard deviation values:

```
scaler.scale_
```

```
array([1.89109236e+00, 1.25962585e+01, 2.28754018e+00,
4.52736275e-01,1.14954037e+03, 6.86792905e+00])
```

Scikit-learn scalers, just like any scikit-learn transformer, return numpy arrays by default. If we want to return a dataframe, 
we need to use the set_output API. We would then set up the transformer as follows before fitting it to the training set:

```
scaler = StandardScaler().set_output(transform="pandas")
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Now we can compare the standardized data with the original, unscaled data to visualize the changes. Let’s plot histograms 
of the original distributions:

```
X_test.hist(bins=20, figsize=(20, 20))
plt.show()
```

![unscaled variables' distribution]({{ site.baseurl }}/assets/images/posts/scaling/raw-data.png)   

And now, let’s plot histograms of the scaled variables:

```
X_test_scaled.hist(bins=20, figsize=(20, 20))
plt.show()
```

![variable distribution after standard scaling]({{ site.baseurl }}/assets/images/posts/scaling/standard-scaled.png)   

We have the variables on a (more) similar scale. Note that the shape of the distribution before and after the transformation remained 
similar. Yet, most variables are skewed, so standard scaling was not the best choice for this dataset.


## Scaling to the maximum and minimum values

Scaling to the maximum and minimum feature values squeezes the values of the variables between 0 and 1.

To implement this scaling technique, we subtract the minimum value from all the observations and divide the result by the 
value range, that is, the difference between the maximum and minimum values:

X_scaled = (X - X_min) / (X_max - X_min)

Note that if the variable has negative values, it will also be squeezed between 0 and 1. Say if the minimum value is -10, 
then 10 - (-10) = -10 + 10 = 0.

Let’s now implement scaling to the minimum and maximum values using scikit-learn.


### MinMaxScaling with Python and sklearn.

To begin, we will import the required packages, load the dataset, and prepare the training and test sets:


```
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
```

Let’s load the California housing dataset from scikit-learn into a pandas dataframe:

```
X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X.drop(labels=["Latitude", "Longitude"], axis=1, inplace=True)
```

Let’s divide the data into training and test sets:

```
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)
```

Let’s set up the minmax scaler and then fit it to the train set so that it can compute each variable’s minimum and maximum 
values:

```
minmax = MinMaxScaler().set_output(transform="pandas")
minmax.fit(X_train)
```

Finally, let’s scale the variables in the train and test sets with the trained scaler:

```
X_train_scaled = minmax.transform(X_train)
X_test_scaled = minmax.transform(X_test)
```

`MinMaxScaler()` stores the maximum and minimum values and the value ranges in its `data_max_`, `min_`, and `data_range_` 
attributes, respectively.

Let’s now compare the distributions of the variables before and after rescaling. Let’s plot the unscaled variables first:

```
X_test.hist(bins=20, figsize=(20, 20))
plt.show()
```

![unscaled variables' distribution]({{ site.baseurl }}/assets/images/posts/scaling/raw-data.png)   

And now, let’s plot histograms of the scaled variables:

```
X_test_scaled.hist(bins=20, figsize=(20, 20))
plt.show()
```

![variable distribution after minmax scaling]({{ site.baseurl }}/assets/images/posts/scaling/minmax.png)   

The variables value range is 1. Note that the shape of the distribution before and after the transformation remained similar.


## Scaling with the median and quantiles

When scaling variables with the median and quantiles, the median value is removed from the observations, and the result 
is divided by the Inter-quartile range (IQR).

X_scaled = (X - X_median) / IQR

The IQR is the difference between the 1st quartile and the 3rd quartile.

This method is known as robust scaling because it produces more robust estimates for the center and value range of the 
variable if the data contains outliers or is highly skewed.


### Robust scaling with Python and sklearn:

Let’s scale variables with the median and IQR using scikit-learn. To begin, we will import the required packages, load the 
dataset, and prepare the train and test sets:


```
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
```

Let’s load the California housing dataset from scikit-learn into a pandas dataframe:

```
X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X.drop(labels=["Latitude", "Longitude"], axis=1, inplace=True))
```

Let’s divide the data into training and test sets:

```
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)
```

Let’s now set up the `RobustScaler()` from scikit-learn and fit it to the training set so that it learns and stores the 
median and IQR:

```
scaler = RobustScaler().set_output(transform="pandas")
scaler.fit(X_train)
```

Finally, let’s rescale the variables in the training and test sets with the trained scaler:

```
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

We can go ahead and display the variable median values stored by `RobustScaler()`:

```
scaler.center_
```

```
array([3.53910000e+00, 2.90000000e+01, 5.22931763e+00,
1.04878049e+00,1.16500000e+03, 2.81635506e+00])
```

Now, let’s display the IQR stored in `RobustScaler()`:

```
scaler.scale_
```

We can see the IQR for each variable in the following output:

```
array([2.16550000e+00, 1.90000000e+01, 1.59537022e+00,
9.41284380e-02,9.40000000e+02, 8.53176853e-01])
```

Let’s now compare the distribution of the variables before and after rescaling. Let’s first plot histograms of the 
unscaled variables:

```
X_test.hist(bins=20, figsize=(20, 20))
plt.show()
```

![unscaled variables' distribution]({{ site.baseurl }}/assets/images/posts/scaling/raw-data.png)   

And now, let’s plot histograms of the scaled variables:

```
X_test_scaled.hist(bins=20, figsize=(20, 20))
plt.show()
```

![variable distribution after robust scaling]({{ site.baseurl }}/assets/images/posts/scaling/robust.png)   

In this opportunity, the scaling did not return variables on a similar scale. For this dataset, minmax scaling was the 
best choice.

## Docs

- [sklearn.preprocessing.standardscaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

- [sklearn.preprocessing.minmaxscaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)

- [sklearn.preprocessing.robustscaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html)


## Additional resources

For tutorials and step by step code implementations on other variable scaling methods, visit our course 
[Feature engineering for machine learning](https://www.trainindata.com/p/feature-engineering-for-machine-learning) or check out our book 
[Python Feature Engineering Cookbook](https://www.amazon.com/Python-Feature-Engineering-Cookbook-transforming-dp-1804611301/dp/1804611301).