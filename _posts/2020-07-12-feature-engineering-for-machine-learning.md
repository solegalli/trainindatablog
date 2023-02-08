---
layout: post
title:  "Feature engineering for machine learning: What is it?"
author: sole
description: Discover why and how we select features in machine learning.
excerpt:  Discover why and how we select features in machine learning.
categories: [ Feature engineering, Python, Machine learning ]
image: assets/images/posts/feateng/cover.gif
---

Feature engineering is the process of transforming variables, and extracting and creating 
new variables from the original data points, to train machine learning models. It is commonly 
done together with exploratory data analysis.

Data in its original format can almost never be used straightaway to train classification 
or regression models. Instead, data scientists devote a huge chunk of their time to data 
preprocessing to train machine learning algorithms.

Imputing missing data, transforming categorical data, feature transformation with math 
functions or discretizing numerical data, are all examples of feature engineering. Feature 
engineering also involves putting the variables on the same scale, for example through normalization.

Finally, feature extraction from various data sources like text, relational databases, 
time series, and images is also key to create input data that can be used to train predictive models.

Feature engineering is key to improving the performance of machine learning algorithms. 
Yet, it is very time-consuming. Fortunately, there are many Python libraries that we can 
use for data preparation. Some of them even automate the process of feature creation. 
These libraries are Pandas, Scikit-learn, [Feature-engine](https://feature-engine.trainindata.com/en/latest/),
[Category Encoders](https://contrib.scikit-learn.org/category_encoders/),
[tsfresh](https://tsfresh.readthedocs.io/en/latest/),
and [Featuretools](https://featuretools.alteryx.com/en/stable/index.html).

In this article, we will answer the following questions:

- Why do we engineer features for machine learning?

- What are the main feature engineering techniques?

- How can we do feature engineering with Python?

Let’s get started.

*If you want to know more about feature engineering in machine learning, check out our 
[Course Feature Engineering for Machine Learning](https://www.trainindata.com/p/feature-engineering-for-machine-learning) 
and our [Python Feature Engineering Cookbook](https://www.amazon.com/Python-Feature-Engineering-Cookbook-transforming-dp-1804611301/dp/1804611301).*

## Why do we engineer features for machine learning?

Features in the raw data are almost never suitable inputs for machine learning models. 
Instead, data scientists need to fill in missing data, transform categorical features into 
numbers, create new variables, and much more.

There are various reasons why we devote so much time to engineering features in data science:

- Some machine learning libraries do not support missing values or categorical data as inputs, for example, Scikit-learn.

- Some machine learning algorithms are sensitive to the variable scale. For example linear regression models, support vector machines, neural networks, and distance-based algorithms like nearest neighbors.

- Some algorithms are sensitive to outliers, for example, linear regression models.

- We can create useful predictor variables by transforming or combining features.

- We can extract many useful features from the raw data, for example, from dates, transactions, time series, and text.

- Overall, feature engineering is used to optimize predictive models, as it can increase the values of the performance metrics.

With all these reasons to engineer the variables in our datasets, it’s time to move on to 
how we carry out data preprocessing in data science.

In the rest of the article, we’ll go over the main feature engineering steps and the most 
widely used methods for data preprocessing.

![feature engineering cartoon]({{ site.baseurl }}/assets/images/posts/dataprep/architecture-design-gif.gif)


## What are the main feature engineering techniques?

As I mentioned previously, data scientists must deal with different data types, missing data, 
categorical features and skewed variables, in order to train machine learning models. 
Creating new features can also improve model accuracy.

The typical steps in a feature engineering process are:

- Missing data imputation

- Categorical variable encoding

- Variable transformation

- Variable discretization

- Handling Outliers

- Feature scaling

- Creating features from dates and time

- Extracting features from relational data and time series

- Extracting features from text

Let’s go over the main methods to carry out each one of these feature engineering steps.

## Missing data imputation

Missing data imputation consists in replacing missing values with numbers. These numbers are 
typically statistical estimates of the missing values. The goal of the imputation is to 
produce a complete data set that can be used to train machine learning models.

There are many missing data imputation methods used in data science. If the variables are 
numerical, missing values can be replaced with the mean or median value of the variable. 
If the variables are categorical, missing observations can be replaced by the most frequent 
category. Data scientists also replace missing data with arbitrary values.

In the following table I summarize the main data imputation methods used by data scientists:

![missing data imputation methods, summary table]({{ site.baseurl }}/assets/images/posts/feateng/Table-missing-data-imputation.png)


Every missing data imputation method has advantages and limitations.

In the following image, I summarize the the main limitations of the missing data imputation 
methods described:

![Limitations of missing imputation methods]({{ site.baseurl }}/assets/images/posts/feateng/04_MissingData.png)

Depending on the percentage of missing data and the model we intend to train, we would choose 
a different way to replace missing values.

## Categorical variable encoding

Categorical variables are those that have categories as values, and not numbers. Examples are 
the variable marital status, with values of married, divorced, single, and other.

Some machine learning models can handle categorical variables right out of the box, like 
algorithms based on decision trees. But even so, with the Scikit-learn implementation of 
decision tree based models, we still need to transform the categories into numbers. The 
process of transforming strings into numbers is called categorical variable encoding.

The following diagram shows the most widely used methods to replace categories by numbers:

![categorical encoding techniques and methods]({{ site.baseurl }}/assets/images/posts/feateng/04_Categorical_Encoding.png)


There are various ways in which data scientists transform the categories into numbers. In 
one-hot encoding, each category becomes a binary variable that indicates the presence or 
absence of the category in an observation.

Data scientists also replace categories with their count or frequency, a method fairly 
frequent in Kaggle. Categories can also be replaced with arbitrary numbers, or with the 
weight of the evidence.

When the data has rare or infrequent categories, data scientists like grouping them into 
a new category called “Other” or “Rare”. This procedure tends to improve machine learning 
model generalization, in particular for tree-based methods.

In the following image, I show the grouping of infrequent categories into a new category c
alled “Rare.”

![grouping infrequent categories before categorical encoding]({{ site.baseurl }}/assets/images/posts/feateng/04_Rare_Labels.png)

## Variable transformation

Some machine learning models make assumptions about the data. When the assumptions are not 
met, the conclusions derived from the model might not be reliable.

Fortunately, we can apply mathematical transformations to make the data meet the assumptions 
of the machine learning model.

In the following image, I show the intended effect of transforming variables in machine learning:

![variable transformation in machine learning]({{ site.baseurl }}/assets/images/posts/feateng/04_Variable_Transformation.png)


The most commonly used variable transformations are the log transform, the reciprocal and 
the square root transformation. The Box-Cox and Yeo-Johnson implementations of exponential 
transformations are also widely used in data science.

The Box-Cox and Yeo-Johnson automatically find the exponent to returns the closest possible 
distribution to the normal distribution. This is why, these transformations are very popular 
in data science. However, it is always advisable to perform data analysis and visualization 
to corroborate that the transformation returned the expected distribution. I say more about 
this in my [article on variance stabilizing transformations](https://www.blog.trainindata.com/variance-stabilizing-transformations-in-machine-learning/).

## Discretization

Discretization, sometimes called binning, refers to sorting the values of the variable 
into bins or intervals. Many predictive models, like decision trees and Naïve Bayes, work 
better with discrete attributes. In fact, decision tree based models make decisions based 
on discrete partitions of the attributes. Thus, discretization can not only improve model 
performance but also reduce the time it takes to train the models.

In the following image, I show the effect of discretization and the main discretization procedures:

![variable discretization in machine learning]({{ site.baseurl }}/assets/images/posts/feateng/04_Discretisation.png)

Data scientists create equal-width or equal-frequency intervals to sort the variable values. 
Sometimes, the limits of the intervals are defined arbitrarily, based on domain knowledge 
of the data and the field of application. And sometimes, machine learning practitioners 
let decision trees find the interval limits. It‘s common to use shallow decision trees to 
craft the features, to avoid overfitting.

## Handling outliers

Outliers are values that are unusually high or unusually low respect to the rest of the 
observations of the variable. Outliers can be truly predictive or simply a byproduct of the 
data collection process we used.

If we want to detect an unusual occurrence like Fraud, or a rare disease, then outliers might 
indeed be very predictive. We probably don’t want to discard those observations. But outliers 
can as well just a measuring artifact, and in these cases removing them may improve model 
performance.

![handling outliers]({{ site.baseurl }}/assets/images/posts/dataprep/outliers.png)

It is common practice to cap the variable values at a certain minimum or maximum value. These 
values are determined by statistical estimates of the variable limits. If the variable is 
normally distributed, data scientists can use the mean and the variance. If the variable is 
skewed, we can use instead the inter-quartile range proximity rule or the percentiles.

## Feature scaling

Many predictive models are sensitive to the scale of the variables. Thus, it is common practice 
to set all features to the same scale. There are multiple ways to scale features, but the most 
commonly used are standardization and min-max scaling.

![feature scaling in machine learning]({{ site.baseurl }}/assets/images/posts/feateng/scaling.png)


Feature standardization involves normalization, that is, subtracting the mean from each value, 
followed by dividing the result by the standard deviation. Feature standardization centers the 
variables at 0 with a variance of 1.

Min-max scaling consists of re-scaling the variable to 0–1. To re-scale the variables, we subtract 
the minimum from each value and divide by the value range. The value range is calculated as the 
maximum minus the minimum value of the variable. When variables are skewed, min-max scaling 
offers a good alternative to standardization.

## Creating features from date and time

Raw dates and time are not suitable to train machine learning models. But, we can extract a 
lot of predictive features from date and time variables. From dates, we can extract the year, 
the semester, the quarter, the month and the week. We can also obtain the day number, the day 
of the week, and much more.

![Creating features from date and time]({{ site.baseurl }}/assets/images/posts/feateng/calendar-gif.gif)


From time, we can extract the hour, the minutes, and the seconds, to name a few.

Capturing time differences into new variables is also common practice in data science. For 
example, from the date of birth and the date of application, we can create a variable with 
the age of the customer.

## Extracting features from transactions and time series

Transaction data is information recorded from transactions, and it is generally stored in 
relational databases. Examples of transaction data are records of every sale made in a shop, 
or the balances in bank and credit accounts over time. We can extract a lot of predictive 
features from transactions in order to predict an outcome.

![transaction data and relational databases]({{ site.baseurl }}/assets/images/posts/feateng/transaction-data.png)


We usually look at a small window of time to extract features from transactions. From these 
windows, we can find the greatest value transaction and the minimum value transaction. We 
can also obtain the mean transaction value, the total number of sales, and so on.

In time series forecasting, it is common to use past values of the data to predict future 
outcomes. We can create lag and window features, capturing information in the past to predict 
the future utilizing machine learning models.

## Extracting features from text

Data scientists can also extract a lot of information from text. Straightforward variables 
are the number of words and unique words, the number of paragraphs and the number of sentences. 
We can obtain also the lexical diversity by the ratio of unique words and total words.

![Extracting features from text]({{ site.baseurl }}/assets/images/posts/feateng/text.png)


More informative features are returned by the number or frequency of occurrence of a word 
in a text. This is normally accomplished by creating bag of words or term-frequency-inverse 
document frequency representations of the data. In addition, we can capture the representation 
of word-combinations, called n-grams. These are some of the most basic NLP techniques to 
craft features from text.


## Feature engineering vs feature selection

Feature engineering is the process of using domain knowledge of the data and statistical 
methods to create predictor variables to train predictive models.

Feature selection, on the other hand, refers to the process of selecting the best features 
from the dataset, and it is generally done after data preprocessing. The aim of feature 
selection is to reduce the number of features used to train a machine learning model, and 
it is key to create simpler and more interpretable machine learning models. It can also 
improve model accuracy.

There are many feature selection algorithms that help us select the most predictive subset 
of features. I discuss these techniques in a separate article.

*For more information, including code examples, about feature selection, check out our book
[Feature Selection in Machine Learning with Python](https://leanpub.com/feature-selection-in-machine-learning/) 
or our course [Feature Selection in Machine Learning](https://www.trainindata.com/p/feature-selection-for-machine-learning).*


## Feature engineering vs dimensionality reduction

Principal component analysis or PCA, is a method used capture the maximum variability of 
the dataset in fewer features. PCA combines the features of the data into new features, 
the principal components. Then, we need to decide how many of those PAs we would input to 
the model.

## How can we do feature engineering with Python?

There are excellent Python libraries for feature engineering. Pandas, Scikit-learn and 
Feature-engine provide tools for missing data imputation, categorical encoding, variable 
transformation and discretization. With these libraries we can also create new variables 
by combining existing features.

Category Encoders is an extremely popular library to encode categorical features. It has 
the most extensive tool set to transform the categories into numbers.

For a short tutorial on how to use Python libraries for feature engineering in machine 
learning, check my article [Python libraries for feature engineering](https://towardsdatascience.com/practical-code-implementations-of-feature-engineering-for-machine-learning-with-python-f13b953d4bcd).

![Python libraries for feature engineering]({{ site.baseurl }}/assets/images/posts/feateng/FE-python-libraries.png)

Featuretools and tsfresh are 2 Python libraries that offer out-of-the-box automated 
feature engineering tools. Featuretools provides tools to create predictor variables from 
transactions and it is a great Python library to handle features that come from more than o
ne dataset. It automates the aggregation of data into new features that can be used for 
classification or regression.

Tsfresh offers automates feature creation from time series, in particular, for time series 
classification and regression. With tsfresh we can automatically create more than 200 
predictor features from time series. tsfresh also includes a feature selection procedure 
to automatically find the most predictive variables.
