---
layout: post
title:  "Data discretization in machine learning"
author: sole
description: Why and how should we discretize data in machine learning.
excerpt:  Why and how should we discretize data in machine learning.
categories: [ Feature engineering, Python, Machine learning ]
image: assets/images/posts/data-disc/cover.gif
---

Data discretization, also known as binning, is the process of grouping continuous values 
of variables into contiguous intervals.This procedure transforms continuous variables into 
discrete variables, and it is commonly used in data mining and data science, as well as 
to train models for artificial intelligence.

In fact, a common step before training machine learning algorithms is the discretization 
of continuous variables. But why do we discretize continuous variables, and how do we 
sort continuous data into discrete values?

These are the questions that we will address throughout this article.

## What is discretization?

In discretization, we convert continuous variables into discrete features. To do this, 
we compute the limits of the contiguous intervals that span the entire variable value 
range. Next, we sort the original values into those intervals. These intervals, which are 
now discrete values, are then handled as categorical data.

The challenge in discretization is identifying the thresholds or limits that define the 
intervals into which the continuous values will be sorted. To this end, there are various 
discretization methods that we can use, each with advantages and shortcomings.

![discretization challenge: identifying the interval limits]({{ site.baseurl }}/assets/images/posts/data-disc/ChallengeDiscretization.gif)

But before we talk about discretization methods, let’s discuss why the discretization of 
continuous features can be useful.

## Why is discretization useful?

Several regression and classification models, like decision trees and Naive Bayes, perform 
better with discrete values.

Decision trees make decisions based on discrete attribute partitions.A decision tree 
assesses all feature values while training to determine the ideal cut-point. As a result, 
the more values the feature has, the longer the training time of the decision tree. 
Therefore, the discretization of continuous features can speed up the training process.

Discretization has additional benefits. People will have an easier time understanding 
discrete values. In addition, if sorting observations into bins with equal frequency, 
skewed values are spread more evenly across the range.

Discretization can also minimize the influence of outliers by placing them in the lower 
or higher intervals together with the remaining values of the distribution. Like this, 
the coefficients of the linear regression models will not be biased by the presence of 
outliers.

Overall, discretization of continuous features makes the data simpler, the learning process 
faster, and can yield more accurate results.

## Is discretization all but an advantage?

Not really. Discretization can result in information loss, for example, by combining values 
that are strongly associated with different classes of the target values into the same bin.

![trade-offs of data discretization]({{ site.baseurl }}/assets/images/posts/data-disc/discretization-tradeoff.png)

A discretization algorithm’s goal is to determine the fewest intervals possible without 
significantly losing information. The algorithm’s task then becomes determining the 
cut-points for those intervals.

This raises the issue of how to discretize variables in machine learning.

## Discretization methods

The most popular discretization algorithms are equal-width and equal-frequency discretization. These are unsupervised discretization techniques because they find the interval limits without considering the target. Using k-means to find the interval limits is another unsupervised discretization technique. In all these methods, the user needs to define the number of bins into which the continuous data will be sorted beforehand.

Decision tree-based discretization techniques, on the other hand, can automatically 
determine the cut-points and optimal number of divisions. This is a supervised discretization
method because it finds the interval limits using the target as guidance.

![supervised discretization vs unsupervised discretization]({{ site.baseurl }}/assets/images/posts/data-disc/DiscretizationMethods.gif)


### Equal–width discretization

Equal-width discretization consists of dividing the range of continuous values into k 
equally sized intervals. Then, if the values of the variable vary between 0 and 100, the 
bins can be 0–20, 20–40, 40–60, 80–100.

We can carry out equal-frequency discretization in Python using the open source library 
Feature-engine.

Let’s make some imports:

```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from feature_engine.discretisation import EqualWidthDiscretiser
```

Let’s load the dataset:

```
X, y = fetch_california_housing(return_X_y=True, as_frame=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
```

Let’s perform equal-width discretization of 3 continuous variables into 8 intervals:

```
variables = ['MedInc', 'HouseAge', 'AveRooms']

disc = EqualWidthDiscretiser(bins=8, variables=variables, return_boundaries=True)

disc.fit(X_train)

train_t = disc.transform(X_train)
test_t = disc.transform(X_test)
```

Equal-width discretization does not alter the variable distribution dramatically. If a 
variable is skewed before the discretization, it will still be skewed after the 
discretization. We can assess the original variable distribution using histograms for 
the continuous variable and after the discretization using bar plots.

### Equal-frequency discretization

Equal-frequency discretization sorts the continuous variable into intervals with the same 
number of observations. The interval width is determined by quantiles. Equal-frequency 
discretization is particularly useful for skewed variables, as it spreads the observations 
over the different bins equally.

We can implement equal-frequency discretization utilizing Scikit-learn.

Let’s first make some imports:

```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
```

Let’s load the dataset:

```
X, y = fetch_california_housing(return_X_y=True, as_frame=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
```

Let’s now carry out equal-frequency discretization with Scikit-learn:

```
disc = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')

disc.fit(X_train[variables])

train_t = X_train.copy()
test_t = X_test.copy()

train_t[variables] = disc.transform(X_train[variables])
test_t[variables] = disc.transform(X_test[variables])
```

We can also implement equal-frequency discretization with 
[Feature-engine](https://feature-engine.trainindata.com/en/latest/api_doc/discretisation/EqualFrequencyDiscretiser.html).

### Discretization with k-means clustering

To create intervals or bins that group similar observations, we can use clustering 
algorithms like k-means. In discretization using k-means clustering, the partitions are 
the clusters identified by the k-means algorithm.

Discretization with k-means requires one parameter, which is k, the number of clusters 
or the number of bins. We can carry out k-means discretization with scikit-learn.


### Using decision trees for discretization

Decision tree methods discretize continuous attributes during the learning process. A 
decision tree evaluates all possible values of a feature and selects the cut-point that 
maximizes the class separation by utilizing a performance metric like the entropy or 
Gini impurity. Then it repeats the process for each node of the first data separation 
and for each node of the subsequent data splits, until a certain stopping criteria is 
reached. Therefore, decision trees can, by design, find the set of cut-points that 
partition a variable into intervals with good class coherence.

Discretization with decision trees is another top-down approach that consists of using 
a decision tree to identify the optimal partitions for each continuous variable.

Feature-engine has an implementation of discretization with decision trees, where 
continuous data is replaced by the predictions of the tree, which is a finite output. 
Each tree is fit with cross-validation to avoid overfitting.


For video tutorials on discretization, check our course 
[Feature Engineering for Machine Learning](https://www.trainindata.com/p/feature-engineering-for-machine-learning).


### Chi-merge

Chi-merge is another univariate supervised discretization procedure. It works as follows: 
First, it sorts the variable values in ascending order. Next, it groups the values into 
intervals that contain identical values. After obtaining these intervals, Chi-merge 
continuously merges adjacent intervals, if the observations in those intervals are not 
significantly different based on the chi-square test. It is then suitable only when the 
target variable is discrete.


## Final thoughts

Data discretization is used to accelerate the training time of decision tree-based 
models, reduce the impact of outliers on linear models, and make data more understandable 
for people.

The challenge in discretization is to find the interval limits that maximize model 
performance while minimizing training times and information loss.

There are various discretization methods that we can apply, including equal-width 
discretization, equal-frequency discretization, discretization with k-means or decision 
trees, and the bespoke method chi-merge.

We can use the open-source libraries Scikit-learn and Feature-engine for most 
discretization procedures.

### References

For a review on discretization techniques you may find the following articles useful:

Kotsiantis and Kanellopoulos, Discretization Techniques: a recent survey. GESTS International 
Transactions on Computer Science and Engineering, Vol.32 (1), 47-58, 2006.


Dougherty et al, Supervised and Unsupervised Discretization of Continuous Features, 
Machine Learning: Proceedings of the 12th International Conference, 1995


Lu et al, Discretization: An Enabling technique, Data Mining and Knowledge Discovery, 6, 
393–423, 2002.


Garcia et al, A survey of Discretization Techniques: Taxonomy and Empirical Analysis in 
Supervised Learning, IEEE Transactions on Knowledge in Data Engineering 25 (4), 2013.


Palaniappan and Hong, Discretization of Continuous Valued Dimensions in OLAP Data Cubes. 
International Journal of Computer Science and Network Security, VOL.8 No.11, November 2008.


## More resources

- [Feature Engineering for Machine Learning](https://www.trainindata.com/p/feature-engineering-for-machine-learning) — Online course

- [Python Feature Engineering Cookbook](https://www.amazon.com/Python-Feature-Engineering-Cookbook-transforming-dp-1804611301/dp/1804611301)

- [Scikit-learn documentation](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_discretization_classification.html)

- [Feature-engine documentation](https://feature-engine.trainindata.com/en/latest/api_doc/discretisation/index.html)