---
layout: post
title: "KNN imputation of missing values in machine learning"
author: sole
description: "KNN imputation is a simple imputation technique to replace missing data for machine learning while preserving the variable distribution."
excerpt: "KNN imputation is a simple imputation technique to replace missing data for machine learning while preserving the variable distribution."
categories: [Data Preprocessing, Feature Engineering]
image: assets/images/posts/knn-imputation-of-missing-values-in-machine-learning/knn-imputation-missing-values.png
---

Missing values are data entries that are not recorded or are absent from a dataset. Missing data occurs due to various reasons such as data collection errors, equipment malfunctions, or respondents choosing not to answer certain questions.

Incomplete datasets make data analysis and training machine learning models challenging. Most machine learning models need numerical values as input. And some Python implementations do also require complete datasets for the training of the models.

Hence, addressing missing values is important step of the data preprocessing pipeline in any data science project, to ensure the quality and accuracy of the analysis. Missing values can be removed from the data, or replaced by estimates of their values through missing data imputation techniques.

We discussed univariate [imputation methods](https://www.blog.trainindata.com/your-guide-to-missing-values-imputation/) using Scikit-learn’s [SimpleImputer](https://www.blog.trainindata.com/imputing-missing-data-with-scikit-learns-simple-imputer/), as well as and multiple imputation in recent articles. In this blog, we’ll focus in KNN imputation.

## KNN imputation

Nearest neighbor imputation algorithms efficiently fill in missing data by replacing each missing value with a value derived from similar cases within the entire dataset.

Imputation with K-Nearest Neighbors (KNN) estimates missing values in a dataset by considering the values of the closest data points, determined by a distance metric like Euclidean distance. The missing value is then assigned the average of these nearest neighbors’ values, weighted by their proximity.

Consider the following dataset with 5 variables and 11 observations. We aim to impute the missing value in the 5th row of variable 2. First, we identify the row’s 3 closest neighbors (highlighted by the squared boxes) using a KNN algorithm. Then, we calculate the average of the values for variable 2 from these neighbors.

![KNN imputation illustration]({{ site.baseurl }}/assets/images/posts/knn-imputation-of-missing-values-in-machine-learning/knn-imputation-scheme.png)

The imputed value is calculated as (Value1 x w1 + Value2 x w2 + Value3 x w3) / 3, where w1, w2, and w3 are weights proportional to the distance of each neighbor from the data point being imputed.

## Considerations

KNN imputation replaces missing values as the weighted average of the closest neighbors to the observations with nan values. This means that this imputation method can only be applied to impute numerical variables.

In addition, the nearest neighbor algorithm computes distances between observations, hence, it is only suitable for numeric data. If your dataset contains variables with other data types, such us datetime or categorical variables, you’ll need to encode them first, or try a different missing value imputation technique.

[![14 Common Feature Engineering Questions]({{ site.baseurl }}/assets/images/posts/winsorization-handling-outliers-in-machine-learning/Want-a-quick-feature-engineering-reference-Download-our-free-booklet-14-Common-Feature-Engineering-Questions-covering-encoding-scaling-missing-data-outliers-and-more-Get-the-free-bookle.png)](https://www.trainindata.com/p/14-common-feature-engineering-questions)

## KNN imputation with Python

We can easily perform KNN imputation using scikit-learn’s KNNImputer to replace null values. This transformer will implement the procedure that we discussed in the precedent section.

Let’s begin by importing the required libraries and prepare the data:

```
import matplotlib.pyplot as plt

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.impute import KNNImputer
```

Let’s load the [credit approval dataset](https://archive.ics.uci.edu/dataset/27/credit+approval) directly from the UCI Machine Learning Repository.

[![Python Feature Engineering Cookbook book cover]({{ site.baseurl }}/assets/images/posts/detect-outliers-in-python/PFEC2ED.png)](https://www.packtpub.com/en-us/product/python-feature-engineering-cookbook-9781835883587)

```
from ucimlrepo import fetch_ucirepo
import random
import numpy as np

variables = ["A2", "A3", "A8", "A11", "A14", "A15", "target"]

credit_approval = fetch_ucirepo(id=27)
data = credit_approval.data.features
data["target"] = credit_approval.data.targets["A16"].map({"+": 1, "-": 0})

# reproduce the missing values injected in the original prepared dataset
random.seed(9001)
for i, var in enumerate(["A3", "A8", "A9", "A10"]):
    idx = list(set(random.randint(i, len(data)) for _ in range(100)))
    data.loc[idx, var] = np.nan

data = data[variables]
```

Let’s divide the dataset into a training set and a test set:

```
X_train, X_test, y_train, y_test = train_test_split(

    data.drop("target", axis=1),

    data["target"],

    test_size=0.3,

    random_state=0,

)
```

Let’s configure the imputer to replace missing data with the weighted average of its 5 nearest neighbors:

```
imputer = KNNImputer(n_neighbors=5,weights="distance").set_output(transform="pandas")
```

We now apply fit(). That will train the 5-KNN algorithm:

```
imputer.fit(X_train)
```

After training the algorithm, we can apply transform to replace the missing values. During transform, the missing value estimation takes place, to find the suitable replacements for each observation.

```
X_train_t = imputer.transform(X_train)

X_test_t = imputer.transform(X_test)
```

The imputed datasets are pandas dataframes, because we used the set_output API to change the container from the default, which is numpy arrays, to pandas dataframes.

## Advantages of KNN imputation

KNN imputation offers some advantages respect to simpler imputation methods like mean value imputation, or replacement with arbitrary values.

For once, it returns better estimates of the missing data. KNN imputation also aims to preserve the original variable distribution, precisely by returning accurate estimates of the null values. By using neighboring data points, it preserves the local structure and relationships within the dataset.

In addition, KNN imputation does not assume an underlying distribution for the data, making it suitable for datasets that do not fit standard distributions. And we can adjust the number of neighbors (k) and the distance metric to better suit the specific dataset and improve imputation accuracy.

## Shortcomings of KNN imputation

On the flip side, imputation with KNN algorithms adds an additional step of complexity, because we need to train a machine learning model to predict and replace missing data, so that we can use the complete datasets to train the machine learning models that we are actually interested in.

Like this, we compound uncertainty, the uncertainty around the missing values with the uncertainty of the machine learning algorithm we want to use.

Training an additional model to predict missing data incurs in higher computation al costs. And also KNN algorithms do not scale, so their application is usually limited to smaller dataset.

## Resources

If you want to learn more about KNN imputation or other missing value imputation methods, check out the following hands-on resources:

- [Feature engineering for machine learning course](https://www.trainindata.com/p/feature-engineering-for-machine-learning)
- [Python Feature Engineering Cookbook](https://www.packtpub.com/en-us/product/python-feature-engineering-cookbook-9781835883587)
- [Feature engineering for time series forecasting course](https://www.trainindata.com/p/feature-engineering-for-forecasting)
