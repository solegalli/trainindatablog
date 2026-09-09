---
layout: post
title: "Mastering Feature Importance in Machine Learning with Python"
author: sole
description: "Find out how to calculate feature importance scores with Python."
excerpt: "Find out how to calculate feature importance scores with Python."
categories: [Feature Selection, Interpretable Machine Learning, Machine Learning]
image: assets/images/posts/feature-importance-with-python/cover-4.png
---

Feature importance plays a crucial role in the field of machine learning, as it allows us to identify and prioritize the most important features that contribute to the predictive power of our models.

Feature importance is used to understand our model’s predictions and also to select the most relevant subset of features and therefore build simpler, faster, and more interpretable machine learning models.

Whether you’re a data scientist, a machine learning enthusiast, or just curious about the inner workings of algorithms, understanding the importance of features is essential for building robust and interpretable models.

In this blog post, we will explore the concept of feature importance, different methods to assess it during data analysis, and how to implement these techniques using Python.

*To learn more about how to calculate feature importance and use it to select features or interpret machine learning models, check out our courses [Feature Selection for Machine Learning](https://www.trainindata.com/p/feature-selection-for-machine-learning) and [Machine Learning Interpretability](https://www.trainindata.com/p/machine-learning-interpretability).*

[![Feature Selection for Machine Learning, online course.]({{ site.baseurl }}/assets/images/posts/feature-selection-with-filter-methods/feature-selection-course-1024x576.png)](https://www.trainindata.com/p/feature-selection-for-machine-learning)

## What is Feature Importance?

In machine learning, feature importance refers to the influence or contribution of each input feature in predicting the target variable. By quantifying the importance of the predictors, we can gain insights into which variables have the most significant impact on our model’s predictions. This knowledge is valuable for various tasks, including feature selection, dimensionality reduction, and model interpretation.

Now, you might be wondering, “How do we calculate feature importance?” There are various methods to calculate feature importance. We can derive importance straightaway from some machine learning models, like linear and logistic regression and decision tree-based models like random forests and gradient boosting machines like xgboost. There are also model-agnostic methods like permutation feature importance. And we can also use statistical tests.

In the rest of this tutorial, I will describe some of these methods and show you how to implement them in Python. Let’s dive in.

## Model-derived feature importance

Model-derived feature importance methods, such as coefficients in linear models or feature importance in decision tree-based algorithms, are specific to the chosen model. These methods derive importance scores directly from the model itself.

### Linear regression models

In linear models like linear regression or logistic regression, the coefficients associated with each feature indicate their importance. Features with larger coefficients contribute more to the model’s predictions.

Deriving feature importance through linear models only makes sense when there is a linear relationship between the features and the target. If this is not the case, then the importance derived is meaningless.

In addition, the magnitude of the coefficient is influenced by the scale of the variable. Changing the variable scale will change its contribution to the model’s prediction. But more importantly, variables on larger scales will overshadow variables on smaller scales. Hence, when assessing feature importance with linear models, it is important to scale the data before training the model.

Let’s see how we can do this in Python.

#### Feature importance from linear models with Python

Let’s’ begin by importing the necessary libraries, classes and functions:

```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from yellowbrick.model_selection import FeatureImportances
```

Now, we load the house price data set as a dataframe. To follow best practices, we split the data into train and test. The importance of the features should be assessed only using the training data, so if we select features based on it, we have a completely naive test set for a fair evaluation.

```
from sklearn.datasets import fetch_openml

variables = [
'YearRemodAdd', 'LotArea', 'SalePrice', 'WoodDeckSF', 'LowQualFinSF',
'FullBath', '2ndFlrSF', 'KitchenAbvGr', 'TotalBsmtSF', 'YearBuilt'
]

data = fetch_openml(name='house_prices', as_frame=True).frame[variables]

X_train, X_test, y_train, y_test = train_test_split(
    data.drop(labels=['SalePrice'], axis=1),
    data['SalePrice'],
    test_size=0.3,
    random_state=0,
)
```

Now we set up and fit a standard scaler:

```
scaler = StandardScaler()
scaler.fit(X_train)
```

Next, we set up and fit a multivariate linear regression. Note that we train the model on the scaled data:

```
model = LinearRegression()
model.fit(
    scaler.transform(X_train),
    y_train,
)
```

And now, let’s make a bar chart with the feature importance for visualization. The feature importance is the absolute value of the regression coefficients:

```
importance = pd.Series(
    data = np.abs(model.coef_),
    index = X_train.columns,
)
importance.sort_values(inplace=True, ascending=False)
importance.plot.bar(figsize=(20,5), fontsize=14)
plt.title("Importance based on linear model coefficients", fontsize=20)
plt.ylabel("Absolute coefficient value", fontsize=20)
plt.show()
```

In the following image, we see the features sorted by their coefficient magnitude:

![Bar plot with the feature importance derived from linear regression, shown in decreasing order.]({{ site.baseurl }}/assets/images/posts/feature-importance-with-python/linear-importance.png)

We can go 1 step further, and automate plotting the variable importance with yellowbrick. [Yellowbrick](https://www.scikit-yb.org/en/latest/api/model_selection/importances.html) is an open-source Python library for predictive model evaluation.

```
plt.rcParams["figure.figsize"] = (5,10)
viz = FeatureImportances(
    model,
    labels=X_train.columns,
)
viz.fit(
    scaler.transform(X_train),
    y_train,
)
viz.show()
```

In the following image we see the features ordered by their importance:

![Barplot created with the Python library yellowbrick showing the feature importance derived from linear regression.]({{ site.baseurl }}/assets/images/posts/feature-importance-with-python/linear-importance-yb-545x1024.png)

In this example, we did not use regularization, but that is always an option. The contribution, that is, the importance of each feature, will change depending on the strength of the regularization penalty.

### Decision tree-based models

Decision tree-based algorithms, such as random forest, provide a natural way to assess feature importance. These algorithms quantify importance based on the reduction in impurity (e.g., Gini impurity) achieved by each feature. Features that lead to the greatest reduction in impurity across the ensemble of trees are considered more important.

#### Feature importance from random forests with Python

Let’s’ begin by importing the necessary libraries, classes and functions:

```
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from yellowbrick.model_selection import FeatureImportances
```

We load and split the data as per the previous section, so I will skip those steps here and jump directly to fitting a random forest and deriving feature importance. I will use regression trees but the procedure is identical if you use the RandomForestClassifier.

```
model = RandomForestRegressor(random_state=20)
model.fit(X_train, y_train)
```

With the trained model, we can now plot the feature importance:

```
importance = pd.Series(
    data = model.feature_importances_,
    index = X_train.columns,
)
importance.sort_values(inplace=True, ascending=False)
importance.plot.bar(figsize=(20,5), fontsize=14)
plt.title("Importance based on randomf forests", fontsize=20)
plt.ylabel("Importance value", fontsize=20)
plt.show()
```

In the following image, we see the importance of the features in decreasing order:

![Bar plot showing feature importance derived from random forests in decreasing order.]({{ site.baseurl }}/assets/images/posts/feature-importance-with-python/random-forest-importance-1024x355.png)

We can go 1 step further and automate plotting with yellowbrick:

```
plt.rcParams["figure.figsize"] = (5,10)
viz = FeatureImportances(
    model,
    labels=X_train.columns,
)
viz.fit(X_train, y_train)
viz.show()
```

In the following image we see the importance of the different features:

![Bar plot created using the Python library yellowbrick showing feature importance derived from random forests.]({{ site.baseurl }}/assets/images/posts/feature-importance-with-python/random-forest-importance-yb-573x1024.png)

### Performance drop-derived feature importance

An alternative way to determine the contribution of a feature to the model’s predictions is to remove the feature, retrain the model, and then assess the drop in performance, if any. This procedure has the advantage that it can be used with models that do not have natural importance scores, like, for example, support vector machines. The variable importance measures are determined by the drop in any performance metric that we are interested in.

[Feature-engine](https://feature-engine.trainindata.com/en/latest/api_doc/selection/RecursiveFeatureElimination.html), a Python open source library for feature engineering and selection, offers this functionality out of the box, so let’s see how we can implement that.

Let’s’ begin by importing the necessary libraries, classes, and functions:

```
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from feature_engine.selection import RecursiveFeatureElimination
```

We load and split the data as per the previous section, so I will skip those steps here and directly compute the feature importance by recursive feature elimination.

The RecursiveFeatureElimination from Feature-engine is designed to select features based on the performance drop. But here, we will only use it to determine the feature’s importance. Through the parameter scoring, we can decide which evaluation metric we want to assess.

```
model = RandomForestRegressor(random_state=20)
rfe = RecursiveFeatureElimination(estimator=model, scoring="r2", cv=2)
rfe.fit(X_train, y_train)
```

After training the RecursiveFeatureElimination, we can plot the feature importance (Note that the methodology of Feature-engine uses cross-validation to better estimate the feature’s contribution).

```
importance = pd.Series(
    data = rfe.feature_importances_,
    index = X_train.columns,
)
importance.sort_values(inplace=True, ascending=False)
importance.plot.bar(figsize=(20,5), fontsize=14)
plt.title("RFE deriveed importance", fontsize=20)
plt.ylabel("Change in R2", fontsize=20)
plt.show()
```

In the following image, we see the drop in performance that occurs when dropping each of the features:

![Bar plot showing feature importance obtained by recursive feature elimination.]({{ site.baseurl }}/assets/images/posts/feature-importance-with-python/rfe-importance-1-1024x360.png)

If we now want to reduce the number of features in the dataset, we can do:

```
X_train_reduced = rfe.transform(X_train)
X_test_reduced = rfe.transform(X_test)
```

In this case, we used recursive feature elimination with a regression model, but we can similarly use it with any other classifier from Scikit-learn, by changing the performance metric, to for example, the ROC-AUC.

## Permutation feature importance

Permutation feature importance is a model-agnostic method that measures feature importance by randomly shuffling the values of a single feature and observing the impact on model performance. It was introduced by Leo Friedman.

By measuring the degradation in performance caused by the shuffled feature, we can quantify its importance. This approach works well with any machine learning algorithm.

### Permutation feature importance with Python

Let’s’ begin by importing the necessary libraries, classes and functions:

```
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
```

We load and split the data as per the previous section, so I will skip those steps here and jump directly to estimating feature importance by feature shuffling:

```
model = RandomForestRegressor(random_state=20)
model.fit(X_train, y_train)
results = permutation_importance(
    model,
    X_train,
    y_train,
    n_repeats=3,
    random_state=0,
)
```

Now let’s plot the feature importance:

```
importance = pd.Series(
    data = results.importances_mean,
    index = X_train.columns,
    )
importance.sort_values(inplace=True, ascending=False)
importance.plot.bar(figsize=(20,5), fontsize=14)
plt.title("Permutation feature importance", fontsize=20)
plt.ylabel("Coefficient of determination", fontsize=20)
plt.show()
```

In the following image, we see the drop in performance caused by feature permutation:

![Bar plot showing permutation feature importance in decreasing order.]({{ site.baseurl }}/assets/images/posts/feature-importance-with-python/permutation-importance-1024x362.png)

The importance score is given by the R-squared which is the default parameter. But we can change it through the permutation_importance function to other metrics like mean squared error (mse).

## Feature importance with statistical tests

Statistical modeling can also be used to infer the predictor contributions towards a target or dependent variable.

ANOVA (analysis of variance), correlation analysis, and the chi-square test are statistical techniques commonly employed to evaluate feature importance in various fields, including data science.

ANOVA assesses the significance of differences in means among groups or categories, allowing data scientists to determine if a particular feature has a significant impact on the outcome of a classification target variable.

Correlation analysis measures the strength and direction of the relationship between two continuous variables, enabling us to identify variables that are strongly associated with a continuous target variable.

The chi-square test evaluates the association between two categorical variables, indicating whether there is a significant relationship between them. With the chi-square test, we can assess the contribution of categorical variables to a discrete target.

Statistical tests will return a probability in the form of a p-value. The smaller the p-value, the more relevant the predictor variable is to the target.

Nonparametric tests can also be used to infer the importance of variables, when the data does not fulfill the assumptions of parametric tests.

By employing these statistical methods, researchers can determine the relative importance of different features with respect to the target variable, aiding in decision-making and identifying key factors that drive the observed patterns or phenomena.

## Wrap up

Feature importance is a fundamental concept in machine learning that allows us to identify the most influential input features in our models. By understanding the importance of features, data scientists and machine learning practitioners can improve model performance and prediction accuracy, gain insights into the underlying data, and enhance interpretability.

Feature importance also guides variable selection, therefore allowing us to create simpler, faster and more interpretable models.

Whether you choose coefficients, decision tree-based methods, permutation feature importance, or SHAP values, the scikit-learn library provides powerful tools to assess and visualize feature importance in Python.

## Additional resources

- Course: [Feature Selection for Machine Learning](https://www.trainindata.com/p/feature-selection-for-machine-learning)
- Course: [Machine Learning Interpretability](https://www.trainindata.com/p/machine-learning-interpretability)
- Book: [Feature Selection for Machine Learning with Python](https://www.trainindata.com/p/feature-selection-in-machine-learning-book)
