---
layout: post
title: "Understanding Permutation Feature Importance for Model Interpretation"
author: sole
description: "Permutation feature importance is obtained by randomly shuffling the feature values and assessing the decrease in the model's performance."
excerpt: "Permutation feature importance is obtained by randomly shuffling the feature values and assessing the decrease in the model's performance."
categories: [Interpretable Machine Learning]
image: assets/images/posts/permutation-feature-importance/Blog-banners.png
---

Permutation feature importance is a technique used in machine learning to assess the importance of different features in a predictive model. The basic idea is to measure how much the model’s performance deteriorates when the values of a particular feature are randomly shuffled or permuted while keeping other variables unchanged.

Permutation feature importance is a model agnostic interpretability method. This means that with this method, data scientists can explain the predictions of intrinsically explainable machine learning models, like linear regression and decision trees, and also the predictions of black box models, like one class support SVMs, isolation forests and neural networks.

In this blog post, we’ll explore the basics of permutation feature importance and its significance in model interpretation. For more comprehensive tutorials and code about this and other machine learning interpretability algorithms, check out our course “[Machine Learning Interpretability](https://www.trainindata.com/p/machine-learning-interpretability)”.

[![Machine learning interpretability online course]({{ site.baseurl }}/assets/images/posts/machine-learning-interpretability/interpretable-machine-learning-1024x576.png)](https://www.trainindata.com/p/machine-learning-interpretability)

## Permutation Feature Importance Mechanism

The methodology involves permuting the values of a single feature, disrupting its relationship with the target variable. By comparing the model’s performance before and after shuffling the feature, we obtain an “importance value”. This importance value reflects the decrease in model performance attributed to the shuffled feature.

Permutation feature importance works like this:

1. **Train the Model:** First, we train a machine learning model on the training data using the original features. This model could be any supervised learning classifier or regression model, such as a decision tree, random forests, or gradient boosting models.
2. **Evaluate Performance:** Next, we evaluate the model’s performance using a performance metric, such as accuracy, precision, or mean squared error, depending on the nature of the problem (classification or regression), and what we want to optimize.
3. **Permute Feature Values:** Now we randomly shuffle the values of a specific feature, breaking any original relationship between that feature and the target variable.
4. **Reevaluate Performance:** We apply the trained model (from step 1) to the dataset with the permuted feature, and we obtain its performance using the same metric.
5. **Calculate Importance:** The difference in performance between the model with the original feature and the model with the permuted feature indicates the importance of that particular feature. If the model’s performance drops significantly when the feature is permuted, it suggests that the feature is important for the model’s predictions.
6. **Repeat for Each Feature:** We repeat steps 3-5 for each feature in the dataset.

By the end of this procedure, we obtain an importance value for each feature in the dataset, which we can then analyze, for example through visualizations, by ranking them and identifying the most important features, or those whose shuffling does not affect the model’s performance at all.

It’s essential to calculate permutation feature importance using the test set or a held-out sample. If the model shows overfitting, shuffling features in the training set may not reflect the model’s generalization capability. Assessing variable importance by permutation in a test set provides more meaningful insights into model performance on unseen data.

## Metric Selection for Importance Evaluation

The feature importance measure is the drop in the model performance after shuffling a feature. We can use any performance metric that we like. For classification, we can use metrics like the ROC-AUC or the accuracy, while for regression, we can use the mean squared error or the r-squared, among others. We can also combine the use of multiple metrics.

## Randomness

Permutation feature importance is a metric obtained by randomly shuffling one feature and observing the resulting decrease in model performance.

Since the shuffle is a random process, different runs yield different values for feature importance. To address this variability, we shuffle each feature multiple times and then calculate the average importance value and its standard deviation.

There are three open source Python libraries that support permutation feature importance: Scikit-learn, Eli5, and Feature-engine. Scikit-learn and ELI5 obtain variable importance by shuffling the same feature multiple times and then averaging the results. In contrast, Feature-engine utilizes cross-validation. It takes a portion of the training set to train the model and evaluates feature importance on a held-out sample, averaging the drop in performance across multiple held-out samples.

## Application in Model Inspection and Feature Selection

Permutation feature importance serves dual purposes: model inspection and [feature selection](https://www.blog.trainindata.com/feature-selection-machine-learning-with-python/). In the context of model inspection, it helps explain how the model makes decisions and identifies crucial features affecting the output. As a feature selection technique, it assists in choosing features with higher importance.

## Python implementation

We mentioned previously, that there are at least 3 open source Python implementations of permutation feature importance. Scikit-learn’s implementation is geared to model inspection, whereas with ELI5 and Feature-engine we can use permutation feature importance for interpretability or for feature selection.

In this section, I’ll show how to carry out permutation feature importance with Scikit-learn, ELI5 and Feature-engine. We’ll use the house prices dataset from Kaggle.

Let’s begin by importing the necessary modules and functions:

```
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
```

Now, we load the house prices dataset. We will use a subset of variables to avoid complicating the demo with feature engineering:

```
variables = ['MSSubClass', 'LotArea', 'OverallQual', 'OverallCond',
             'YearBuilt', 'YearRemodAdd', 'BsmtFinSF1',
             'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',  '1stFlrSF',
             '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',
             'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
             'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars',
             'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
             '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
             'MoSold', 'YrSold', 'SalePrice']
```

```
data = fetch_openml(name='house_prices', as_frame=True).frame[variables]
```

Next, we separate the dataset into a training data set and a testing data set:

```
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(labels=['SalePrice'], axis=1),
    data['SalePrice'],
    test_size=0.3,
    random_state=0)
```

```

```

Following up, we train a random forest regression model:

```
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=3,
    random_state=2909,)

rf.fit(X_train, y_train)
```

We assess the performance of the model trained on the original dataset:

```
rf.score(X_train, y_train)
0.8078308037529935
```

```
rf.score(X_test, y_test)
0.780510833461595
```

```

```

Given that random forests are intrinsically explainable, we can obtain the global feature importance by analyzing the reduction in impurity using the gini metric. This comes out of the box with Scikit-learn:

```
pd.Series(
    rf.feature_importances_,
    index=rf.feature_names_in_,
).sort_values(
    ascending=False).plot.bar(figsize=(15, 5))
```

```
![Feature importance derived from the random forests]({{ site.baseurl }}/assets/images/posts/permutation-feature-importance/feature-importance-rf-1024x438.png)
```

We will now evaluate feature importance by permutation utilizing Scikit-learn.

### Permutation Feature Importance with Sklearn

To evaluate permutation feature importance with [Scikit-learn](https://scikit-learn.org/stable/modules/permutation_importance.html), we need to import the permutation importance function:

```
from sklearn.inspection import permutation_importance
```

Now, we can evaluate the feature importance by shuffling its values 5 times and obtaining the average drop in model performance:

```
perm = permutation_importance(
    rf,
    X_test, # we use the test set
    y_test,
    scoring= "r2", # we can use any metric
    n_repeats=5,
    random_state=0,
)
```

```

```

Let’s now put the values into a pandas dataframe, where the columns are in the index and the columns are the mean and standard deviation of the importance value of each feature:

```
importance = pd.DataFrame(
    {"importance_mean": perm["importances_mean"],
     "importance_std": perm["importances_std"]},
    index=rf.feature_names_in_)
```

To wrap up, let’s make a bar plot with the feature importance:

```
importance["importance_mean"].sort_values(
    ascending=False
).plot(figsize=(15, 5), kind="bar", yerr=importance["importance_std"])
plt.title("Permutation feature importance")
plt.ylabel("Performance drop (R2)")
plt.show()
```

As we see from this and the previous plot, permutation feature importance and the importance derived from the random forests show the same features as the most important ones.

![Bar plot showing the permutation feature importance determined with scikit-learn]({{ site.baseurl }}/assets/images/posts/permutation-feature-importance/pfi-skl-1024x460.png)

### Permutation Feature Importance with ELI5

We will now determine the permutation feature importance with [ELI5](https://eli5.readthedocs.io/en/latest/blackbox/permutation_importance.html) and following up we will use those values to select features. Let’s import the required functions and classes:

```
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.feature_selection import SelectFromModel
```

Now we calculate the permutation feature importance:

```
perm = PermutationImportance(
    rf,
    scoring="r2",
    n_iter=3,  # number of times each feature is shuffled
    cv="prefit",
    random_state=5,
)
perm.fit(X_test, y_test)
```

We can now collect the mean and standard deviation of the feature importance in a dataframe:

```
importance = pd.DataFrame({
    "importance": perm.feature_importances_,
    "std": perm.feature_importances_std_,
    },
    index = X_test.columns.to_list()
)

```

Following up, we can plot the feature importance:

```

importance["importance"].sort_values(
    ascending=False
).plot(figsize=(15, 5), kind="bar", yerr=importance["std"])

plt.title("Permutation feature importance")
plt.ylabel("Performance drop (R2)")
plt.show()

![Bar plot showing permutation feature importance calculated with ELI5]({{ site.baseurl }}/assets/images/posts/permutation-feature-importance/pfi-eli-1024x451.png)
```

To select features, we use the `perm` class within `SelectFromModel` as follows:

```
sel = SelectFromModel(
    perm,
    threshold=0.01, # select features above this value
    prefit=True,
)

selected_vars = X_train.columns[sel.get_support()]
X_train_t = pd.DataFrame(sel.transform(X_train), columns=selected_vars, index=X_train.index)
X_test_t = pd.DataFrame(sel.transform(X_test), columns=selected_vars, index=X_test.index)
```

With `X_train_t.head()` we see the final dataset, that contains only 6 of the original features.

### Permutation Feature Importance with Feature-engine

[Feature-engine](https://feature-engine.trainindata.com/en/latest/user_guide/selection/SelectByShuffling.html) incorporates feature inspection and selection in one single transformer:

```
from feature_engine.selection import SelectByShuffling
```

To determine the feature importance based on the random forest model we do like this:

```
sel = SelectByShuffling(
    variables=None, # automatically examine all numerical variables
    estimator=rf, # the ML model
    scoring='r2', # the metric to evaluate
    threshold=0.01,# the maximum performance drop allowed to select the feature
    cv=3, # cross-validation
    random_state=1 # seed
)
sel.fit(X_train, y_train)
```

To plot the feature importance we execute the following:

```
pd.Series(sel.performance_drifts_).sort_values(
    ascending=False).plot.bar(figsize=(15, 5))

plt.title("Permutation feature importance")
plt.ylabel("Performance drop (R2)")
plt.show()
```

![Bar plot showing permutation feature importance determined with feature-engine]({{ site.baseurl }}/assets/images/posts/permutation-feature-importance/pfi-fe-1024x440.png)

And to select the features, we simply do like this:

```
X_train_t = sel.transform(X_train)
X_test_t = sel.transform(X_test)
```

Simple!

### Considerations and Caveats

It’s crucial to note that the importance derived from permutation feature importance is relative to the model’s performance. A poorly performing model may assign low importance to a feature, while a well-performing model may emphasize its importance. Therefore, always assess the model’s overall predictive power before interpreting feature importance.

The importance value is determined by the drop in performance elicited by each feature. However, it’s worth noting that the aggregated importance of individual features may not be additive when evaluated separately.

Shuffling can be computationally expensive, especially when dealing with numerous features or repeated shuffling.

Randomness, while not a significant concern for highly important features, it becomes critical for borderline important ones. To address this issue, it is critical to obtain a measure of the dispersion of the importance, like the standard deviation.

Permutation feature importance is susceptible to correlation among features. Correlated features may exhibit reduced importance individually, leading to potential misinterpretation. Combining permutation feature importance with recursive feature elimination can mitigate this issue, emphasizing the importance of each feature in isolation.

In summary, when applying permutation feature importance, evaluate features in the test set, be mindful of the non-additive nature of feature importance, consider computational costs, and address issues related to randomness and correlation. This comprehensive approach ensures a more accurate understanding of feature importance in machine learning models.
