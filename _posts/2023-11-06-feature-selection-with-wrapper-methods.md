---
layout: post
title: "Feature Selection with Wrapper Methods in Python"
author: sole
description: "Learn what wrapper methods for feature selection are, their advantages and limitations, and how to implement them in Python."
excerpt: "Learn what wrapper methods for feature selection are, their advantages and limitations, and how to implement them in Python."
categories: [Feature Selection, Machine Learning]
image: assets/images/posts/feature-selection-with-wrapper-methods/wrapper-methods-feature-selection.gif
---

Imagine you’re preparing a delicious cake for Christmas. You have a pantry full of ingredients, but you don’t know the perfect recipe. Would you throw everything into the pot? Probably not! You’d select the ingredients that make your dish taste the best. Similarly, in the world of data science and machine learning, [feature selection](https://www.blog.trainindata.com/feature-selection-machine-learning-with-python/) is the art of picking the most relevant features that will help you build an efficient and effective model.

The three important feature selection methods you should know as a data scientist are filter, wrapper, and embedded methods.

In filter methods, all variables are ranked based on their impact on the output/target. The top ones are chosen to build a model. Some examples include the chi-square test and evaluating correlation coefficients.

In the wrapper methods, different subsets of features are evaluated by training a model for each subset. We then compare the performance and choose the right combination.

In embedded methods, a model is trained on all the features and it learns the importance of each feature as it gets trained. Lasso regularization for linear models, and tree derived feature importance are the commonly used embedded methods.

To learn more about feature selection methods, check out our [Feature Selection for Machine Learning course](https://www.trainindata.com/p/feature-selection-for-machine-learning) and [Feature Selection in Machine Learning book](https://www.trainindata.com/p/feature-selection-in-machine-learning-book).

[![Feature Selection for Machine Learning, online course.]({{ site.baseurl }}/assets/images/posts/feature-selection-with-filter-methods/feature-selection-course-1024x576.png)](https://www.trainindata.com/p/feature-selection-for-machine-learning)

While each method has its own merits, today, our spotlight is on **Wrapper Methods**. Why? Because they provide a balance between relying purely on statistical measures (like filter methods) and completely depending on specific algorithms (like embedded methods). Grab a cup of coffee and gear up to master this feature selection technique!

## What are wrapper methods?

Let’s go back to the task of choosing ingredients for your cake. Wouldn’t you want to ensure that all the chosen ingredients mix well apart from their individual tastes? You could try out baking multiple cakes with different combinations of ingredients and choose the best! That’s exactly what is done in wrapper methods.

Wrapper methods wrap the feature selection procedure around a predictive model. The selection process of this technique can be summarized into 3 broad steps:

1. Create all possible feature subsets.
2. Train a machine learning model on each feature subset.
3. Evaluate the results of all models and choose the best.

The machine learning model used is usually a regression or a classification model depending on the data type. For example, decision tree-based algorithms like Random forest, and SVM (Support Vector Machine) can be used.

For evaluation of the model, you need to choose a relevant metric. For regression tasks, it could be R-squared or RMSE. For classification tasks, you can use the F1 score from the confusion matrix, precision, and recall.

The most challenging part of this feature selection technique is step 1: creating the subsets.

Why?

For a dataset with 5 features, the number of possible feature combinations would be 32. But, datasets we come across in real life are much bigger than that. For a dataset with 200 variables, the subsets possible are around 1.1259 trillion! (For a set of n features, the number of possible subsets is `2^n` ).

The computational cost of assessing 1 trillion feature subsets is extremely high, if not prohibitive! Lucky for us, there are a few alternative methods that can be used to assess a smaller number of feature combinations. In the next sections, I’ll discuss these approaches and show how you can implement them using open-source Python libraries.

## Exhaustive search

The first and simplest method is the Exhaustive Feature Selection (EFS) algorithm. As the name suggests, we perform an exhaustive search across all possible subsets to find the best set of features. You can use this method when the number of features is relatively small.

As an example, if the dataset contains 4 features, the EFS algorithm will evaluate the following feature subsets:

- All possible combinations of 1 feature.
- All possible combinations of 2 features.
- All possible combinations of 3 features.
- All 4 features.

![Table showing all possible feature combinations created from a data set with 4 features: A, B, C and D.]({{ site.baseurl }}/assets/images/posts/feature-selection-with-wrapper-methods/all-possible-feature-combinations-1024x726.png)

All possible feature combinations created from a dataset with 4 features: A, B, C and D.

The procedure will train a machine learning model for each one of the feature subsets with cross-validation and then determine the model’s performance. We use cross-validation to avoid overfitting of the model.

The EFS algorithm will evaluate `2^4` feature combinations, that is, 16 feature subsets. This means, that it will train 16 machine learning models, with cross-validation. Thus, it is easy to see how this exercise becomes computationally costly with bigger datasets.

You can perform EFS using the open-source implementation in the [MLXtend package](https://rasbt.github.io/mlxtend/user_guide/feature_selection/ExhaustiveFeatureSelector/). To show you an example in this tutorial, I’ll be using the breast cancer dataset from sklearn.

### EFS with Python

The first step would be to import the necessary modules from Pandas and Sklearn. The feature selector can be imported from the MLXtend library.

```

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
```

The next step is to load the breast cancer dataset and split it into a training and a testing set:

```
breast_cancer = load_breast_cancer()
X = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
y = breast_cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
```

This dataset contains 30 input variables.

Before we proceed to set up an exhaustive search using MLXtend, we need to define the following parameters:

1. **estimator**: You need to pass the machine learning algorithm that will be wrapped by the search.
2. **scoring:** The performance metric to evaluate. In this example, I’ll be using the roc-auc.
3. **cv:** The parameter K in K-fold cross-validation
4. **min_features:** The number of minimum features a subset can have.
5. **max_features:** The maximum number of features a subset can have.

The `min_features` and `max_features` parameters will help you reduce the feature subset space. For example, a data scientist would be sure that using just 1-2 features out of 30 would not give a good performance. Limiting the subset space speeds up the search search process.

Now, let’s set up the EFS algorithm. By calling `fit()`, we trigger the search:

```

efs = EFS(
    estimator=RandomForestClassifier(
    n_estimators=3,
    random_state=0),
    min_features=1,
    max_features=4,
    scoring='roc_auc',
    cv=2,
)

efs = efs.fit(X_train, y_train)
```

Once the fitting is done, we can find the optimal subset of predictors by executing:

`efs.best_feature_names_`

which returns the following features:

`('compactness error', 'worst area', 'worst concavity', and 'worst symmetry')`.

In the attribute `efs.subsets_`, the transformer stores the subsets evaluated, the score in each cross-validation fold, and the mean score across all folds.

For this particular dataset, and with the search defined in the previous code block, the EFS evaluated 31930 feature subsets, which can be obtained by executing `len(efs.subsets_.keys())`.

Let’s print out the details of a random subset of features:

```
efs.subsets_[60]
```

The output would be something like this:

```
{'feature_idx': (1, 3),
 'cv_scores': array([0.86057056, 0.90953947]),
 'avg_score': 0.8850550192906221,
 'feature_names': ('mean texture', 'mean area')}
```

This subset consists of 2 features, mean texture, and mean area, and we see the performance of the model trained using these features in each cross-validation fold as well as the average.

Finally, we can go ahead and transform the data:

```
X_train_t = efs.transform(X_train)

X_test_t = efs.transform(X_test)

```

The final result consists of Numpy arrays containing only the 4 selected features. Here’s a sample output of how the transformed `X_test_t` would look:

```
array([[2.265e-02, 8.444e+02, 5.106e-01, 3.585e-01],
        [8.082e-03, 6.329e+02, 1.390e-01, 2.444e-01],
        [9.238e-03, 6.889e+02, 6.260e-02, 2.136e-01],
        [1.377e-02, 8.197e+02, 1.565e-01, 2.636e-01],
        ...
        [2.277e-02, 2.232e+03, 6.810e-01, 3.643e-01]])
```

We’ve now conducted an exhaustive search. Note that we have 4 variables as a result from the feature extraction process as we specified it as the maximum number.

## Forward feature selection

Let’s move on to the next feature selection technique in wrapper methods: The forward feature selection. FFS is a sequential method. Here, we start with a single variable and keep adding more to get the best performance.

Here’s how the technique works:

1. Choose a machine-learning model for your dataset
2. For each feature in your data, train a machine learning model
3. Select the feature that gave the best-performing classifier or regression model.
4. For the next step:
5. Combine the selected feature(s) from the previous step with each of the remaining variables.
6. Evaluate models for all these combinations.
7. Choose the combination that gives the best-performing algorithm.
8. Continue this process, adding one feature at a time.
9. Stop when a predefined stopping criterion is met.

In each iteration, a new feature is added to the model along with the previous ones.

Forward feature selection is more efficient than the exhaustive search method.

How?

Forward feature selection has the advantage that, by starting with smaller feature subsets, it is more computationally efficient than other wrapper methods. However, for this same reason, it does not contemplate feature interactions. Or at least not until sufficient features have been added to the subset.

An important step in the forward feature selection is to define a stopping criteria. The most obvious stopping condition is when the performance of the classifier or regression model does not improve beyond a certain threshold. This has the advantage of focusing the search on performance. On the downside, the threshold for improvement is arbitrarily set by the user. Alternatively, we can stop the search after a certain number of features have been selected.

Let’s now implement forward feature selection with Scikit-learn and MLXtend. The 2 Python implementations are very similar. Both offer a number of features as stopping criteria. Scikit-learn also offers a threshold on performance improvement as a method to stop the search.

I’ll be using the same breast cancer dataset throughout this tutorial. You can import pandas, sklearn modules, load the datasets, split them into train and test sets exactly as we did in the previous section. Now, forward feature selection can be implemented through both sklearn and mlxtend.

### FFS with sklearn

Import the `SequentialFeatureSelector` class from Scikit-learn as shown below.

```
from sklearn.feature_selection import SequentialFeatureSelector as SFS
```

Let’s use a Random forest classifier to train models and a three-fold cross validation to avoid overfitting. The metric for evaluation is ROC-AUC. The `tol` parameter is used to define the threshold at which the search will stop.

In below code, we set it to 0.001, which means that if the metric increases above 0.001, the search would end. Let’s fit the feature selector :

```
sfs = SFS(
    estimator=RandomForestClassifier(n_estimators=5, random_state=0),
    n_features_to_select='auto',
    tol=0.001,
    direction='forward',
    scoring='roc_auc',
    cv=3,
)

sfs = sfs.fit(X_train, y_train)
```

Next, we can print out the names of the selected variables by executing:

`sfs.get_feature_names_out()`

which returns the following variables:

```
array(['mean smoothness', 'mean concavity', 'worst texture', 'worst perimeter', 'worst concavity'], dtype=object)
```

Finally, we can reduce the data to the selected variables with `tansform()`:

```
X_train_t = sfs.transform(X_train)
X_test_t = sfs.transform(X_test)
```

This will return Numpy arrays containing the selected features only.

Next, let’s carry out forward search with MLXtend.

### **FFS with MLXtend**

First, import the SFS class from MLXtend package. In this method, rather than setting a threshold as stopping criteria, we will set the number of features we want at the end.

```
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
```

To carry out the search, we will use Random forests and the roc-auc as performance metric, with 3-fold cross validation, and we will stop the search after we identify the best 10 features.

```
sfs = SFS(
    estimator=RandomForestClassifier(n_estimators=5, random_state=0),
    k_features=10,  # the number of features to retain
    forward=True, # the direction of  the search
    verbose=1,  # print out intermediate steps
    scoring='roc_auc',
    cv=3,
)

sfs = sfs.fit(X_train, y_train)
```

If you set `verbose=1`, MLXtend logs information about the search as it proceeds:

```
[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[Parallel(n_jobs=1)]: Done  30 out of  30 | elapsed:    0.9s finished
Features: 1/10[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[Parallel(n_jobs=1)]: Done  29 out of  29 | elapsed:    0.8s finished
Features: 2/10[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
......
Features: 8/10[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[Parallel(n_jobs=1)]: Done  22 out of  22 | elapsed:    0.7s finished
Features: 9/10[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[Parallel(n_jobs=1)]: Done  21 out of  21 | elapsed:    0.6s finished
Features: 10/10
```

After the search, we can find the selected features by executing:

`sfs.k_feature_names_`

which returns the following variables:

```
('mean perimeter',
 'mean compactness',
 'mean fractal dimension',
 'compactness error',
 'concave points error',
 'worst texture',
 'worst perimeter',
 'worst smoothness',
 'worst concave points',
 'worst symmetry')

```

Finally, we can reduce the data to the selected features:

```
X_train_t = sfs.transform(X_train)
X_test_t = sfs.transform(X_test)
```

The resulting datasets are also Numpy arrays containing only the selected variables. You can print out `X_test_t` and check that it has only the selection top 10 combination of features.

## Backward feature elimination

Let’s now look at our final wrapper method: backward elimination. As the name suggests, our approach here is the inverse of what we did in forward selection.

Here’s how this approach works:

1. Start with the entire set of features.
2. Fit an ML model using all features and note its performance.
3. For the next step:
4. Generate all possible feature subsets by excluding one feature at a time.
5. Train a machine learning model for each subset.
6. Determine the performance of each model.
7. Remove the feature that results in the model with the lowest performance.
8. Repeat the process, eliminating one feature at a time.
9. Continue until a predefined stopping criterion is met.

In short, the approach performs recursive feature elimination in each iteration until the criteria are met. The stopping criteria should be defined, keeping both the performance and optimization of the search in mind.

What is the advantage of backward feature selection?

It can capture feature interactions as it starts training with the complete set of features. This is something lacking in the previous method.

But, there are two sides to every coin. As it starts with complete feature set, it is computationally less efficient than forward feature selection

Backward feature selection also needs a criteria to stop the search. This criteria can be met when the performance of the classifier or regression model does not degrade beyond a certain threshold. This has the advantage of focusing the search on performance. On the downside, the threshold for model degradation is arbitrarily set by the user. Alternatively, we can stop the search after a certain number of features have been selected.

In the coming paragraphs, we will implement backward feature selection with Scikit-learn and MLXtend. We will use the same classes as for forward feature selection; we just need to change the direction of the search.

### Backward feature elimination with sklearn

I will use the California housing data. The target variable is continuous, hence, we’ll train models like linear regression or random forests for regression. If you have custom data, you can read the CSV file into a pandas data frame and follow the below steps.

Let’s begin by loading the libraries, functions, and classes:

```
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

```

Let’s load the data and separate it into a training and a testing sets;

```
X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
```

Next, import the `SequentialFeatureSelector` from Scikit-learn:

```
from sklearn.feature_selection import SequentialFeatureSelector as SFS
```

Now, let’s set up the search to wrap random forests and use the `R-square` metric to evaluate the feature subsets, utilizing 3-fold cross-validation. We’ll stop the search when the `R-square` does not decrease beyond 0.001.

With `fit()` we trigger the search:

```
sfs = SFS(
    estimator = RandomForestRegressor(n_estimators=5, random_state=10),
    n_features_to_select="auto"
    tol=0.001,
    direction='backward',
    scoring='r2',
    cv=3,
)

sfs = sfs.fit(X_train, y_train)
```

We can visualize the selected features by executing:

`sfs.get_feature_names_out()`

which returns the following variable names:

```
array(['MedInc', 'HouseAge', 'AveBedrms', 'AveOccup', 'Latitude',      'Longitude'], dtype=object)
```

Now we can reduce the dataset to the selected variables:

```
X_train_t = sfs.transform(X_train)
X_test_t = sfs.transform(X_test)
```

The result are Numpy arrays with the selected features.

### Backward Feature Selection with MLXtend

Let’s now carry out backward feature elimination with MLXtend. Let’s import the transformer:

```
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
```

Let’s set up the search around random forests, using the `R-squared` to evaluate the feature subsets, and utilize 3-fold cross-validation. We’ll stop when we reach a subset of 5 features.

With `fit()` we trigger the search:

```
sfs = SFS(
    estimator=RandomForestRegressor(
    n_estimators=5, random_state=10),
    k_features=5,
    forward=False,
    verbose=1,
    scoring='r2',
    cv=3,
)

sfs = sfs.fit(X_train, y_train)
```

MLXtend outputs logging information as it proceeds with the search:

```
[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[Parallel(n_jobs=1)]: Done   8 out of   8 | elapsed:    6.8s finished
Features: 7/5[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
......
Features: 5/5
```

We can visualize the selected variables by executing:

`sfs.k_feature_names_`

which returns:

`('MedInc', 'HouseAge', 'AveOccup', 'Latitude', 'Longitude')`.

After that, we can reduce the data to the selected variables:

```
X_train_t = sfs.transform(X_train)
X_test_t = sfs.transform(X_test)
```

MLXtend also returns Numpy arrays with the selected features.

# Conclusion

I hope this tutorial gave a comprehensive understanding of the wrapper feature selection techniques.

Feature selection and feature engineering are crucial steps in any machine learning project. As the saying goes, “Garbage in, Garbage out!” Ensure that your feature extraction and data preprocessing is effective, but also computationally efficient.

While an exhaustive search is simple, it is feasible only for a small set of features. Forward and backward selection are more computationally efficient alternatives, but they can also be too costly with bigger training data sets.

In these cases, you can begin to narrow down the feature space with alternative feature selection algorithms, like filter methods and embedded methods.

To learn more about feature selection methods, check out our [Feature Selection for Machine Learning course](https://www.trainindata.com/p/feature-selection-for-machine-learning) and [Feature Selection in Machine Learning book](https://www.trainindata.com/p/feature-selection-in-machine-learning-book).

[![Feature Selection in Machine Learning with Python, book cover]({{ site.baseurl }}/assets/images/posts/machine-learning-books-for-beginners/book-cover-1024x1024.png)](https://www.trainindata.com/p/feature-selection-in-machine-learning-book)
