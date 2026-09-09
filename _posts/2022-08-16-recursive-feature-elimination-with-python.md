---
layout: post
title: "Recursive feature elimination with Python"
author: sole
description: "Recursive feature elimination is the process of selecting features sequentially, in which features are removed one at a time or a few at a time."
excerpt: "Recursive feature elimination is the process of selecting features sequentially, in which features are removed one at a time or a few at a time."
categories: [Feature Selection, Machine Learning]
image: assets/images/posts/recursive-feature-elimination-with-python/rfe_python.gif
---

Recursive feature elimination (RFE) is the process of selecting features sequentially, in which features are removed one at a time, or a few at a time, iteration after iteration.

Given a machine learning model, the goal of recursive feature elimination is to select features by recursively considering smaller and smaller sets of features.

In RFE, first an estimator is trained using all features, and then the importance of each variable is obtained. If using Scikit-learn, these would be obtained either from the coefficients of a linear regression model (`coef_`) or the importance derived from decision trees (`feature_importances_`). Then, the least important feature or group of features would be removed, and a new machine learning model would be trained using the remaining features.

#### RFE initial steps:

1. Train a machine learning model
2. Derive feature importance
3. Remove least important feature(s)
4. Re-train the machine learning model on the remaining features

![Diagram showing the steps involved in recursive feature elimination.]({{ site.baseurl }}/assets/images/posts/recursive-feature-elimination-with-python/initialsteps.png)

After this point, there are 2 different implementations of RFE. One by Scikit-learn and one by Feature-engine. Let’s see what they are about.

*For tutorials on feature selection, check out our course [Feature Selection for Machine Learning](https://www.trainindata.com/p/feature-selection-for-machine-learning) or our book [Feature Selection in Machine Learning with Python](https://www.trainindata.com/p/feature-selection-in-machine-learning-book).*

[![Feature Selection for Machine Learning, online course.]({{ site.baseurl }}/assets/images/posts/feature-selection-with-filter-methods/feature-selection-course-1024x576.png)](https://www.trainindata.com/p/feature-selection-for-machine-learning)

## RFE in Scikit-learn

In the Scikit-learn implementation, features continue to be removed based on feature importance. That means that steps 2-4 are repeated until a stopping criteria is reached. In Scikit-learn, the stopping criteria is an arbitrary number of final features.

#### RFE in Scikit-learn

1. Train a machine learning model
2. Derive feature importance
3. Remove least important feature(s)
4. Re-train the machine learning model on the remaining features
5. Repeat 2 to 4 until the desired number of features is reached

![Diagram showing the recursive feature elimination implementation of Scikit-learn]({{ site.baseurl }}/assets/images/posts/recursive-feature-elimination-with-python/rfesklearn.png)

This implementation of recursive feature elimination accommodates the changes in feature importance induced by changing feature subsets. This is important because when there is colinearity, both the coefficients of linear models and the importance derived from trees are affected. Thus, by removing features recursively, if one of the colinear features is removed, the coefficient or feature importance of the remaining feature(s) increases.

Because several machine learning models are trained, one at each iteration, the RFE algorithm is computationally costly. To accelerate the search, we could remove several features at each iteration, at the expense of model performance degradation.

### How does RFE by feature importance compare to embedded feature selection methods?

When using [Lasso regularization](https://www.blog.trainindata.com/lasso-feature-selection-with-python/) or just training decision tree based models, we automatically obtain feature importance. And we could use this importance to select the features straightaway. So how does RFE compare to these selection methods?

Compared to selecting features based on feature importance, this method has the advantage that it considers the re-adjustments in importance after a feature or a small subset of features is removed. Thus, it is better suited to handling features that are highly correlated. On the downside, as it trains several predictive models, it is more computationally costly than embedded methods.

![Diagram comparing recursive feature elimination to embedded methods for feature selection. ]({{ site.baseurl }}/assets/images/posts/recursive-feature-elimination-with-python/rfevsembedded.png)

### Python implementation

Recursive feature elimination is available in Scikit-learn through the [RFE](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html) or [RFECV](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html#sklearn.feature_selection.RFECV) classes. Let’s see how we can carry out RFE with Python. In particular, we can use these classes with any algorithm that returns the attributes `coef_` or `feature_importance_`, which means that it can be used with linear and logistic regression, all decision tree-based models, and SVMs.

We will carry out recursive feature elimination based on feature importance utilizing the breast cancer dataset.

Let’s import the libraries, functions, and classes:

```
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.feature_selection import RFE
```

Let’s load the dataset and separate it into a training and testing set:

```
breast_cancer = load_breast_cancer()
X = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
y = breast_cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
```

Let’s carry out the recursive feature elimination by removing 2 features at each iteration and stopping the search when there are 8 features remaining in the subset:

```
rfe_method = RFE(
    RandomForestClassifier(n_estimators=10, random_state=10),
    n_features_to_select=8,
    step=2,
)

rfe_method.fit(X_train, y_train)
```

By executing `X_train.columns[(rfe_method.get_support())]` we return the selected variables:

```
Index(['mean area', 'mean concave points', 'area error', 'worst texture',
       'worst perimeter', 'worst area', 'worst compactness',
       'worst concave points'],
      dtype='object')
```

We can remove the features from the training set and testing set like this:

```
X_train_selected = sel_.transform(X_train)
X_test_selected = sel_.transform(X_test)
```

And that’s it. Now we have selected features recursively based on the importance derived from random forests.

**Note**

In the original description of RFE, the authors propose the use of recursive feature elimination based on the importance derived from support vector machines (SVMs). But this algorithm can be extended, as we have seen, to those algorithms that support the estimation of feature importance intrinsically, like decision tree-based algorithms.

## RFE in Feature-engine

There is an alternative implementation of RFE in which the features are ranked based on the performance of the machine learning model trained on the progressively smaller feature subsets.

In the [Feature-engine implementation](https://feature-engine.readthedocs.io/en/latest/user_guide/selection/RecursiveFeatureElimination.html), features are removed based on a drop in the performance of the classifier or regression model. The performance is measured by any model performance metric of interest.

### RFE in Feature-engine:

1. Train a machine learning model
2. Derive feature importance
3. Remove least important feature(s)
4. Re-train the machine learning model on the remaining features
5. Calculate the change in performance of the second model with respect to the previous one.
6. If the performance decreases beyond a certain threshold, keep the feature from 3.
7. Repeat steps 3-6 until all the features have been evaluated.

There are a number of differences between Feature-engine’s and Scikit-learn’s implementations.

1. In Scikit-learn, features are removed based on their importance. In Feature-engine they are removed based on the model performance (i.e., accuracy, MSE, etc).
2. In Scikit-learn, the least important feature would be removed at the end of each round. In Feature-engine, the feature would be removed if the performance did not drop. Otherwise, it would be retained, therefore providing an additional test for the feature’s importance.
3. In Scikit-learn, the procedure continues until a certain number of features is reached. In Feature-engine, the procedure continues until all features have been evaluated.

![Diagram comparing the recursive feature elimination implementations between Scikit-learn and Feature-engine]({{ site.baseurl }}/assets/images/posts/recursive-feature-elimination-with-python/comparison.png)

In short, in the Scikit-learn RFE implementation, we would remove the feature of the least importance. With Feature-engine, we would remove a feature if its removal reduced the performance of the machine learning model. In Feature-engine, the feature importance is used only to determine the order in which the features will be eliminated and therefore evaluated.

### Python implementation

We will select features with recursive feature elimination using a regression data set. The procedure for classification is very similar. We just need to train a classifier instead and choose a suitable performance metric.

Let’s import the libraries, functions, and classes:

```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from feature_engine.selection import RecursiveFeatureElimination
```

Let’s load the California housing data set and separate it into a training and a testing set:

```
X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
```

Let’s set up a gradient boosting machine for regression. We will use it to evaluate the features during the search.

```
model = GradientBoostingRegressor(
    n_estimators=5,
    random_state=10,
)
```

Let’s set up a recursive feature elimination search, that uses the previous gradient boosting machine and the R2 to evaluate the feature subsets using 2-fold cross-validation. Cross-validation helps improve the generalization of the feature subset.

We will remove those features that cause a decrease in R2 greater than 0.001. With `fit()`, we start the search:

```
RFE_model = RecursiveFeatureElimination(
    estimator = model, # the ML model
    scoring = 'r2',
    threshold = 0.001,
    cv=2,
)

RFE_model.fit(X_train, y_train)
```

The `RecursiveFeatureElimination()` stores a few parameters learned during the search. By executing `RFE_model.initial_model_performance_,` we obtain the performance of the gradient boosting machine trained using the entire dataset: `0.3639885983830904`.

Let’s now carry out some visualization to understand what this method was doing. The `RecursiveFeatureElimination()` also stores the importance of the features derived from the preceding gradient boosting machine:

```
RFE_model.feature_importances_.plot.bar(figsize=(10, 5))
plt.ylabel('Feature importance')
plt.title('Feature importance derived from the GBM')
plt.show()
```

The output of the preceding code block shows the importance of the features derived from the regression model:

![Barplot showing mportance of the features derived from the regression model]({{ site.baseurl }}/assets/images/posts/recursive-feature-elimination-with-python/fig1.png)

We can also plot the changes in the R2 caused by the elimination of features:

```
pd.Series(RFE_model.performance_drifts_).plot.bar(figsize=(10, 5))
plt.title('Performance change after removing features recursively')
plt.ylabel('R2 change when feature was removed')
plt.show()
```

In the following plot, we see the change in the value of R2 caused by the elimination of each feature:

![Barplot showing the change in the value of R2 caused by the elimination of each feature]({{ site.baseurl }}/assets/images/posts/recursive-feature-elimination-with-python/fig2.png)

By executing `sel.features_to_drop_` we obtain the features that will be removed from the data: `['HouseAge', 'AveBedrms', 'Population', 'Latitude', 'Longitude']`.

Let’s now reduce the dataset to the selected variables:

```
X_train_t = RFE_model.transform(X_train)
X_test_t = RFE_model.transform(X_test)
```

And that’s it, we have now selected features recursively based on model performance.

## RFE or other selection methods?

OK, but why is this better than other feature selection methods? To answer this question, let’s make a small recap of the feature selection procedures that are available.

## Feature selection algorithms

Feature selection is the process of selecting the best subset of features from the available data to train classifiers or regression models. Feature selection helps produce simpler and faster models and can help prevent overfitting. Feature selection is an important aspect of any data science project.

![Illustration for feature engineering]({{ site.baseurl }}/assets/images/posts/recursive-feature-elimination-with-python/cover.gif)

A number of feature selection methods have been developed over the years to select relevant features in machine learning. Some methods run fast but do not consider feature interactions. Some other methods evaluate subsets of features but are computationally very costly. Thus, different models serve different situations better, depending on the dimension of the feature space, the characteristics of the machine learning algorithms, and the available computing resources.

The best feature selection algorithm, in theory, would be one that evaluates all possible feature subsets. In practice, an exhaustive evaluation of all feature subsets for large numbers of features is impossible due to the combinatorial explosion of the number of subsets.

Univariate feature selection methods select the variables that individually make good enough predictions. These methods include statistical tests like correlation, ANOVA, chi-squared and [mutual information](https://www.blog.trainindata.com/mutual-information-with-python/). They eliminate variables that are useless for discrimination (noise), but they do not yield compact feature sets because features are redundant. Moreover, complementary features that individually do not offer good predictive value would be removed and therefore missed.

There are alternative feature ranking methods that are baked into the machine learning induction procedure that consider the interaction between features. For example, in linear regression models, the coefficients that multiply each variable indicate how much that variable contributes to the outcome. The bigger the coefficient, the higher the feature contribution and therefore its importance.

Decision tree-based algorithms also assign importance to the features during the induction. The importance of the feature is the normalized information gain within the tree, or across all trees if training ensembles. These feature importance metrics, like the coefficients of a linear regression or the importance derived by trees, do consider the interaction between features and, as such, would not remove features whose value individually is not high but in combination with other features helps predict the target better.

The problem with the embedded feature selection methods is that the importance assigned to the features is influenced by correlation among features. It is well established that multi-colinearity influences the coefficients of linear models. And it also influences the importance derived from decision tree-based models. For example, in a decision tree, if 2 features are identical or highly co-linear, any of the 2 can be taken to make a split at a certain node, and thus its importance will be higher than that of the second feature. And this is just random. If we create an ensemble, there will be decision trees that use the first feature and decision trees that use the second feature, and thus, the importance of each feature would be half of what it would be if each feature was by itself in the data.

Thus, in datasets where there is high feature redundancy, a better option is to carry out recursive feature elimination.

## Feature selection and deep learning

Should we select features for neural networks in artificial intelligence? This is a question that I get often. The thing is, neural networks are better than traditional machine learning models only when using big data. That is, when having a lot of observations and a lot of variables. And the beauty of neural networks is that they will figure out a way to match patterns from the original variables to the target.

Thus, although my experience with deep learning is limited, I would say that recursive feature elimination, like all other feature selection techniques, aims to reduce the feature space necessary to train traditional machine learning models, and therefore make them faster and easier to understand. Qualities that are in demand when using the models within organizations are on demand.

## Wrap-up

If you made it this far, thank you for reading.

For tutorials on feature selection, check out our course [Feature Selection for Machine Learning](https://www.trainindata.com/p/feature-selection-for-machine-learning) or our book [Feature Selection in Machine Learning with Python](https://www.trainindata.com/p/feature-selection-in-machine-learning-book).

[![Feature Selection in Machine Learning with Python, book cover]({{ site.baseurl }}/assets/images/posts/machine-learning-books-for-beginners/book-cover-1024x1024.png)](https://www.trainindata.com/p/feature-selection-in-machine-learning-book)

## References

- Guyon, et. al., Gene selection for cancer classification using support vector machines, Mach. Learn., 46, 389–422, 2002.
