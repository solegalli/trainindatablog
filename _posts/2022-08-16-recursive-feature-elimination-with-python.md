---
layout: post
title:  "Recursive feature elimination with Python"
author: sole
categories: [ Feature selection, Python, Machine learning ]
image: assets/images/posts/rfe/rfe_python.png
---

Recursive feature elimination (RFE) is the process of selecting features sequentially, in which 
features are removed one at a time, round after round. 

OK, but why is this better than other feature selection methods? To answer this question, 
let’s make a small recap of the feature selection procedures that are available.


## Feature selection algorithms

Feature selection is the process of selecting the best subset of features from the available 
data to train classifiers or regression models. Feature selection helps produce simpler 
and faster models and can help prevent overfitting. Feature selection is an important 
aspect of any data science project.

![feature selection in machine learning]({{ site.baseurl }}/assets/images/posts/rfe/fe-gif.gif)   

A number of feature selection methods have been developed over the years to select relevant 
features in machine learning. Some methods run fast but do not consider feature 
interactions. Some other methods evaluate subsets of features but are computationally very 
costly. Thus, different models serve different situations better, depending on the dimension 
of the feature space, the characteristics of the machine learning algorithms, and the 
available computing resources.

The best feature selection algorithm, in theory, would be one that evaluates all possible 
feature subsets. In practice, an exhaustive evaluation of all feature subsets for large 
numbers of features is impossible due to the combinatorial explosion of the number of subsets.

Univariate feature selection methods select the variables that individually make good 
enough predictions. These methods include statistical tests like correlation, ANOVA, 
chi-squared and [mutual information](https://www.blog.trainindata.com/mutual-information-with-python/). 
They eliminate variables that are useless for discrimination (noise), but they do not yield compact feature sets because features are 
redundant. Moreover, complementary features that individually do not offer good predictive 
value would be removed and therefore missed.

There are alternative feature ranking methods that are baked into the machine learning 
induction procedure that consider the interaction between features. For example, in linear 
regression models, the coefficients that multiply each variable indicate how much that 
variable contributes to the outcome. The bigger the coefficient, the higher the feature 
contribution and therefore its importance.

Decision tree-based algorithms also assign importance to the features during the induction. 
The importance of the feature is the normalized information gain within the tree, or across 
all trees if training ensembles. These feature importance metrics, like the coefficients 
of a linear regression or the importance derived by trees, do consider the interaction 
between features and, as such, would not remove features whose value individually is not 
high but in combination with other features helps predict the target better.

The problem with the embedded feature selection methods is that the importance assigned to 
the features is influenced by correlation among features. It is well established that 
multi-colinearity influences the coefficients of linear models. And it also influences the 
importance derived from decision tree-based models. For example, in a decision tree, if 2 
features are identical or highly co-linear, any of the 2 can be taken to make a split at a 
certain node, and thus its importance will be higher than that of the second feature. And 
this is just random. If we create an ensemble, there will be decision trees that use the 
first feature and decision trees that use the second feature, and thus, the importance of 
each feature would be half of what it would be if each feature was by itself in the data.

Thus, in datasets where there is high feature redundancy, a better option is to carry out 
recursive feature elimination.

*For tutorials on feature selection, check out our course 
[Feature Selection for Machine Learning](https://www.trainindata.com/p/feature-selection-for-machine-learning) or our 
book [Feature Selection in Machine Learning with Python](https://leanpub.com/feature-selection-in-machine-learning/).*


## RFE feature importance

RFE works as follows: the procedure:

1- Trains a machine learning model.

2- Derives the feature importance and ranks the features.

3- Removes the feature with the smallest importance.

4- Repeats 1 to 3 until a stopping criteria is met.

In short, features are ranked, and the feature with the least importance is removed. And 
the process is repeated over and over. Like this, recursive feature elimination accommodates 
the changes in feature importance induced by changing feature subsets.

Because several machine learning models are trained, one at each iteration, the algorithm is 
computationally costly. To accelerate the search, we could remove several features at each 
iteration, at the expense of model performance degradation.

Compared to selecting features based on feature importance, this method has the advantage 
that it considers the re-adjustments in importance after a feature or a small subset of 
features is removed. Thus, it is better suited to handling features that are highly correlated. 
On the downside, as it trains several predictive models, it is more computationally costly 
than embedded methods.

In the original description of RFE, the authors propose the use of recursive feature elimination 
based on the importance derived from support vector machines (SVMs). But this algorithm can be 
extended to those algorithms that support the estimation of feature importance intrinsically, 
like decision tree-based algorithms.

This implementation is available in Scikit-learn through the [RFE](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html) 
or [RFECV](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html) classes. In 
particular, we can use these classes with any algorithm that returns the attributes `coef` 
or `feature_importance`, which means that it can be used with linear and logistic regression, 
all decision tree-based models, and SVMs.

Let’s see how we can carry out RFE with Python. 

### Python implementation

We will carry out recursive feature elimination based on feature importance utilizing the 
breast cancer dataset.

Let's import the libraries, functions, and classes:

```
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
```

Let's load the dataset and separate it into a training and testing set:

```
breast_cancer = load_breast_cancer()
X = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
y = breast_cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
```

Let's carry out the recursive feature elimination by removing 2 features at each iteration 
and stopping the search when there are 8 features remaining in the subset:

```
rfe_method = RFE(
    RandomForestClassifier(n_estimators=10, random_state=10),
    n_features_to_select=8,
    step=2,
)

rfe_method.fit(X_train, y_train)
```

By executing X_train.columns[(rfe_method.get_support())] we return the selected variables:

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

And that's it. Now we have selected features recursively based on the importance derived 
from random forests.


## RFE based on model performance


There is an alternative implementation of RFE in which the features are ranked based on the 
performance of the machine learning model trained on the progressively smaller feature subsets.

The method consists of the following steps:

1. Create a machine learning model that makes use of all of the features.

2. Rank the features according to their importance derived from the model in 1.

3. Discard the least important feature and train a new machine learning model.

4. Calculate the change in performance of the second model with respect to the previous one. 

5. If the performance decreases beyond a certain threshold, keep the feature from 4.

6. Repeat steps 4-5 until all the features have been evaluated.


In the former RFE implementation, we would remove the feature of the least importance. 
Here, we would remove a feature that produces the machine learning model with the worst model 
performance if the model performance is worse than an arbitrary threshold. There is an 
implementation of this selection method in the open-source library 
[Feature-engine](https://feature-engine.readthedocs.io/en/latest/user_guide/selection/RecursiveFeatureElimination.html).


### Python implementation

We will select features with recursive feature elimination using a regression data set. 
The procedure for classification is very similar. We just need to train a classifier 
instead and choose a suitable performance metric.

Let's import the libraries, functions, and classes:

```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from feature_engine.selection import RecursiveFeatureElimination
```

Let's load the California housing data set and separate it into a training and a testing set:

```
X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
```

Let's set up a gradient boosting machine for regression. We will use it to evaluate the features during the search.

```
model = GradientBoostingRegressor(
    n_estimators=5,
    random_state=10,
)
```

Let's set up a recursive feature elimination search, that uses the previous gradient 
boosting machine and the R2 to evaluate the feature subsets using 2-fold cross-validation. 
Cross-validation helps improve the generalization of the feature subset.

We will remove those features that cause a decrease in R2 greater than 0.001. With `fit()`, 
we start the search:

```
RFE_model = RecursiveFeatureElimination(
    estimator = model, # the ML model
    scoring = 'r2', 
    threshold = 0.001, 
    cv=2,
)

RFE_model.fit(X_train, y_train)
```

The `RecursiveFeatureElimination()` stores a few parameters learned during the search. By 
executing `RFE_model.initial_model_performance_,` we obtain the performance of the gradient 
boosting machine trained using the entire dataset: `0.3639885983830904`.


Let’s now carry out some visualization to understand what this method was doing. The 
`RecursiveFeatureElimination()` also stores the importance of the features derived from 
the preceding gradient boosting machine:

```
RFE_model.feature_importances_.plot.bar(figsize=(10, 5))
plt.ylabel('Feature importance')
plt.title('Feature importance derived from the GBM')
plt.show()
```

The output of the preceding code block shows the importance of the features derived from 
the regression model:

![feature importance derived from the regression model]({{ site.baseurl }}/assets/images/posts/rfe/fig1.png)   

We can also plot the changes in the R2 caused by the elimination of features:

```
pd.Series(RFE_model.performance_drifts_).plot.bar(figsize=(10, 5))
plt.title('Performance change after removing features recursively')
plt.ylabel('R2 change when feature was removed')
plt.show()
```

In the following plot, we see the change in the value of R2 caused by the elimination of each feature:

![change in feature importance by recursive feature elimination]({{ site.baseurl }}/assets/images/posts/rfe/fig2.png)   
   
By executing `sel.features_to_drop_` we obtain the features that will be removed from 
the data: `['HouseAge', 'AveBedrms', 'Population', 'Latitude', 'Longitude']`.

Let's now reduce the dataset to the selected variables:

```
X_train_t = RFE_model.transform(X_train)
X_test_t = RFE_model.transform(X_test)
```

And that’s it, we have now selected features recursively based on model performance.

If you made it this far, thank you for reading. 

*Don't forget to check out our course [Feature Selection for Machine Learning](https://www.trainindata.com/p/feature-selection-for-machine-learning) and our 
book [Feature Selection in Machine Learning with Python](https://leanpub.com/feature-selection-in-machine-learning/).*


## References

- Guyon, et. al., Gene selection for cancer classification using support vector machines, Mach. Learn., 46, 389–422, 2002.