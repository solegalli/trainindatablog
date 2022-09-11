---
layout: post
title:  "Feature selection in machine learning with Python"
author: sole
categories: [ Feature selection, Python, Machine learning ]
image: assets/images/posts/fspython/cover.png
---

An essential step in any data science project is to select the most predictive variables. There are various methods of 
feature selection. Some scale well but consider only features individually, some are extremely computationally costly and 
thus applicable only to relatively small datasets, and some others fall somewhere in the middle.

There are also various implementations of feature selection algorithms in Python, utilizing open-source libraries. 
Covering all of them in an article, is almost impossible. Instead, in this blog, I will introduce the Python libraries 
for feature selection, highlight which selection methods are available in each of them, and then demo some of the feature selection implementations.

*For tutorials and step by step code implementations on additional feature selection methods, check out our course 
[Feature Selection for Machine Learning](https://www.trainindata.com/p/feature-selection-for-machine-learning) or our 
book [Feature Selection in Machine Learning with Python](https://leanpub.com/feature-selection-in-machine-learning/)*.

You may also like my talk at DataTalks.Club:

<p><iframe style="width:100%;" height="315" src="https://www.youtube.com/watch?v=blvmNWbcPDo" frameborder="0" allowfullscreen></iframe></p>

So how can we do feature selection in Python?

Before we jump onto the demos, let’s do a short recap on feature selection and the different methods that we can use to 
find the best feature subsets.


## Feature selection methods

In feature selection, we select a subset of features from the data set to train machine learning algorithms. Feature 
selection techniques differ from dimensionality reduction in that they do not alter the original representation of the 
variables but merely select a smaller set of features.

By reducing the number of features, we can improve the performance of the machine learning models (i.e., avoid overfitting), 
while reducing training time and creating more interpretable machine learning models.

Feature selection methods have been traditionally grouped into filter methods, wrapper methods, and embedded methods.

![feature selection methods]({{ site.baseurl }}/assets/images/posts/fspython/ch1-fig3.png)
**Image taken from [Feature Selection in Machine Learning with Python](https://leanpub.com/feature-selection-in-machine-learning/)**

Filter methods select the best features based on the feature characteristics, ignoring their interaction with the machine 
learning model. They rank the features and then select the top-ranking ones. Ranking methods normally use statistical 
tests like chi-square, ANOVA, correlation, and mutual information.

Wrapper methods wrap the search for the most relevant features around a predictive model. They generate multiple feature 
subsets and then evaluate their performance based on the classifier or regression model. The selected features are those 
from the subset that returned the best performing model.

Embedded methods "embed" the selection procedure in the training of the predictive model. Lasso and feature importance 
from decision trees are the classical examples of embedded methods. The coefficients of linear models can also be used 
to select important features.

Feature selection and feature engineering are widely used in data science during the preprocessing of the data. So how 
can we do that in Python?


## Python libraries for feature selection

There are 3 Python libraries with feature selection modules: Scikit-learn, MLXtend and Feature-engine.

Scikit-learn contains algorithms for filter methods, wrapper methods and embedded methods, including recursive feature elimination.

MLXtend contains transformers to implement forward, backward and exhaustive search.

Feature-engine contains alternative feature selection methods based on machine learning model performance, feature shuffling 
and also feature selection techniques that support categorical variables.

![feature selection methods]({{ site.baseurl }}/assets/images/posts/fspython/table.png)
**Feature selection implementations by the different libraries.**

In this article, we will implement various feature selection techniques with Scikit-learn and Feature-engine.

### Feature selection with Scikit-learn

Scikit-learn contains algorithms for filter methods, wrapper methods and embedded methods, including recursive feature 
elimination. Among the filter methods, we can select features using their variance or based on ANOVA. Let’s explore these procedures.


#### Variance

With Scikit-learn, we can remove irrelevant features by looking at feature variability. Features whose standard 
deviation is zero are constant and can be removed. In this example, we will create a toy dataset with 3 constant 
variables, and then we will remove them with Scikit-learn.

```
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.feature_selection import VarianceThreshold

# Toy dataset with redundant and constant features
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_classes=2,
    random_state=10,
)

X = pd.DataFrame(X)

# Add constant features
X[[0, 5, 9]] = 1

# To remove constant features
sel = VarianceThreshold(threshold=0)

# fit finds the features with zero variance
X_t = sel.fit_transform(X)  
```

`X_t` contains predictors whose variability is greater than 0.


#### Chi-square test

The chi-square test is suitable for selecting categorical variables when the target variable is also categorical. It ranks 
features based on the p-values returned by the test and then selects the top-ranked features.

Note that Scikit-learn’s chi-square function does not carry out the intended procedure. This is a known 
[issue](https://github.com/scikit-learn/scikit-learn/issues/21455). Instead, use `scipy.stats.chi_contingency`.

#### ANOVA

ANOVA is suitable for selecting continuous variables when the target variable is categorical. Let’s explore how we can 
select features using ANOVA and Scikit-learn. We will use the breast cancer dataset:

```
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split

# load dataset
breast_cancer = load_breast_cancer()
X = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
y = breast_cancer.target

# Separate data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Rank and select features
sel = SelectKBest(score_func = f_classif, k=10).fit(X_train, y_train)

# remove features
X_train_t = sel.transform(X_train)
X_test_t = sel.transform(X_test)
```

 With `SelectKBest` we indicate the number of features we want to select. This is an arbitrary value, but can be optimized
with cross-validation.

### Lasso regularization

Lasso can shrink some of the coefficients of a linear model to 0, therefore selecting features out-of-the-box. Here, I’ll 
show how to select features using Lasso using a classification and a regression dataset.

Let's begin by importing the libraries, functions, and classes:

```
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
```

We will next import the breast cancer dataset from Scikit-learn with the aim of predicting 
if a tumor is benign or malignant. This is a classification dataset. Next, we will split 
the data into a training and a testing set:

```
breast_cancer = load_breast_cancer()
X = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
y = breast_cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
```

Let's set up the standard scaler from Scikit-learn:

```
scaler = StandardScaler()
scaler.fit(X_train)
```

Next, we will select features utilizing logistic regression as a classifier, with the Lasso regularization:  

``` 
selector = SelectFromModel(
    LogisticRegression(C=0.5, penalty='l1', solver='liblinear', random_state=10))

selector.fit(scaler.transform(X_train), y_train)
```

By executing `selector.get_support()` we obtain a boolean vector with True for the features that have non-zero coefficients:

```
array([False,  True, False, False, False, False, False,  True,  True,
       False,  True, False, False, False, False,  True, False, False,
       False,  True,  True,  True,  True,  True,  True, False,  True,
        True,  True, False])
```

We can identify the names of the set of features that will be removed like this:

```
removed_feats = X_train.columns[(selector.estimator_.coef_ == 0).ravel().tolist()]
```

If we execute removed_feats we obtain the following array with the features that will be removed:

```
Index(['mean radius', 'mean perimeter', 'mean area', 'mean smoothness',
       'mean compactness', 'mean concavity', 'mean fractal dimension',
       'texture error', 'perimeter error', 'area error', 'smoothness error',
       'concavity error', 'concave points error', 'symmetry error',
       'worst compactness', 'worst fractal dimension'],
      dtype='object')
```

We can remove the features from the training and testing sets like this:

```
X_train_selected = selector.transform(scaler.transform(X_train))
X_test_selected = selector.transform(scaler.transform(X_test))
```

If we now execute X_train_selected.shape, X_test_selected.shape, we obtain the shapes of the 
reduced datasets: ((426, 14), (143, 14)).

Go ahead and change the value of the penalty (C) to see if the result changes. The best 
value of C, and thus, the best feature subset, can be determined with cross-validation.

####  Recursive feature elimination

Recursive feature elimination is a sequential process where a feature is removed after each iteration, and the importance 
of the features is reassessed after each elimination. In Scikit-learn, we can implement recursive feature elimination
with the [RFE](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html) 
or [RFECV](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html#sklearn.feature_selection.RFECV).

Let’s select features recursively using the importance of random forests. We will use the breast cancer data set, and 
separate the data into a train and a test data set. Features should be selected based on the training data only.

```
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split

# load dataset
breast_cancer = load_breast_cancer()
X = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
y = breast_cancer.target

# Separate data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf = RandomForestClassifier(n_estimators=10, random_state=10)

sel_ = RFE(
    clf,
    n_features_to_select=8,
    step=2,
)

sel_.fit(X_train, y_train)

X_train_selected = sel_.transform(X_train)
X_test_selected = sel_.transform(X_test)
```

The result consists of Numpy arrays with the selected features.

### Feature selection with Feature-engine

Feature-engine contains many classes to select features based on recursive feature elimination or addition, feature shuffling,
population stability index, mean target value, cardinality and more. Check out [Feature-engine's documentation](https://feature-engine.readthedocs.io/en/latest/api_doc/selection/index.html#)
for more details.

#### Univariate feature selection

Feature-engine includes univariate feature selection methods based on the target variable mean value per category or 
bin and a single feature classifier or regressor performance metric.

In single feature model performance, a machine learning model is trained for each feature, with only that feature as input, 
and the features are ranked based on this model performance.

Let’s select features based on single feature model performance utilizing cross-validation:

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import RandomForestClassifier

from feature_engine.selection import SelectBySingleFeaturePerformance

# load dataset
breast_cancer = load_breast_cancer()
X = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
y = breast_cancer.target

# Separate data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

sel = SelectBySingleFeaturePerformance(
    estimator=RandomForestClassifier(random_state=10),
    scoring='roc_auc',
    cv=3,
    threshold=None,
)

X_train_t = sel.fit_transform(X_train, y_train)
X_test_t = sel.transform(X_test)
```

We can do some visualization to explore the importance of the features based on the single feature classifiers:

```
pd.Series(sel.feature_performance_).sort_values(
    ascending=False).plot.bar(figsize=(10, 5))

plt.ylabel('roc-auc')
plt.title('Univariate performance')
```

![feature importance based on a single feature classifier]({{ site.baseurl }}/assets/images/posts/fspython/singleperformance.png)
**Image taken from [Feature Selection in Machine Learning with Python](https://leanpub.com/feature-selection-in-machine-learning/)**

#### Correlation

When training linear models like linear or logistic regression, multicollinearity may affect model performance. Thus, it 
might be useful to remove correlated features.

Feature-engine contains algorithms that select features based on the feature correlation. The `SmartCorrelationSelector`
finds groups of correlated features and then retains the one with fewer missing data points, higher cardinality or variability, 
or greater model derived importance.

How can we find correlated features? We can use the `pandas.corr` method.

```
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

from sklearn.datasets import make_classification

# Toy dataset with correlated features

X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_redundant=7,
    n_classes=2,
    random_state=10,
)

X = pd.DataFrame(X)

# the default correlation method of pandas.corr is pearson
corrmat = X_train.corr(method='pearson')

# we can make a heatmap with seaborn
sns.heatmap(corrmat, annot=True)
plt.show()
```

![feature correlation]({{ site.baseurl }}/assets/images/posts/fspython/correlation.png)
**Image taken from [Feature Selection in Machine Learning with Python](https://leanpub.com/feature-selection-in-machine-learning/)**


## More feature selection in Python

If you want to learn more about feature selection and how to carry it out in Python, check out our course and book:

- course: [Feature Selection for Machine Learning](https://www.trainindata.com/p/feature-selection-for-machine-learning)
- book: [Feature Selection in Machine Learning with Python](https://leanpub.com/feature-selection-in-machine-learning/)

Both the course and the book contain a great amount of information regarding:

- the logic of the algorithm
- their advantages and limitations
- code examples using real world datasets

Course and book are suitable for beginner and intermediate data scientists alike.
