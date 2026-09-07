---
layout: post
title: "Feature Selection with Embedded Methods"
author: sole
description: "Learn what embedded methods for feature selection are, their advantages and limitations, and how to implement them in Python."
excerpt: "Learn what embedded methods for feature selection are, their advantages and limitations, and how to implement them in Python."
categories: [Feature Selection, Machine Learning]
image: assets/images/posts/feature-selection-with-embedded-methods/embedded-methods-feature-selection-1.png
---

Imagine you’re working with a large dataset, and you want to train a machine learning algorithm. The challenge lies in deciding which features from the myriad of variables should be considered to build an effective model. This is where feature selection comes into play, allowing us to sift through the data clutter and create more interpretable and robust models.

[Feature selection](https://www.blog.trainindata.com/feature-selection-for-machine-learning/) consists of selecting a set of features from a data set to train machine learning algorithms. The aim of the feature selection process is to reduce the number of features, which leads to increased interpretability and more resilient models.

Feature selection methods can be divided into three groups: filter methods, wrapper methods, and embedded methods.

## Feature Selection Methods

[Filter methods](https://www.blog.trainindata.com/feature-selection-with-filter-methods/) are model agnostic feature selection techniques that select features based on the characteristics of the data, independently of the machine learning model. They have the lowest computation cost, and include the chi-square test and Pearson’s correlation coefficient, among others.

[Wrapper methods](https://www.blog.trainindata.com/feature-selection-with-wrapper-methods/) are feature selection algorithms that wrap the search around a predictive model. They generate multiple feature subsets, train a classification or regression model on each subset (iteration), and determine its performance.

Wrapper methods have the highest computational cost. Examples of wrapper methods include forward selection, backward elimination and exhaustive search.

Embedded methods “embed” the selection procedure in the training of the predictive model. The search for an optimal subset of features is built into the training of the classifier or the regression algorithm. Hence, embedded methods train only one machine learning model to select features. Their computational cost equals the model training time.

In this blog post, we will focus on embedded methods.

To learn more about these and other feature selection methods, check out our [Feature Selection for Machine Learning course](https://www.trainindata.com/p/feature-selection-for-machine-learning) and [Feature Selection in Machine Learning book](https://www.trainindata.com/p/feature-selection-in-machine-learning-book).

[![Feature Selection for Machine Learning, online course.]({{ site.baseurl }}/assets/images/posts/feature-selection-with-filter-methods/feature-selection-course-1024x576.png)](https://www.trainindata.com/p/feature-selection-for-machine-learning)

## Embedded methods

**Embedded methods** “embed” the selection in the model building phase. A typical embedded feature selection workflow involves:

- Training a machine learning model.
- Deriving feature importance.
- Selecting the top ranking predictor variables.

The most common embedded strategies are the Lasso regularization for in linear models and the feature importance obtained from the information gain in decision trees.

Note however that not all machine learning models can naturally embed a feature selection process. One method that can’t for example is support vector machines (svm).

## Lasso

Linear regression models predict the outcome based on a linear combination of the feature space. The coefficients are determined by minimizing the squared difference between the real and the predicted value of *the target.*

There are three main regularization procedures: the **Ridge** and the **Lasso** regularization, and elastic net which combines the former 2.

In Lasso regression, the coefficients are shrink by a given constant. In Ridge regression, the square of the coefficients are penalized by a constant. The aim of shrinking the coefficients is to reduce bias and prevent overfitting. The best constant needs to be estimated through hyperparameter optimization.

It turns out that the Lasso regularization has the ability to set some of the coefficients to zero. Then, we can safely remove those features from the data.

In the following image, we see the change in the coefficients with increasing regularization penalties. As the penalty increases, more and more coefficients are set to zero.

![Diagram showing the change in the coefficients value with increasing Lasso regularization penalties]({{ site.baseurl }}/assets/images/posts/feature-selection-with-embedded-methods/fig5.png)

In contrast, the Ridge regularization does not have that property, or at least not until the penalty is very big, as can be witnessed in the following image:

![Diagram showing the change in the coefficients value with increasing Ridge regularization penalties]({{ site.baseurl }}/assets/images/posts/feature-selection-with-embedded-methods/fig6.png)

Stronger regularization lead to a bigger dimensionality reduction. We can optimize the penalization utilizing cross-validation, either to improve model performance, as I teach in the [Feature Selection course](https://www.trainindata.com/p/feature-selection-for-machine-learning), or to improve interpretability, as I discuss in the [Interpreting Machine Learning Models course](https://www.trainindata.com/p/machine-learning-interpretability).

### Lasso implementation in Python

Let’s see how to select relevant features with Lasso in Python. Let’s import the libraries, functions, and classes:

```
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
```

Let’s import the breast cancer dataset from Scikit-learn and split it into a training and a testing set:

```
breast_cancer = load_breast_cancer()
X = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
y = breast_cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
```

Let’s set up a scaler to standardize the features:

```
scaler = StandardScaler()
scaler.fit(X_train)
```

Next, we will select features utilizing a logistic regression with the Lasso regularization:

```
sel_ = SelectFromModel(
    LogisticRegression(
        C=0.5, penalty='l1', solver='liblinear', random_state=10),
    )
sel_.fit(scaler.transform(X_train), y_train)
```

By executing `sel_.get_support()` we obtain a boolean vector with True for the features that have non-zero coefficients:

```
array([False, True, False, False, False, False, False, True, True,
False, True, False, False, False, False, True, False, False,
False, True, True, True, True, True, True, False, True,
True, True, False])
```

We can identify the names of the removed features like this:

```
removed_feats = X_train.columns[(sel_.estimator_.coef_ == 0).ravel().tolist()]
```

If we execute `removed_feats` we obtain the following array with the features that will be removed:

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
X_train_selected = sel_.transform(scaler.transform(X_train))
X_test_selected = sel_.transform(scaler.transform(X_test))
```

If we now execute:

`X_train_selected.shape, X_test_selected.shape`

we obtain the shapes of the reduced datasets:

`((426, 14), (143, 14))`

## Feature importance from decision trees

Decision tree algorithms predict an outcome by making sequential partitions of the data. At each node, a feature and a value are selected to carry out a partition. The best partition is that that maximizes the decrease in impurity.

There are different metrics that can be used to determine “impurity”. In classification, the algorithm minimizes the Gini or the entropy. In regression, the algorithm minimizes the mean squared error (like in least squares), the mean absolute error, or the Poisson deviance.

The importance of each feature is given by the total reduction in impurity throughout the tree. For example, if a feature is used to partition data in node 1 and then again in node 3, the importance of that feature is determined by the sum of the impurity reduction at both nodes.

Random forests grow many classification trees in parallel and the prediction is the average of the various trees. Hence, the feature importance is the given by the average importance across trees.

Gradient boosting machines, like xgboost, instead, build sequential trees that minimize the difference between the predictions of the tree and the residuals of the previous tree. Here, the feature importance is the sum of importances across trees.

We can then select the features of the highest importance.

### Python implementation

Let’s select the most important features from decision tree based models. We will use random forests and the breast cancer dataset.

Let’s begin by importing the libraries, functions, and classes:

```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
```

Let’s load the dataset and separate it into a training and a testing set:

```
breast_cancer = load_breast_cancer()
X = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
y = breast_cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
```

Let’s select features based on their importance derived from a random forest classifier:

```
sel_ = SelectFromModel(RandomForestClassifier(n_estimators=10, random_state=10))
sel_.fit(X_train, y_train)
```

`SelectFromModel` will select features whose importance is greater than the average importance of all features. This can be modified through the parameter `threshold`.

With `sel_.get_support()` we obtain a boolean vector with `True` for the features that were selected:

```
array([ True, False, False, True, False, False, True, True, False,
False, True, False, False, False, False, False, False, False,
False, False, False, False, True, True, False, False, False,
True, False, False])
```

We can make a list with the selected features as follows:

```
selected_feat = X_train.columns[(sel_.get_support())]
```

By executing `len(selected_feat)` we obtain the number of selected features: `8`. By executing `selected_feat` we obtain the names of the selected variables:

```
Index(['mean radius', 'mean area', 'mean concavity', 'mean concave points',
'radius error', 'worst perimeter', 'worst area',
'worst concave points'],
dtype='object')
```

Let’s plot the feature importance:

```
pd.Series(
    sel_.estimator_.feature_importances_.ravel(),
    index=X_train.columns).plot.bar(figsize=(10,5),
)
plt.ylabel('Feature importance')
plt.show()
```

Below we see the importance assigned to each feature by random forests:

![Bar plot showing the feature importance derived from random forests]({{ site.baseurl }}/assets/images/posts/feature-selection-with-embedded-methods/ch7-fig3.png)

Finally, we can reduce the datasets to the selected variables:

```
X_train_selected = sel_.transform(X_train)
X_test_selected = sel_.transform(X_test)
```

And that’s it, we have now selected features based on the importance obtained from decision trees.

## Embedded methods and Recursive Feature elimination

The importance derived from linear regression or decision trees is influenced by correlations. Co-linearity, decreases the value of importance in general. Hence, to avoid removing correlated features, that could be more important than other features that are not correlated with anything else, we tend to combine embedded methods with recursive feature elimination.

In recursive feature elimination, we re-train the model after removing one or more irrelevant features, and hence, if a correlated feature is removed, the remaining correlated ones would show increased importance.

For more information about embedded methods in the context of feature selection or interpretability, check out the following resources:

- [Feature selection for machine learning](https://www.trainindata.com/p/feature-selection-for-machine-learning), online course
- [Feature selection in machine learning](https://www.trainindata.com/p/feature-selection-in-machine-learning-book), book
- [Interpretable machine learning](https://www.trainindata.com/p/machine-learning-interpretability), course

## References

### Lasso:

- Tibshirani R, Regression Shrinkage and Selection via the Lasso, J. R. Statistics Society, 58: 267-288, 1996.
- Hastie, Tibshirani, Wainwright, Statistical Learning with Sparsity, The Lasso and Generalizations, CRC Press, Taylor and Francis Group, 2015.

### Decision tree importance:

- Breiman, et. al., Classification and Regression Trees, Wadsworth, Belmont, CA, 1984.
- Breiman, Random Forests, Machine Learning 45:5–32, 2001.

### Recursive feature elimination:

- Guyon, et. al., Gene selection for cancer classification using support vector machines, Mach. Learn., 46, 389–422, 2002.
