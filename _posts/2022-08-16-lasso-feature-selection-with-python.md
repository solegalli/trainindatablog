---
layout: post
title:  "Feature selection with Lasso in Python"
author: sole
categories: [ Feature selection, Python, Machine learning ]
image: assets/images/posts/lasso/lasso.png
---

Lasso is a regularization constraint introduced to the objective function of linear models 
in order to prevent overfitting of the predictive model to the data. The name Lasso stands 
for Least Absolute Shrinkage and Selection Operator.

It turns out that the Lasso regularization has the ability to set some coefficients 
to zero. This means that Lasso can be used for variable selection in machine learning. If 
the coefficients that multiply some features are 0, we can safely remove those 
features from the data. The remaining are the important features in the data.

Lasso was designed to improve the interpretability of machine learning models by reducing 
the number of features. Other regularization methods, like Ridge regression or elastic net, 
do not share this property.

![lasso feature selection]({{ site.baseurl }}/assets/images/posts/lasso/gradient-descent.gif)   

Let’s do a short recap on linear models and regularization.

*For tutorials on feature selection check out our course 
[Feature Selection for Machine Learning](https://www.trainindata.com/p/feature-selection-for-machine-learning) or our 
book [Feature Selection in Machine Learning with Python](https://leanpub.com/feature-selection-in-machine-learning/).*


## Linear models

Linear regression models aim to predict the outcome based on a linear combination of the 
predictor variables given by:

![linear regression model equation]({{ site.baseurl }}/assets/images/posts/lasso/fig1.png)   

The values of the regression coefficients are usually determined by minimizing the squared 
difference between the real and the predicted value of y:

![ordinary least-square function]({{ site.baseurl }}/assets/images/posts/lasso/fig2.png)

This is called the "ordinary least-square" (OLS) loss.

In high-dimensional feature spaces, that is, if the data set has a lot of features, linear 
models are likely to overfit the data. To prevent this, the search for the optimal coefficients 
is done with regularization.

There are two main regularization procedures: the Ridge and the Lasso regularization. With 
the Lasso regression, the coefficients are estimated by minimizing the following equation:

![lasso regularization equation]({{ site.baseurl }}/assets/images/posts/lasso/fig3.png)

where the last term is the regularization constrain, and lambda is the regularization parameter 
that governs the strength of the constraint.

The Ridge regression estimates the regression coefficients by minimizing:

![ridge regularization equation]({{ site.baseurl }}/assets/images/posts/lasso/fig4.png)

where the constraint on the coefficients is given by the sum of the squared values of beta 
instead of their module.

In both regularization procedures, the absolute value of the coefficients of the linear 
model is shrunk to reduce bias, or in other words, to prevent overfitting. However, only 
Lasso can reduce the coefficients value to zero and, as such, help reduce the number of 
features in the data as an integral part of the optimization algorithm.

In the following image, we see the values of the coefficients for 15 features of the 
breast cancer dataset, estimated by a Lasso regression with varying constraints, which in 
the plot are called penalties. As the value of the penalty increases, more and more 
coefficients are set to zero.


![feature selection by lasso]({{ site.baseurl }}/assets/images/posts/lasso/fig5.png)


In contrast, the Ridge regularization does not have that property, or at least not until 
the penalty term is very large, as can be witnessed in the following image:


![change in coefficient value by ridge regression]({{ site.baseurl }}/assets/images/posts/lasso/fig6.png)


Lasso feature selection is known as an embedded feature selection method because the feature 
selection occurs during model fitting.


Let’s see how we can select features with Python and the open source library Scikit-learn.


### Python implementation

We will show how to select features using Lasso using a classification and a regression 
dataset.

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
sel_ = SelectFromModel(
    LogisticRegression(C=0.5, penalty='l1', solver='liblinear', random_state=10))
sel_.fit(scaler.transform(X_train), y_train)
```

By executing sel_.get_support() we obtain a boolean vector with True for the features that have non-zero coefficients:

```
array([False,  True, False, False, False, False, False,  True,  True,
       False,  True, False, False, False, False,  True, False, False,
       False,  True,  True,  True,  True,  True,  True, False,  True,
        True,  True, False])
```

We can identify the names of the set of features that will be removed like this:

```
removed_feats = X_train.columns[(sel_.estimator_.coef_ == 0).ravel().tolist()]
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
X_train_selected = sel_.transform(scaler.transform(X_train))
X_test_selected = sel_.transform(scaler.transform(X_test))
```

If we now execute X_train_selected.shape, X_test_selected.shape, we obtain the shapes of the 
reduced datasets: ((426, 14), (143, 14)).

Go ahead and change the value of the penalty (C) to see if the result changes. The best 
value of C, and thus, the best feature subset, can be determined with cross-validation.

Let's now select features in a regression dataset. Let's import the California housing 
dataset, with the aim of predicting house prices. Next, we separate the data into a training set and a testing set:

```
X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
```

Let's set up a standard scaler to scale the features:

```
scaler = StandardScaler()
scaler.fit(X_train)
```

Next, we select features with a Lasso regularized linear regression model:

```
sel_ = SelectFromModel(Lasso(alpha=0.001, random_state=10))
sel_.fit(scaler.transform(X_train), y_train)
```

By executing sel_.get_support() we obtain a boolean vector with True for the features that will be selected:

```
array([ True,  True,  True,  True,  True,  True,  True,  True])
```

We can obtain the name of the selected features by executing sel_.get_feature_names_out(). 

We can reduce the datasets as follows:

```
X_train_selected = sel_.transform(scaler.transform(X_train))
X_test_selected = sel_.transform(scaler.transform(X_test))
```

That's it, we have now selected features utilizing the ability of the Lasso regularization to shrink coefficients to zero.

If you made it this far, thank you for reading. 

*Don't forget to check out our course [Feature Selection for Machine Learning](https://www.trainindata.com/p/feature-selection-for-machine-learning) and our 
book [Feature Selection in Machine Learning with Python](https://leanpub.com/feature-selection-in-machine-learning/).*


## References

- Tibshirani R, Regression Shrinkage and Selection via the Lasso, J. R. Statistics Society, 58: 267-288, 1996.

- Hastie, Tibshirani, Wainwright, Statistical Learning with Sparsity, The Lasso and Generalizations, CRC Press, Taylor and Francis Group, 2015.

- For a mathematical demonstration of the Lasso property visit [this link](https://bit.ly/3zWBO8L)

- For a visualization of the Lasso property visit [this link](https://bit.ly/3QrWNXY)