---
layout: post
title: "Partial Dependence Plots with Python: A Comprehensive Guide"
author: sole
description: "Discover partial dependence plots, how they help you understand your machine learning model's predictions, and implement them in Python."
excerpt: "Discover partial dependence plots, how they help you understand your machine learning model's predictions, and implement them in Python."
categories: [Interpretable Machine Learning]
image: assets/images/posts/partial-dependence-plots-with-python/partial-dependence-plots.png
---

Partial dependence plots are a model agnostic visualization tool that allow us to understand the relationship between a target variable and a specific feature in any machine learning model. In this article I will describe what partial dependence plots are, their advantages and limitations, and how to implement them in Python.

> Master this and other interpretable machine learning tools with our course [Interpreting Machine Learning Models](https://www.trainindata.com/p/machine-learning-interpretability).

[![Machine learning interpretability online course]({{ site.baseurl }}/assets/images/posts/machine-learning-interpretability/interpretable-machine-learning-1024x576.png)](https://www.trainindata.com/p/machine-learning-interpretability)

## **Interpretable Machine Learning**

Interpretability in machine learning is vital for understanding how models make decisions. It allows us to gain insights into the underlying patterns, evaluate the model’s behavior, address ethical concerns such as bias and discrimination, build trust in the model’s results, and facilitate collaboration between different stakeholders.

Some models, like linear regression, decision trees random forests and gradient boosting machines, are intrinsically explainable. By analyzing the components of these model, we can understand which features influence their predictions and obtain a measure of “feature importance.”

Other models like neural networks, are unexplainable by design, the so called black-box models. For black box models we can compute feature importance by using post hoc methods like [permutation feature importance](https://www.blog.trainindata.com/permutation-feature-importance/).

In general, these values of feature importance tell us how important the feature is for the model’s prediction, but we can’t say if the feature influences the output in a positive or negative manner. This is when partial dependence plots come into play.

## **What are Partial Dependence Plots (PDPs)?**

Partial dependence plots are an essential tool for interpreting machine learning models and gaining a deeper understanding of their behavior. These plots allow us to visualize the relationship between **a specific feature** and the model’s predictions while holding all other features constant.

In a partial dependence plot, we can see how the value of the model’s output changes at different values of the predictor. Hence, we can understand how important the feature is, and also, whether the relationship is linear, monotonic, or something else, and the direction of influence of the feature on the prediction.

PDP plots complement the information that we obtain from global interpretation methods, like permutation feature importance or the importance derived from decision trees. From these methods we obtain a value of feature importance and from PDP plots we understand the specific relationship between the predictor and the target variable.

Master partial dependence plots and additional interpretability methods with our comprehensive course [Interpreting Machine Learning Models](https://www.trainindata.com/p/machine-learning-interpretability).

## **Step-by-Step Guide to Creating Partial Dependence Plots**

Partial dependence plots allow us to visualize the relationship between a specific predictor variable and the target variable while holding all other variables constant.

Creating a PDP involves the following steps:

1. Fit the machine learning model using training data.
2. Select the feature of interest for which you want to create the partial dependence plot.
3. Vary the value of the selected feature, while keeping other features constant.
4. For each value of the selected feature, predict the target variable using the trained model.
5. Determine the mean predicted value per value of the feature.
6. Plot the feature values on the x-axis and the corresponding mean predicted values on the y-axis.
7. Connect the predicted target values with a line to visualize the relationship.

Simple!

### **Creating Synthetic Data Points**

In PDP, we take a dataset and we set the values of one feature to a series of specific numbers or categories. In other words, we generate synthetic data points by arbitrarily assigning one value to all observations in the dataset, while keeping all other features constant. See the image below for an illustration:

**![Animation showing the procedure to create partial dependence plots.]({{ site.baseurl }}/assets/images/posts/partial-dependence-plots-with-python/PDP-gif.gif)**

There are various ways in which we can determine which values we should use.

Scikit-learn takes equidistant values between the minimum and maximum value of the variable if numerical. PDP Box, instead, sorts the variable into intervals of equal-width or equal frequency, and takes the middle point of the interval as the data point to which we will set the feature.

For categorical variables, we will set the feature to each one of the categories.

### **Obtaining the predictions**

Once we generated these synthetic data points, we use the trained machine learning model to predict outcomes for each observation in the dataset. Then, we take the average of these predicted outcomes for each feature value.

We then plot the averaged predictions against the different values of the chosen predictor variable. And that constitutes the PDP. This visualization gives you an understanding of how changes in that specific feature influence predictions made by your machine learning model.

> Master machine learning interpretability with the most comprehensive course available online: [Interpreting Machine Learning Models](https://www.trainindata.com/p/machine-learning-interpretability).

## **Interpreting Partial Dependence Plots**

Partial dependence plots provide insights into how a feature affects the predictions made by the model. Here are a few key insights that can be obtained from PDPs:

1. **Directionality**: The slope and direction of the line connecting the feature values indicate the direction of influence. A positive slope suggests a positive relationship, while a negative slope indicates a negative relationship.
2. **Strength of Influence**: The steeper the slope of the line, the stronger the influence of the feature on the target variable.
3. **Non-Linear Relationships**: PDPs can reveal non-linear relationships between the feature and the target variable. For example, if the line is not straight and exhibits curves or bends, it suggests a non-linear relationship.
4. **Feature Importance**: By comparing the PDPs of different features, it is possible to identify the most important features in the model.
5. **Interaction Effects**: PDPs can also be used to identify if there are interactions between features. If the slopes of the lines change when another feature is varied, it indicates an interaction effect between the two features.

## **Implementing Partial Dependence Plots with Python**

Python provides a range of libraries that make it convenient to generate PDPs for different machine learning models. In this article, I show how to create PDP plots with Scikit-learn and PDP Box. We will use 2 real-world examples.

For additional code demos, including Dalex and a manual implementation that gives you the maximum versatility, check out our course “[Interpreting Machine Learning Models](https://www.trainindata.com/p/machine-learning-interpretability).”

### **PDP plots with Scikit-learn**

Scikit-learn supports the implementation of partial dependence plots for categorical and numerical variables. It is straightforward to implement, particularly if interpreting sklearn models. However, it does not show the number of observations per category or value and therefore, we lack a confidence metric around how much we can trust those plots for certain values of the variable.

Let’s start by importing the modules and functions:

```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from sklearn.inspection import PartialDependenceDisplay
```

Now we load and prepare some data:

```
X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X = X.drop(columns=["Latitude", "Longitude"])

X["AveRooms"] = np.where(X["AveRooms"]<7, 0, 1)
X["AveBedrms"] = np.where(X["AveBedrms"]<7, 0, 1)
```

```

```

We separate the dataset into a training set and a testing set:

```
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)
```

Let’s train a random forest regression model to predict house price:

```
rf = RandomForestRegressor(
    criterion="squared_error",
    n_estimators=25,
    max_depth=3,
    random_state=3,
)

rf.fit(X_train, y_train)
```

And finally, we create PDP plots for five of the variables:

```
fig, ax = plt.subplots(figsize=(15, 10))
ax.set_title("Partial Dependence Plots")

PartialDependenceDisplay.from_estimator(
    estimator=rf,
    X=X_test,
    features=(0, 5, 2, 3), # the features to plot
    categorical_features = (2, 3), # categorical features
    random_state=5,
    ax=ax,
)

plt.show()
```

```

```

In the following plots, we can see that the income is the most important variable, because the house price changes a lot with the different median income values. And we also see that the relationship is monotonic but not linear, which makes sense because we are using random forests.

We also see that the average number of rooms or bedrooms, which we modelled as categorical, do not influence the house price.

![Partial dependence plots created with Scikit-learn]({{ site.baseurl }}/assets/images/posts/machine-learning-interpretability/partial-dependence-sklearn-1024x717.png)

### PDP plots with PDP BOX

Let’s now use a different library specifically designed to create PDP plots: PDP Box.

Let’s start by making some imports:

```
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from feature_engine.encoding import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from pdpbox import pdp
```

Next, let’s load and prepare the data:

```
data = pd.read_csv('https://www.openml.org/data/get_csv/16826755/phpMYEkMl')
data = data.replace('?', np.nan)
def get_first_cabin(row):
    try:
        return row.split()[0]
    except:
        return np.nan
data['cabin'] = data['cabin'].apply(get_first_cabin)

data["cabin"] = data["cabin"].str[0]
data.loc[data["cabin"] == "T", "cabin"] = "G"

data.fillna({
    "cabin":"M",
    "embarked": "S",
}, inplace=True)

data[['pclass', 'sibsp', 'parch']] = data[['pclass', 'sibsp', 'parch']].astype(int)
data['fare'] = data['fare'].astype(float)
data['fare'] = data['fare'].fillna(data['fare'].median())
usecols = ["pclass", "sibsp", "parch", "sex", "fare", "embarked", "cabin", "survived", ]
data = data[usecols]
data = OneHotEncoder(drop_last=False).fit_transform(data)

X = data.drop(columns='survived')
y = data['survived']
```

```
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

```

Now, we train an xgboost classifier for binary classification:

```
xgbc = XGBClassifier(n_estimators=10,
                     max_depth=1,
                     random_state=3,)

xgbc.fit(X_train, y_train)
```

To use PDPbox to generate the plots, we start by creating a PDP analysis object using the PDPIsolate class

```
pdp_sex_female = pdp.PDPIsolate(
    model=xgbc,
    df=pd.concat([X_test, y_test], axis=1),
    model_features=X_test.columns,
    feature="sex_female",
    feature_name="gender",
)
```

Let’s make a PDP plot for gender:

```
fig, axes = pdp_sex_female.plot(
    center=False,
    plot_lines=False,
    plot_pts_dist=False,
    to_bins=False,
    engine='matplotlib',
)

fig.set_figheight(5)
```

In the following image we see the partial dependence between the survival probability and gender:

![PDP plot between survival and gender]({{ site.baseurl }}/assets/images/posts/partial-dependence-plots-with-python/pdp-gender-1024x345.png)

We can create a PDP plot for a continuous variable:

```
fig, axes = pdp_fare.plot(
    center=True,
    plot_lines=False,
    show_percentile=True,
    plot_pts_dist=True,
    to_bins=True,
    engine='matplotlib',
)
fig.set_figheight(6)

![PDP plot between survival and fare.]({{ site.baseurl }}/assets/images/posts/partial-dependence-plots-with-python/pdp-fare-survival-1024x393.png)
```

And with PDP box we can also create 1 PDP box for all the [one hot encoded](https://www.blog.trainindata.com/one-hot-encoding-categorical-variables/) variables that come from the variable cabin:

```
cabin_features = sorted([c for c in X_test if c.startswith('cabin_')])

pdp_cabin = pdp.PDPIsolate(
    model=xgbc,
    df=pd.concat([X_test, y_test], axis=1),
    model_features=X_test.columns,
    feature=cabin_features, # we set all the OHE features that correspond to cabin here
    feature_name="cabin", # we can set the name of the main variable here (before pre-processing)
)

fig, axes = pdp_cabin.plot(
    center=True,
    plot_lines=False,
    plot_pts_dist=True,
    to_bins=False,
    engine='matplotlib',
);

fig.set_figheight(5)
```

![PDP plot for a variable that was encoded using one hot encoding.]({{ site.baseurl }}/assets/images/posts/partial-dependence-plots-with-python/pdp-one-hot-encoding-1024x336.png)

## Advantages of Partial Dependence Plots

PDP plots are commonly used in data science. They provide useful visualizations of the relationship between the variable of interest and the model’s predictions.

If the feature of interest is not correlated with other features, then the PDP accurately represent how the feature influences the prediction on average.

## Limitations of Partial Dependence Plots

While PDPs can provide valuable insights into the relationships between predictors and a target variable, they are not without their challenges.

One of the primary limitations of PDPs is the potential for biases in interpretation. These plots illustrate the average effect of a single predictor on the target variable while holding other predictors constant. However, this assumption may not always hold true, especially in complex models with interactions and non-linear relationships.

Another challenge arises when dealing with high-dimensional data. PDPs become less interpretable as the number of predictors increases, making it difficult to visualize and understand the effects of each individual predictor on the target variable.

To mitigate these limitations, it is important to complement PDP analysis with other techniques such as feature importance rankings or model diagnostics. Additionally, considering domain knowledge and conducting robust sensitivity analyses can help address potential biases and improve the reliability of interpretations derived from PDPs.

## **Alternative Interpretability Algorithms**

To overcome the limitations of partial dependence plots, we could use instead accumulated local effect plots. To expand the interpretation of partial dependence plots instead of plotting the mean predicted target value, we can plot each observations. This is what is known as individual conditional expectation plots (ICE plots).

Other model agnostic methods include SHAP and Shapley values.

## **Conclusion**

Partial dependence plots are a valuable tool in interpreting and visualizing how individual features influence the predicted outcome in machine learning models. By understanding the relationships between features and the target variable, we can gain insights into the model’s behavior, identify important features, and ensure model transparency and fairness.

Whether you are a data scientist, a machine learning practitioner, or an enthusiast, incorporating partial dependence plots into your model interpretability toolkit can enhance your understanding and gain insights into the complex relationships within your machine learning models.

## **Additional Resources**

[Interpreting Machine Learning Models](https://www.trainindata.com/p/machine-learning-interpretability).

[Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.PartialDependenceDisplay.html)

[PDP Box](https://pdpbox.readthedocs.io)
