---
layout: post
title: "Interpretability in Machine Learning. An Overview"
author: shri
description: "Discover what machine learning interpretability is and why it matters. Learn various interpretable machine learning methods and apply them in Python."
excerpt: "Discover what machine learning interpretability is and why it matters. Learn various interpretable machine learning methods and apply them in Python."
categories: [Interpretable Machine Learning, Machine Learning]
image: assets/images/posts/machine-learning-interpretability/ml-interpretability-blog.png
---

In today’s digital world, most businesses leverage machine learning models trained on datasets to make critical decisions. Hence, it is crucial to understand how the machine learning model works and how it produces its output. This is what model interpretability is about.

Machine learning interpretability refers to the capacity to express what a model has learned and the factors influencing its outputs in clear and understandable terms. Clear and understandable for who, you may ask? For us, people.

That’s as simple as it gets. Either we understand what the model is doing, or we don’t.

## When Do We Need Interpretability?

Understanding how artificial Intelligence models work is crucial, because they allow us to take responsibility for the decisions our organization makes, and provide answers to those whose our models directly affect.

Take, for instance, banks using credit risk algorithms to decide whether to approve or deny a loan to a customer. When a loan is denied, banks need to be able to explain why the model, and hence themselves, made that decision.

The need for interpretable machine learning becomes particularly obvious in areas like financial modelling, insurance, healthcare, policing, recruitment, safety, and other fields where the outputs of our models, black box or not, directly affect people.

However, I’d argue that interpretability is important **always**. Why? Because it allows us to understand what the model is doing, how the model is doing it, and protect us, our organization and those who the model affects from various risks, including financial loss, reputation damage, and missed opportunities.

In this article, I’ll first discuss why interpretability is important in real-world applications. Then, I’ll explore different methods and their key characteristics. And finally, I’ll show you how to apply some of those methods in Python.

Let’s dive in.

## Why Does Interpretability Matter?

Developing interpretable machine learning systems can help businesses in multiple spheres:

- **Enhance Domain Knowledge:** Machine learning models not only automate decision-making processes but also generate valuable insight. Data scientists can access this insight by, for example, understanding the relationships in their data through interpretations. For instance, in finance, models can discern fraudulent claims or predict house prices based on key drivers.
- **Fairness and Compliance:** [Interpreting machine learning models](https://www.trainindata.com/p/machine-learning-interpretability) gives us information on the key factors driving the model’s prediction, and also tells us, if it is a fair and unbiased decision. For example, we can verify that loan rejections are based on financial criteria rather than discriminatory factors like ethnicity, gender, or neighborhood. Interpretable machine learning, in other words, is the key to ensuring ethical standards and regulatory requirements.
- **Improve Model Performance:** By identifying important variables and removing noisy features from the training data, or ensuring that the model does not rely too heavily on a few variables, we can create simpler models that are yet robust and accurate.
- **Building Transparency and Trust in AI:** It is essential to ensure accountability, fairness, and ethical use of the technology, fostering user and public confidence. This, in simpler terms means, that for the use of machine learning to be successful in an organization, the employees and the people affected by those models, need to trust the decisions that the model makes.

## Interpretable Machine Learning Methods

In artificial intelligence, we have a wide range of algorithms, from linear regression to neural networks, that vary in complexity and hence, interpretability. There is a trade-off between complexity and interpretability, and I will say more about in coming paragraphs.

We can group machine learning algorithms into models that are intrinsically explainable, like linear regression, logistic regression, decision trees, random forests and, to some degree, gradient boosting machines, and those models that are unexplainable by design, the so called black box models. The classic example are deep-learning models.

### Intrinsically Explainable Models:

Some machine learning models are interpretable by design, offering insights into their predictions, just out-of-the-box. Examples include linear regression, decision trees, and rule-based models.

These models offer transparency by providing explicit rules or features that influence their predictions. They are suitable for scenarios where transparency is critical, such as regulatory compliance, medical diagnosis, or financial forecasting.

To interpret these models, we usually use model-specific methods. These model specific methods involve directly analyzing the components and parameters of the model. For example, in linear regression models, examining the coefficients provides valuable insights into how the model makes predictions, and how each feature affects the predictions. The higher is the coefficient value, the higher its feature importance.

We can interpret decision trees by directly looking at the tree architecture. In a tree, each node splits data based on a specific value of a feature. Data travels through subsequent nodes until it reaches the final leave, where it will be assigned an output value. By examining the tree architecture we can exactly determine the fate of each observation, which in other words means, we fully understand how the tree makes decisions.

### Black Box Models: Post hoc Methods

Intrinsically explainable models are suitable for a great number of situations, particularly when using tabular data. But sometimes, we need to use black box models, like deep neural networks, complex ensemble methods, or unsupervised models. These models either have superior performance when it comes to capturing complex data patterns, or are tailored to certain applications, like grouping observations together (i.e., by clustering).

While black box deep learning models excel in computer science tasks related to computer vision, or natural language processing, they are not easily interpretable. In other words, there is a trade-off between performance and interpretability in machine learning. So how can we gain trust in these models?

To interpret these models, we can use the so called model-agnostic methods, also known as post-hoc methods. These methods are applied after training the model.

Post-hoc methods aim to explain the black box model outputs without relying on understanding the internal model structure. They work by performing input data perturbations and observing how the model behaves, or they rely on surrogate models to interpret the predictions of the black box.

To explain the model, some post-hoc methods alter the input data to observe changes in model predictions**.** The classic example is permutation feature importance.

Alternatively, training simpler models, that is, an intrinsically explainable model, to predict the black box predictions, allows us to understand, at least to some degree, the decision-making process of he black box. These are the so called global surrogates.

By leveraging post hoc methods, data science practitioners can unlock insights from complex models, enabling transparency and trust in decision-making processes across various domains.

![]({{ site.baseurl }}/assets/images/posts/machine-learning-interpretability/post-hoc-methods-1024x576.png)

## Global vs local explainability

Global interpretability is the assessment of a feature’s contribution to the output of a machine learning model, across the entire dataset. For example, consider a decision tree-based model like XGBoost. We can get the feature importance for all the input features by looking at the decrease in impurity each feature makes at each node in each tree within the ensemble.

Global interpretability helps data scientists understand if the top features are domain-relevant, and also extract new insight from the data. Statistical tests, such as regression and correlation analyses, fall within the realm of global interpretability methods.

While global interpretability provides a bird’s-eye view of model behavior, local interpretability zooms in on individual predictions. By analyzing the components, and tracing data through decision trees or rules, we can uncover the rationale behind individual predictions. This offers valuable insights in fields like credit risk assessment or fraud detection, where the top features may vary segment-wise.

## Global interpretability methods for intrinsically explainable models

Let’s have a look at the widely used intrinsically explainable models and the common interpretability methods used:

1. **Linear Regression**: This model assumes a linear relationship between input features and the target variable. Interpretability methods such as feature importance analysis, coefficient significance testing, and partial dependence plots help in understanding how each feature contributes to the predicted outcome.
2. **Logistic Regression**: Used for classification tasks, logistic regression estimates the probability that a given input belongs to a particular class. Interpretability techniques include odds ratios, coefficient analysis, and visualizations like ROC curves to understand model performance and feature influence.
3. **Generalized Linear Models (GLMs)**: GLMs extend linear regression to accommodate different types of response variables and error distributions. Interpretability methods for GLMs are similar to those for linear regression, with additional considerations for the chosen distribution and link function.
4. **Decision Trees**: Decision trees partition the feature space into regions based on feature thresholds, making them intuitively interpretable. Techniques such as tree pruning, feature importance ranking, and visualization of decision paths aid in understanding which features are most influential in the model. Many libraries provide in built feature importance for decision tree-based models.

## Model-Agnostic Interpretability methods

With the increase of black box models for predictions, there has also been an influx of different interpretability methods proving global and local explanations. Let’s take a look at the methods under both categories:

### Global Model Agnostic Methods:

1. [**Partial Dependence Plots**](https://www.blog.trainindata.com/partial-dependence-plots-with-python/) **(PDP)**: This method allows us to see how the predicted outcome changes with variations in a single feature’s value, while keeping other features fixed. It’s a visualization tool, which we would normally use to understand the relationship between the most important features with the target.
2. **Accumulated Local Effects (ALE) Plots**: In this method, we estimate the average marginal effect of a feature within a small interval of its values. This helps in detecting non-linear relationships between the features and the target, while overcoming some of the limitations of PDP, like for example the assumption that features are not correlated. Again, as it is a visualization tool, we’d only use it to explain a subset of features.
3. **Feature Interaction Analysis**: In this method, we analyze how different combinations of features interact with each other, and the combined influence of them on the target. This method helps us find complex relationships like higher-order interactions that may not be evident when considering features individually.
4. **Functional Decomposition**: As the name suggests, this method decomposes the model’s predictions into additive components associated with each feature. We decompose the prediction function into interpretable components.
5. [**Permutation Feature Importance**](https://www.blog.trainindata.com/permutation-feature-importance/): In this method, the values of features are randomly shuffled to determine the change in model’s performance. The larger the performance deterioration after shuffling a feature, the higher its significance. The beauty of this method is that we can evaluate any performance metric that we want.

### Local Model Agnostic Methods:

1. **Individual Conditional Expectation (ICE) Plots**: When we create ICE plots, we plot the variation of the model’s prediction when a specific feature’s value is varied for an individual instance. Unlike [PDP](https://www.blog.trainindata.com/partial-dependence-plots-with-python/), ICE plots offer insights into the behavior of the model for a particular data point, allowing for a more granular understanding.
2. **Local Interpretable Model-agnostic Explanations (LIME)**: LIME uses an intrinsically explainable model to explain how the black box model makes a prediction for a certain observation. It trains a prediction model, such as a linear regression or a decision tree, that locally mimics the behavior of the complex model, like for example an arrangement of neurons in a neuronal network. This provides insights into why a particular prediction was made for that particular data point.
3. **Counterfactual Explanations**: Counterfactual explanations aim to answer the question: “What changes to the input features would result in a different prediction?” In this method, we generate hypothetical feature values similar to the original ones, but which result in a different prediction. This sheds light on the decision-making process for a particular scenario.

## SHAP and Shapley Values

SHAP stands for SHapley Additive exPlanations. Shapley values are a concept from cooperative game theory and are used in machine learning to fairly allocate the contribution of each feature to the prediction of a model, providing insights into the importance of individual features in the model’s output.

SHAP values are additive, meaning that the sum of the SHAP values for all features plus the model’s average prediction equals the actual prediction for a given instance.

At a local level, Shapley values tell us how a particular prediction was made for a specific instance by a classifier or a regression model. At a global level, aggregating SHAP values across multiple data points helps us identify features that consistently contribute to the model’s predictions.

SHAP can handle missing values, ensuring that the explanations remain valid even when certain features are missing for a few data points. Due to its robustness, SHAP has been used to interpret ML models across various fields and use cases within in healthcare, finance, and safety engineering, among others.

## Python libraries for interpretability

There are many open-source Python libraries that provide frameworks for interpretable ML. Some of the noteworthy packages include :

- **SHAP**: A popular library that provides a unified framework for computing SHAP values.
- **LIME**: This library is used to generate local approximations of complex models.
- **ELI5** (Explain Like I’m 5): Originally designed to explain intrinsically explainable models locally, now it supports a variety of methods, including [permutation feature importance](https://www.blog.trainindata.com/permutation-feature-importance/) and LIME.
- **InterpretML**: It’s a Microsoft package that offers a suite of interpretability methods like global feature importance, individual instance explanations, and [partial dependence plots](https://www.blog.trainindata.com/partial-dependence-plots-with-python/). It is particularly famous for its interpretable constrained GBM (gradient boosting machine).
- **Alibi and Dalex**: These are 2 interpretable machine learning libraries that host various post-hoc interpretability methods.

## Interpretable ML with Python

Let’s look at a few examples of how we can interpret models using some of these Python libraries.

I’ll begin by making global and local explanations for an intrinsically explainable model trained to predict house price. I’ll use the California housing dataset available in sklearn.

The first step is to import the necessary libraries and load the dataset.

```
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X = X.drop(columns=["Latitude", "Longitude"])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
X.head()
```

In the following image, we see the different features of the dataset, like age of the house and number of rooms, among others:

**![First 5 rows of the California housing dataset.]({{ site.baseurl }}/assets/images/posts/machine-learning-interpretability/california-dataset.png)**

Next, lets train a simple random forest regressor on this dataset.

```
rf = RandomForestRegressor(
    criterion="squared_error",
    n_estimators=3,
    max_depth=3,
    random_state=3,
)

rf.fit(X_train, y_train)
```

Once the training is completed, we want to interpret the model: we want to find out how each feature affects the house price.

**A word of advice**: interpreting models makes sense only if the model makes accurate predictions. So before jumping onto explaining what the model does, we need to make sure we carried out the corresponding model validation.

### Global explanations

Let’s start with the global explanations. We can obtain these straightaway from the `feature_importance` attribute from the Scikit-learn’s estimator.

We can check the feature importances calculated by each tree as shown below:

```

global_exp = pd.Series(rf.feature_importances_, index=rf.feature_names_in_)
global_exp.sort_values(ascending=False).plot.bar()
plt.ylabel("Feature importance")
plt.title("Feature importance - Global explanations")
plt.show()
```

In the following plot, we see that `MedInc` is the most important feature:

### Local explanations

Next, let’s see how to obtain local explanations. We will use the package `treeinterpreter`.

First, we obtain the individual explanations for each one of the observations in the test set, and then, we plot the explanations for the **first observation** in the test set:

```
from treeinterpreter import treeinterpreter as ti
```

```
prediction, bias, contributions = ti.predict(rf, X_test)
```

```
pd.Series(contributions[0], index=X_train.columns).plot.bar()
plt.axhline(y=0, color='r', linestyle='-')
plt.ylabel("Feature importance")
plt.title("Local explanations")
plt.show()
```

In the following plot, we see that `MedInc` increases the price a bit for this house, but it’s average occupancy takes the price down a lot.

**![Local explanations extracted from random forests for 1 observation. Local explanations help with interpretability in machine learning at a sample level.]({{ site.baseurl }}/assets/images/posts/machine-learning-interpretability/rf-local-explanations.png)**

### Partial dependence plots

Next, let’s look at how to create partial dependence plots. We need to import a function from Scikit-learn’s inspection’s module. Then, use the `from_estimator()` method, providing the data rows, the model and the features to plot.

```

from sklearn.inspection import PartialDependenceDisplay
fig, ax = plt.subplots(figsize=(15, 10))
ax.set_title("Partial Dependence Plots")

PartialDependenceDisplay.from_estimator(
    estimator=model,
    X=X_test,
    features=(0, 5, 2, 3), # the features to plot
    random_state=5,
    ax=ax,
)

plt.show()
```

In the following image, we see how changes in each feature’s value influence the prediction of the model:

![Partial dependence plots created with Scikit-learn]({{ site.baseurl }}/assets/images/posts/machine-learning-interpretability/partial-dependence-sklearn-1024x717.png)

If instead, we set the parameter `kind` to `‘individual’`, we create ICE plots. Let’s take a look:

```

fig, ax = plt.subplots(figsize=(15, 10))
ax.set_title("ICE Plots")

 PartialDependenceDisplay.from_estimator(
    estimator=model,
    X=X_test,
    features=(0, 5, 2, 3), # the features to plot
    kind = "individual",
    random_state=5,
     ax=ax, )
plt.show()
```

**![Individual conditional expectation plots for the California housing dataset. This visualization tool helps aid interpretability in machine learning models.]({{ site.baseurl }}/assets/images/posts/machine-learning-interpretability/ice-plots-1024x699.png)**

# Wrapping up

I hope I gave you a flavor of why interpretable ML is paramount, and a good overview of the different methods that you can use to interpret your models.

I showed you a few Python examples of how to interpret a machine learning model using open-source Python libraries. For more methods and details regarding these and other interpretability methods, check out our course “[Interpreting Machine learning Models](https://www.trainindata.com/p/machine-learning-interpretability)”.

![Machine learning interpretability online course]({{ site.baseurl }}/assets/images/posts/machine-learning-interpretability/interpretable-machine-learning-1024x576.png)

The code examples were taken from this Github repository: [https://github.com/solegalli/machine-learning-interpretability](https://www.trainindata.com/p/machine-learning-interpretability)
