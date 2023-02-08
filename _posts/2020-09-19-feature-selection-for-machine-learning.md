---
layout: post
title:  "Feature Selection in Machine Learning"
author: sole
description: Discover why and how we select features in machine learning.
excerpt:  Discover why and how we select features in machine learning.
categories: [ Feature selection, Python, Machine learning ]
image: assets/images/posts/featsel/cover.png
---

When building a machine learning model for a business problem, it’s rare that all the variables 
in the data will need to be incorporated into the model. Sure, adding more variables rarely 
makes a predictive model less accurate, but there are certain disadvantages to including 
an excess of features.

It is commonly said that data scientists spend a huge amount of time doing data preprocessing, 
feature engineering, and feature extraction. Feature selection is also a very common step 
in many data science projects, and we usually test more than one feature selection method 
to find the best subset from the original features.

In this article, I discuss the importance of selecting features in machine learning. I 
highlight why we should select features when using our models for business problems. And 
then go over the main feature selection algorithms.

What we’ll cover:

- What is feature selection in machine learning?

- Importance of feature selection in machine learning

- Feature selection methods: filter, wrapper, embedded and hybrid

Let’s get started.


*For tutorials and step by step code implementations on additional feature selection methods, check out our course 
[Feature Selection for Machine Learning](https://www.trainindata.com/p/feature-selection-for-machine-learning) or our 
book [Feature Selection in Machine Learning with Python](https://leanpub.com/feature-selection-in-machine-learning/)*.

<iframe width="560" height="315" src="https://www.youtube.com/embed/blvmNWbcPDo" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## What is feature selection in machine learning?

Feature selection is the process of identifying and selecting a subset of variables from 
the original data set to use as inputs in a machine learning model.

A data set usually contains a large number of features. We can employ a variety of methods 
to determine which of these are actually relevant features when making predictions.

Each of the different methods have advantages and disadvantages to consider. But why should 
we select features to begin with?

![feature selection]({{ site.baseurl }}/assets/images/posts/featsel/FS-arrow-gif.gif)


## Importance of feature selection in machine learning

At a glance, it may seem that the more information one feeds to the machine learning model, 
the more accurate it will be. However, simpler models are more effective in a variety of 
ways, particularly when used in an organization.

Machine learning models with fewer features are:

- Easier to interpret

- Less redundant (have reduced feature redundancy)

- Faster to train

- Easier to deploy to production

If a model has 10 predictive variables, instead of a hundred, people can understand the 
effect of the variables on the outcome much more clearly.

Being able to interpret the result of a machine learning model is important when the model 
is used in real life, say for example to prevent fraud. When an observation is flagged as 
potentially fraudulent by a machine learning classifier, the fraud investigators will 
investigate it further. It helps them, if there are a few, clear indicators, the variables 
in the model, that point them in the right direction.

Oftentimes many of the variables in our datasets are redundant. Redundant variables won’t 
add information to the machine learning model. In fact, in some cases, large feature 
subsets may reduce the performance of some machine learning models. In particularly, those 
derived from tree-based algorithms.

Additionally, simpler models require shorter training times. Training a machine learning 
model using less variables, reduce the computational cost. This speeds up model training 
and also allows the models to score the incoming data much faster. If a business intends 
to use the model to make real time decisions, speed is particularly important.

Simpler models require simpler production code. For a model with less features, we need 
less error handling code and less unit testing. This makes model deployment easier to implement.

Models that use less features also have reduced the risk of data dependent errors; if a 
business uses data collected from a third party API, the more variables in the model, the 
more exposure they have to any errors or, simply, changes made by the third party, or even 
worse, a temporary shut-down of the API.

Simpler models could generalize better and therefore show reduced overfitting. Too many 
variables often add noise to the model rather than predictive value. This is often referred 
to as the curse of dimensionality. Eliminating noisy and irrelevant features could make 
the model generalize better to new, unseen data, which gives way for more accurate predictions.

So selecting feature when using machine learning models in business, is important. How can 
we then select the most predictive features?

![feature selection methods]({{ site.baseurl }}/assets/images/posts/featsel/data-analysis.gif)


With the following feature selection techniques, an organization can benefit from simpler 
models while still achieving effective and robust machine learning models.

## How do we select features?

A feature selection procedure combines a search technique with an evaluation method. The 
search technique proposes new feature subsets, and the evaluation measure determines the 
how good the subset is.

In a perfect world, a feature selection method would evaluate all possible subsets of feature 
combinations and determine which one results in the best performing regression model or 
classifier.

However, computational cost inhibits such a practice in reality. In addition, the optimal 
subset of features varies between machine learning models. A feature subset that optimizes 
one model’s performance won’t necessarily optimize another’s.

To combat these challenges, there are many feature selection algorithms we can use to find 
a close-to-optimal subset of features in a manner that is computationally cost efficient.

Generally, these methods have characteristics that allow them to group into one of three 
categories: filter methods, wrapper methods, and embedded methods.

**Filter methods** select features based on characteristics in the data. They look at each 
feature individually, or compare them to other features, and assess how important they are.

**Wrapper methods** examine all or almost all possible feature combinations to identify the 
optimal feature subset. Because of this, they are known as “greedy” algorithms.

**Embedded methods** describe selection procedures that occur alongside fitting the machine 
learning model. Combining these two processes allows them to consider the interaction 
between the model and the features. And they are less computationally costly than wrapper methods.

Hybrid methods are feature selection algorithms that contain some characteristics of wrapper, 
some of filter and some of embedded methods.

Now that I’ve introduced the categories of feature selection procedures, I’ll go into more 
detail about each, highlight the advantages and disadvantages.

## Filter methods

A typical filter algorithm consists of two steps: it ranks features based on certain criteria 
and then chooses the highest-ranking features to train the machine learning models.

Filter methods are generally univariate, so they rank each feature independently of the rest. 
Because of this, the filter methods tend to ignore any interactions that occur between 
features. Thus, redundant variables will not necessarily be eliminated by filter methods.

However, some multivariate filter selection methods exist as well. These consider features 
in relations to others in the data set, making them naturally capable of handling redundant 
features. Their selection criteria scans for duplicated features and correlated features 
and provide simple but powerful methods to quickly remove redundant information.

The filter methods are basic, but essential. Though they don’t usually find the best feature 
subset, they are model agnostic and are often an easy first step to reduce the feature space 
without significant computational cost.

I’ll describe some of the popular filter methods here, as well as provide insight about 
when to use them.

### Constant, quasi-constant, and duplicated features

The most basic and intuitive methods for feature selection consist of removing constant, 
quasi-constant, or duplicated features.

Constant features only show one value for all the observations in the data set. That is, 
they show absolutely no variability.

Quasi-constant features are similar; if most observations share the same value, then we’d 
label that feature quasi-constant. In practice, quasi-constant features typically refer to 
those variables where more than 95 to 99 percent of the observations show the same value.

Constant and quasi-constant features can be eliminated by fixing a variance threshold, and 
then removing those features whose variance is below that threshold. The open-source 
library sklearn has a dedicated transformer to do this: VarianceThreshold.

Feature-engine, also allows you to drop constant and quasi-constant features through the 
DropConstantFeatures() class.

Duplicated features, as the name indicates, are those that are in essence, identical. That is, 
for every observation, they show the same value.

Removing constant and duplicated features can reduce the feature space dramatically.

![removing constant and duplicated features]({{ site.baseurl }}/assets/images/posts/featsel/paper-function-gif.gif)

Although it sounds obvious and overly simple, many datasets contain a lot of constant, 
quasi-constant, and duplicated features. In fact, duplicated features often arise when 
generating new features by one-hot encoding of categorical variables. Removing these features 
is an easy but effective way to reduce the dimension of the feature space, without losing 
any significant information.

### Correlation

Correlation measures the association between two or more variables. The Pearson correlation 
coefficient is one measure of association, that determines the linear association between 
2 continuous variables. There are other coefficients, like Kendall’s tau or Spearman’s 
that quantify non-linear associations.

The higher the correlation, the more linearly associated the variables are. The central 
hypothesis is that good feature sets contain features that are highly correlated with the 
target, yet uncorrelated with each other.

If two variables are correlated, we can predict one from the other. Therefore, if two 
features are correlated, the model only really needs one of them, as the second one does 
not add additional information.

![correlation measures the linear association between variables]({{ site.baseurl }}/assets/images/posts/featsel/linear-regression.png)

Though correlated features shouldn’t diminish the accuracy of the model, there are certain 
detriments to having them. For linear models, like linear regression or logistic regression, 
multi-collinearity may reduce performance or mask the true contribution of each feature to 
the prediction.

For decision trees and random forests, the model building methods will assign roughly the 
same importance for both correlated features, but this would be roughly half the importance 
it would assign if there were just one feature present, affecting the model interpretability.

### Statistical measures

The filter methods discussed so far evaluate features individually or compare features to 
other features. Statistical methods for feature selection, evaluate features in light of 
the target. In short, the compare the distribution of the features when the target is of 
class 0 vs 1.

![statistical tests are used to rank features]({{ site.baseurl }}/assets/images/posts/featsel/rank-features.png)

Statistical tests compare the distribution of the features when the target is of class 0 
with that shown when the target is of class 1. Then, it assigns a statistical value or 
p-value depending on how different the distributions are.

With this value, it ranks the features. And then the top ranking features are selected.

**Mutual information** measures the mutual dependence between two variables, in this case, the 
feature and the target. Mutual information is similar to correlation, but more general; 
it doesn’t strictly represent linear association. It measures how much knowing one of 
these variables reduces uncertainty in the other.

The **chi-square test** uses the chi-Square distribution to measure the dependency between two 
variables. It is suitable when evaluating categorical features against a binary or multi-class 
target variable.

The chi-square test essentially compares the actual frequencies of the values of a variable 
with the expected frequencies if it had no relation to the target.

For example, we can think of Titanic survivors. If there was no relationship between gender 
and survival, about 50 percent of the survivors would be male and 50 percent female.

However, it turns out about 75 percent of the survivors were female, suggesting that gender 
and survival have a significant relationship. Thus, the variable gender is very likely, a 
good predictor of the survival.

![contingency table used by the chi-square test]({{ site.baseurl }}/assets/images/posts/featsel/contingency-table.png)


**ANOVA** is another method to measure dependencies between two variables, but unlike the 
chi-square test, is suited to continuous variables. It requires a binary target, and it 
essentially compares the distribution of the variable when the target is one versus the 
distribution when the target is zero.

ROC-AUC or RMSE measure the performance of a model for either classification or regression 
methods, respectively. We can select features by building a machine learning model using 
only one feature and then evaluating the model’s performance with one of these metrics. 
After repeating this for every feature, we can rank the features with this metric, then 
select the top-ranking ones.

## Wrapper methods

In comparison to filter methods, wrapper methods tend to be more computationally expensive, 
but select a better set of features.

Wrapper methods use a specific machine learning algorithm in the selection process and 
choose the best subset of features tailored for that algorithm. This subset of features 
may not be optimal for a different machine learning model. In other words, wrapper methods 
are not model agnostic.

A typical wrapper method will perform the following:

![wrapper methods procedure]({{ site.baseurl }}/assets/images/posts/featsel/FS_4.png)


Wrapper methods start by searching through different subset of features, then creating a 
model with each. They evaluate these models to select the best one, and afterwards, they 
iterate to define a new subset based on the previous best subset.

Deciding when to stop this search comes down to monitoring whether the performance doesn’t 
increase or decrease beyond a certain threshold, depending on what method you’re using. 
These thresholds are often arbitrary and defined by the user.

I’ll discuss these procedures in more details for specific wrapper methods, including 
forward feature selection, backward feature selection, and exhaustive search.

### Forward selection

Forward feature selection algorithm begins by evaluating all feature subsets that consist 
of only one input variable. It selects the “best” feature and afterwards, adds all the 
other features to it, individually, and selects a second feature that creates the new best 
performing model.

The process repeats over and over, adding one feature at a time, until it meets certain 
criteria. After adding additional features to the subset, if the machine learning model 
performance doesn’t improve by more than a specific threshold, then we can stop the search 
and select this feature subset.

![forward feature selection]({{ site.baseurl }}/assets/images/posts/featsel/FS_5.png)


### Backward selection

Backward feature selection works in an opposite way. It begins by building a machine 
learning model with all of the features. Then creates new subsets by removing every feature, 
one at a time, and training a new machine learning model.

It evaluates each subset to find the best one, then continues the process of removing a new 
single feature that results in the best model.

![backward feature selection]({{ site.baseurl }}/assets/images/posts/featsel/FS_6.png)

The process stops when an additional feature is removed and the performance doesn’t decrease 
past a certain arbitrary threshold.

### Exhaustive search

An exhaustive search is the most “greedy” method. It tries all combinations of features, 
for any number of features from one until the maximum number of features available.

This method is extremely computationally expensive, almost prohibitively, but would provide 
the best subset of features, at least in theory.

For example, if there were four features to choose from (A, B, C, and D), an exhaustive 
search would build and evaluate models for fifteen feature subsets.

It would go through all possible groupings: each feature on its own, every combination of 
paired features, every threesome combination, and the set of all features, as depicted below.

![exhaustive search for features]({{ site.baseurl }}/assets/images/posts/featsel/FS_7.png)


## Embedded methods

Finally, we have embedded methods. These encompass the benefits of both the wrapper and 
filter methods, by evaluating interactions of features but also maintaining reasonable 
computational cost.

The typical steps for embedded methods involve training a machine learning algorithm 
using all the features, then deriving the importance of those features according to the 
algorithm used.

Afterwards, it can remove unimportant features based on some criteria specific to the 
algorithm, of which I’ll cover some examples of shortly.

### Lasso

The Lasso regularization adds a penalty to different coefficients to reduce their freedom, 
thereby reducing its tendency to fit the noise. The higher the penalty, the less the model 
will overfit and the better it will generalize to unseen data.

During the Lasso fitting algorithm, the model tries to minimize the difference between the 
predicted and estimated value of the observation with the penalty.

Lasso can shrink some coefficients of the linear regression to zero. This indicates that 
the predictor can essentially be multiplied by zero to estimate the target and consequently 
doesn’t add to the overall prediction of the output.

In this way, Lasso regularization helps determine which features can be removed from the model.

### Regression coefficients

Linear models aim to predict a target based on given variables by assigning a coefficient to 
each. They follow the form of this equation:

![linear regression equation]({{ site.baseurl }}/assets/images/posts/featsel/FS_8.png)


Here, each variable X is multiplied by a coefficient, beta, to predict the final value of y. 
The betas here are directly proportional to how much the feature contributes to y. This holds 
true for any regularized or unregularized linear methods. Consequently, we are able to use 
these coefficients to select features.

There are a few caveats, however: this method assumes there is a linear relation between the 
predictors and the target and the scale of the features affects interpretability of the 
coefficients.

Some features may only contain values between zero and one, while others may range into 
the thousands or greater, which would affect the size of the coefficient. To truly compare 
features, we’d need to put all the features into similar scales.

If all of these conditions hold and the features are in a similar scale, we can safely infer 
that the coefficients align with variable importance and therefore remove any that have 
negligible coefficients.

### Decision tree feature importance

Random forests and decision trees are generally very popular among the machine learning 
algorithms because they provide good predictive performance, low overfitting, and easy 
interpretability.

Part of the interpretability encompasses how straightforward it is to derive the importance 
of each feature on a decision, which makes it one of the best embedded methods of feature selection.

Random forests are a collection of decision trees. Each decision tree is comprised of a 
series of “questions” based on different features in the data set.

For each feature, the decision tree formulates a question of the type “is the value of 
the observation bigger than a for feature x?”

If the answer is yes, the observation is allocated on one side of the node, if it is no, 
the data is taken to the other side of the node.

The answer leads to the highest possible decrease of impurity, or in other words, the 
maximum information gain, meaning it gives the best possible separation of the class.

![diagram of a decision tree]({{ site.baseurl }}/assets/images/posts/featsel/FS_9.png)

Random forests consist of hundreds to thousands of decision trees built over a random 
extraction of the observations and features from the data set. The random extraction helps 
train de-correlated trees, and thus improve generalization.

Each node of the decision tree separates the data into two buckets. The importance of the 
feature is derived by combining the height of the node in which the feature is used, with 
how many different trees use the feature, and by how much the impurity is decreased.

We can therefore select features by building a random forest, looking at the importance 
derived for each feature, and selecting the most important ones.

## Hybrid methods

As the name might suggest, hybrid methods combine pieces of wrapper methods and embedded 
methods. They build several algorithms at each round of selection like wrapper methods, 
but they don’t examine every possible feature combination.

In addition, hybrid feature selection algorithms select features based on a specific 
machine learning model and evaluate the model performance.

Some common hybrid methods are recursive feature elimination and recursive feature addition.

### Recursive feature elimination

Recursive feature elimination (RFE) consists of the following steps:

![Recursive feature elimination procedure]({{ site.baseurl }}/assets/images/posts/featsel/FS_10.png)

It starts with ranking the feature with the importance derived from an embedded method, 
such as those discussed earlier.

Next, we remove the least important feature and build a new machine learning algorithm. We 
then calculate a performance metric, such as ROC-AUC, MSE, or RMSE.

If the metric decreases by more than an arbitrarily set threshold, then the feature should 
be kept. Otherwise, we can remove that feature and repeat this process until a feature 
removal causes the performance metric to decrease past this threshold.

The difference between this method and the step backward feature selection method is that 
here, we don’t remove all features first in order to determine which to remove. We simply 
select the least important feature based on what the machine learning model derives as 
important. It removes only one feature, rather than all the features at each step.

Because of this, in recursive feature elimination we train significantly less number of 
models compared to step backward feature selection.

### Recursive feature addition

The recursive feature addition algorithm works similarly, but adding features one at a 
time rather than removing one.

![recursive feature addition procedure]({{ site.baseurl }}/assets/images/posts/featsel/FS_11.png)

It begins by ranking the features with one of the embedded methods, then builds a model 
with only the one most important feature.

Again, we calculate the performance metric of our choice. Then we add the second most 
important feature and train a model. And if the metric increases over an arbitrarily set 
threshold, then we should keep the feature. Otherwise, it can be removed.

This process will repeat until all features have been evaluated. Similar to recursive 
elimination, it differs from step forward feature selection because it doesn’t add all 
features to determine what to keep, but instead, just adds the most important one.

This method and the recursive feature elimination method usually work faster than wrapper 
methods and better than embedded methods. They account for correlations depending on how 
stringent the threshold is set, but on the downside, this threshold is, again, set arbitrarily; 
the smaller the threshold, the more features will be selected.

### Random shuffling of the feature values

Another popular method of feature selection consists in randomly shuffling the values of a 
specific variable and observing how the permutation affects the performance metric of the algorithm.

This is a good alternative to embedded methods, for those artificial intelligence algorithms 
that do not derive feature importance during the induction, like for example, nearest 
neighbors or support vector machines (svm). However, this is also commonly used with linear 
and decision tree based models.

If the variable is important, the random permutation will decrease the performance metrics 
dramatically. On the other hand, permuting unimportant variables should have little to no 
effect on the performance of the model. Thus, this information can be used to determine 
which variables are predictive enough to keep in the feature subset.

![permutation feature importance]({{ site.baseurl }}/assets/images/posts/featsel/FeatureShufflingGIf.gif)

## Feature selection vs dimensionality reduction

Very often, students of my course “Feature Selection for machine learning model”, ask me 
why I do not cover PCA in the course. The thing is, principal component analysis (PCA) is not 
a feature selection method. It is a dimensionality reduction procedure. With PCA, we combine 
the existing features into principal components, and then we select a subset of them. But 
we still need all the features to create the few selected PA. So it does not really reduce 
the amount of effort needed to deploy a model to production.

In short, in feature selection, we select a subset of features from the data set to train 
machine learning algorithms. They do not alter the original features distribution. 
Dimensionality reduction methods, like PCA, alter the original representation of the variables.


## Feature selection and deep learning

Many students ask me if the methods that I teach in my course are also suitable for deep 
learning. The thing is, deep learning models give us an advantage over traditional 
off-the-shelf algorithms only when we have high-dimensional data with lots of observations 
and features. Only in these cases will neural networks return models that outperform 
traditional machine learning algorithms.

Selecting features is commonly done when training traditional machine learning models. 
Hence, all the methods that have been developed and discussed in the literature are suitable 
only for these methods.

## Wrapping up

Feature selection in machine learning models allows for accurate models that are simpler 
and less computationally expensive to implement.

The methods I’ve outlined in this blog post should serve as a guide for selecting features 
to optimize a model in a variety of different situations.

Though there isn’t one perfect method, each method provides advantages and disadvantages, 
and between all the methods available, one should be able to find one to optimize the 
machine learning model.


## References

- [Feature Selection for Machine Learning](https://www.trainindata.com/p/feature-selection-for-machine-learning) — Online course for beginners and intermediate data scientists

- [Feature Selection in Machine Learning with Python](https://leanpub.com/feature-selection-in-machine-learning/) — Book

- [Feature Selection with Feature-engine](https://feature-engine.trainindata.com/en/latest/user_guide/selection/index.html) — Python open source library

- [Feature Selection with Scikit-learn](https://scikit-learn.org/stable/modules/feature_selection.html) — Python open source library
