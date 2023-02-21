---
layout: post
title: "Hyperparameter Tuning For Machine Learning: All You Need to Know"
author: sole
description: Discover what are and how to find the best hyperparameters for your machine learning models.
excerpt: Discover what are and how to find the best hyperparameters for your machine learning models.
categories: [Hyperparameter optimization, Python, Machine learning]
image: assets/images/posts/html/cover.gif
---

Landing on the best possible combinations of hyperparameters is one of the most important, 
as well as confusing, choices we could be faced with when developing a machine learning model. 
Even the most seasoned experts would agree that the algorithms and processes involved in 
choosing the best hyperparameters are highly complex. At the same time, they’re absolutely 
vital to get right.

Hyperparameters are the section of parameters that a user predefines to control the learning 
process. Their values are set before the learning process begins and help the machine 
learning model achieve the best performance on a particular task.

Hyperparameters are top-level parameters. Their values are set by the machine learning 
engineer before the learning algorithm begins training the model. They are used by the 
learning algorithm while it is still learning but are not a part of the resulting model. 
The trained model parameters that emerge after learning eventually become the model.

In short, hyperparameters are vital to creating the best model to predict a certain task. 
They are set before hand, that is, before building the model on the training data.

But before we understand how hyperparameters can improve the performance of a classifier 
or regression model, it’s necessary first to get a grasp of all the concepts involved. 
Let’s start with a basic tutorial.

*For a detailed tutorial and step by step code implementations on hyperparameter optimization 
for machine learning, check out our course 
[Hyperparameter Optimization for Machine Learning](https://www.trainindata.com/p/hyperparameter-optimization-for-machine-learning).*

 
## Parameter Vs. Hyperparameter

The terms parameter and hyperparameter may sound similar but are actually very different 
in machine learning terms. It can be quite confusing to understand the differences between 
these terms for a beginner. Maybe this break up will help:

Model parameters are critical to machine learning algorithms. They are usually internal 
to the machine learning model, and their values can be estimated from a dataset. They 
make up the actual pieces of the neural network, or the linear regression or the decision 
tree, and are required by the model when making predictions. A common example could be 
the weights within a neural network. Usually, those are determined by the algorithm as a
result of training.

Some common examples of parameters include:

- The coefficients (or weights) of linear and logistic regression models

- Weights and biases of a neural network

- The cluster centroids in clustering

- The point of partition at each node in a decision tree


Hyperparameters, on the other hand, are explicitly specified parameters that control the 
training process as a whole. They are parameters that cannot be learned from the regular 
training process and must be fixed before the process even begins. They express essential 
properties of the model, such as complexity or the learning rate.

![hyperparameter search cartoon]({{ site.baseurl }}/assets/images/posts/html/parameters.png)


Some common examples of hyperparameters include:

- Learning rate in optimization algorithms (for example, gradient descent)

- Choice of the optimization algorithm

- The number of hidden layers in a deep neural network

- The drop-out rate or the activation functions in a neural network

- Number of iterations (epochs) in training a neural network

- The regularization penalty in a linear model (Lasso vs Ridge)

- The depth of the decision tree

- The number of decision trees in a random forest

- The criterion to evaluate the splitting of samples at each node (for example Gini or entropy)

 
## Differences Between Parameters and Hyperparameters:

Now that we’ve gone through the meaning of both words in machine learning terms let’s take 
a look at some of the key distinctions that set them apart. 

- While model parameters are essential for making predictions, model hyperparameters are necessary for optimizing the model

- Parameters are specified or estimated during the actual training of the model, while hyperparameters are usually set before the beginning of the training

- Parameters are usually learned and set by the model, while hyperparameters are generally set manually by a machine learning engineer, or some sort of (additional) optimization method

- Parameters are internal to the model, while hyperparameters are external

- The value of a parameter can be estimated by optimization algorithms such as gradient descent, while the hyperparameter values are set by the process known as hyperparameter tuning

 
## What is Hyperparameter Tuning?

Let’s leave parameters aside for now and understand more about hyperparameters. To find 
and select the optimal set of hyperparameters for a machine learning model, we follow a 
process known as hyperparameter tuning or hyperparameter optimization.

This process involves trying various combinations of different hyperparameter values, 
training a model for each combination, and comparing the performance of the models on a 
validation set or test set. Once these processes are complete, we can arrive at the set 
of hyperparameters that work best for the final model.

A hyperparameters search consist of:

- Defining the hyperparameter space: the range of values to test for each hyperparameter

- Finding a method for sampling candidate hyperparameters (for example, grid search)

- A cross-validation scheme (usually hyperparameters are tested with cross-validation to obtain a measure of the dispersion error, obtained by assessing model performance over the data subset that was not used for model training)

- A performance metric to minimize (or maximize), like for example the ROC-AUC or the MSE

Hyperparameter tuning can be time-consuming when done manually but can also be carried out 
by several automated methods. These methods offer alternative ways to sample the candidate 
hyperparameter combinations, and they include, among others:

- Manual Search

- Grid Search

- Random Search

- Bayesian Optimization


Whichever way you decide to go about it, hyperparameters are a crucial part of juicing 
out the best performance from a machine learning model.

![hyperparameter search cartoon]({{ site.baseurl }}/assets/images/posts/html/hyperparameter-search.gif)


In case you’re still unsure about how this works, check out this simple example:

One of the most common examples of a hyperparameter is the learning rate. The hyperparameter 
controls how much the model needs to change in response to the estimated error when the 
model’s weights are updated. It’s a crucial parameter for building a neural network.

If we were to now optimize this learning rate hyperparameter, how would we go about it? 
We could, for example, set this rate to 0.01 before the first training session. If we then 
determine that 0.01 is too high, we could set the learning rate to something like 0.003 
for the next training session. Using similar logic, we would set the value higher in 
case it’s too low. This is hyperparameter tuning, explained in the simplest of terms. 

 
## Why is Hyperparameter Optimization Important?

Hyperparameter tuning is an essential part of controlling the machine learning model. 
Without it, the model parameters don’t produce the best results. This could mean higher 
errors for the model, or in other words, reduced performance, which is not what we want. 

Hyperparameter tuning, thus, controls the behavior of a machine learning model. It ensures 
that the model gives us good, accurate results.

So how do you find the ideal set of hyperparameters? Read on to find out.

 
## Methods to Find Hyperparameters

As we’ve now established, it’s absolutely essential to select the correct set of hyperparameters 
to train a machine learning model and improve its performance. The model parameters have 
to be tweaked to successfully carry out parameter tuning, which can be quite a time-consuming 
process. Often, hyperparameter optimization can be carried out manually, but in other cases 
can be automated, which saves a lot of time. Let’s look at some of the most common methods 
used below.

### Manual Search

One approach to finding the best set of hyperparameter values for an algorithm is to adjust 
them manually.

To manually tune hyperparameters, developers need to try a different combination of 
hyperparameters and select the model that performs the best. Manual hyperparameter optimization 
consists in looping through different hyperparameter values, and evaluating the model 
performance for each combination. We would then select the combination of hyperparameters 
that returns the best performing model.

The idea here is first to take big jumps in the hyperparameter values, followed by small 
jumps to focus on a specific value that makes the model perform better. 

As an example, let’s say we are trying to optimize the n_estimators and max_depth for an 
XBGClassifier model. We would start comparing the performance of models with 10, 100 and 
1000 estimators, and None, 1 or 4 of depth. And then, we would narrow down the search to 
those values that returned the best-performing model so far. So, let’s say that the model 
with 100 estimators worked best; we would then narrow down the search to values between 
100 and 200, and so on.

A manual search can be a long and tiresome process; however, it helps the data scientist 
get a better feeling and understanding of how the hyperparameters affect the model's 
performance, and for simpler models, it is very easy to find the best hyperparameters 
manually, without the need to go into more computationally expensive methods.

### Grid Search

Grid Search is the most basic method for hyperparameter optimization. Once the machine 
learning engineer figures out which values of hyperparameters they want to assess, the 
grid search will compute all possible combinations of those values.

Back to the example of the XBGClassifier, if we want to evaluate the following values 
for n_estimator: `[10, 100, 500, 1000]` and for max_depth: `[None, 1, 2, 3]`, a Grid Search 
will create all possible combinations with these values (4x4=16 combinations), resulting in:

```
[10, None], [10, 1], [10, 2], [10, 3]

[100, None], [100, 1], [100, 2], [100, 3]

[500, None], [500, 1], [500, 2], [500, 3]

[1000, None], [1000, 1], [1000, 2], [1000, 3]
```

The search would then proceed to train 16 machine learning models with each of these 
combinations, determine the performance of each model, and select the combination of 
hyperparameter values that returns the best value for the performance metric.


### Random Search

Unlike a Grid Search, which tries all combinations of hyperparameters, a Random Search 
runs the model through random hyperparameter combinations. In addition, unlike in Grid 
Search, in Random Search the data scientist defines distributions for each hyperparameter 
instead of specific values.

So for example, for the hyperparameter max_depth, we could define a uniform distribution 
constrained between 1 and 5. For the hyperparameter n_estimators we could define a uniform 
distribution between 10 and 1500. For a learning rate we could define a log-uniform distribution 
between 0 and 1. For categorical hyperparameters like the method to evaluate the decrease in 
impurity, we would still need to define each value, for example Gini or entropy. 

In Random Search, hyperparameter values will be sampled at random from these distributions. 
Given that, very often, similar values of hyperparameters return similarly performing models, 
it is not necessary to assess every possible value. Examining values at random will be more 
than enough to find the area or specific range of values that return the best performing models.

This decreases the time and complexity it takes to find the best hyperparameters, as well 
as the total number of combinations on which the models are trained. As a result, it’s 
usually the preferred model when compared to Grid Search.

The Random Search method also ensures that we don’t end up with a biased model that relies 
on value sets chosen arbitrarily by users, as is the case with a manual search.

### Bayesian Hyperparameter Optimization

Bayesian Optimization aims to solve some of the drawbacks of Random Search. A Random Search 
may end up evaluating too many unsuitable combinations of hyperparameters, simply because 
it determines the combinations at random.

In Bayesian Optimization, the search is guided. Thus, the hyperparameter search would spend 
more time looking around those values that have already shown promise. Thus, it reduces 
the amount of time it takes to compute a large number of hyperparameter combinations. It 
is also automated.

When looking for the  best hyperparameter combination in data science, we usually define 
what is called an objective function. This objective function depends on the machine 
learning model we want to train, the metric we want to optimize (for example, the accuracy), 
the training set and the hyperparameters. The objective function is black box. This means 
that it is not possible to know its shape or distribution. In Bayesian optimization, the 
function is “guessed” or modeled using Gaussian processes or Tree-structured Parzen 
Estimators (TPE).

We then use a follow up function to determine the space with the most promising hyperparameter 
values. These functions are called acquisition functions, and among them we find the 
expected improvement (ei) or the probability of improvement (pi).

Grid Search and Random Search don’t take into account previous iterations’ results when 
choosing the next hyperparameter value. Bayesian optimization does. For this reason, it 
is considered to provide some of the best hyperparameters.

In bayesian Optimization, we first perform a Random Search sampling at random a few model 
hyperparameter combinations. After these iterations, Bayesian Optimization applies a 
probabilistic function to select the hyperparameter values that works best within the 
machine learning model. In following iterations, the search will be carried out, one set 
of values at a time, selecting from values close to those that were deemed to be the best.

While Grid Search and Random Search can run in parallel, this is, different hyperparameter 
combinations can be evaluated at the same time, in Bayesian Optimization, the hyperparameter 
values are examined in sequence. We need to know the performance of the first sets of 
hyperparameter to inform the search for the subsequent values.

Hence, in theory, Bayesian Optimization will take fewer iterations to find the best 
hyperparameters, but because it is sequential, the overall wall clock time may not be 
reduced. Thus, Bayesian optimization is suitable when searching for hyperparameter of 
computationally costly machine learning models, like deep neural networks, because overall, 
we end up training fewer models.

 
## Python Tools For Hyperparameter Optimization

Now that you know the distinction between the hyperparameter tuning methods, you should 
have a better idea of which fits your machine learning model best. The next step, in this 
case, would be to use a tool to apply the different hyperparameter search algorithms. 
Several open-source Python tools are already available for this purpose, and each tool 
has its own pros and cons. Here are some of the most popular ones.

### Scikit-learn

Scikit-learn is the go-to Python library for machine learning, hosting a wide array of 
predictive algorithms. [Scikit-learn](https://scikit-learn.org/stable/modules/grid_search.html) 
is also the go-to library to optimize the hyperparameters of its own algorithms.

Scikit-learn supports Grid Search and Random Search with cross-validation through the 
GridSearchCV and RandomizedSearchCV. Scikit-learn also offers support for multi-fidelity 
hyperparameter optimization through successive halving, which he have not covered in this 
tutorial.

The entire Scikit-learn package provides a range of machine learning-related modules that 
can be applied with minimum effort. It comes with built-in with tasks that allow you to 
carry out any activity —from preprocessing to using the sklearn.model_selection module to 
select the best hyperparameters.

Scikit-learn, and other similar packages for hyperparameter tuning, such as scikit-optimize 
and scikit-hyperband, have a consistent interface across many implemented classes, providing 
a massive benefit to data scientists. Built on top of NumPy and SciPy it also provides 
tools for data analysis and data mining, making it one of the engineers’ favorite packages.

![open source Python libraries for hyperparameter optimization in machine learning]({{ site.baseurl }}/assets/images/posts/html/libraries-comparison.png)


### Scikit-optimize

[Scikit-optimize](https://scikit-optimize.github.io/stable/) is built on top of Scikit-learn, 
and it extends its functionality by supporting Sequential model-based optimization in Python, 
that is, Bayesian Optimization. Through the class BayesianSearchCV, we can set up Bayesian 
optimization using Gaussian processes or Random forests to model the objective function, 
just as we would do with the GridSearchCV and RandomizedSearchCV from Scikit-learn.

Scikit-optimize also provides functions out-of-the-box to produce visualizations of the 
hyperparameter search, and better understand how the different hyperparameter values 
affect the model performance.

### Optuna

[Optuna](https://optuna.readthedocs.io/en/stable/index.html) is a Python library that’s best 
suited for automated hyperparameter optimization. It supports different hyperparameter 
search algorithms, such as Grid Search, Random Search, and Bayesian optimization using 
Tree-Structured Parzen Estimators to compute optimal hyperparameters for a machine learning model.

The beauty of Optuna is that it has a define-by-run way of setting up the hyperparameter 
search function. This means, pretty much, that you can optimize with ease almost any hyperparameter 
from your desired machine learning model. You are not constrained by the design of the model 
or the API you are using to develop it. Hence, Optuna is, in my opinion, the most versatile 
hyperparameter search library, and I expect it to be the most widely used.

With Optuna, it’s possible to create your own hyperparameter tuning method class. Optuna 
also integrates with other popular packages, such as Scikit-learn, to give you a smooth 
user experience. It provides easy scalability with little or no changes to the code and 
is easy to install since it has few requirements. What more could you ask for?

### Hyperopt

[Hyperopt](http://hyperopt.github.io/hyperopt/) is another Python library that uses Bayesian 
optimization with Tree-Structured Parzen Estimators (TPE) as well as other learning algorithms 
like Random Search and Simulated Annealing (SA).

The highlight of Hyperopt is that it allows you to create very complex parameter spaces as 
well as easily configure your search space. It was one of the first available libraries 
for hyperparameter optimization, hence the most popular back then and remaining very popular 
today as well.

Beyond that, Hyperopt allows you to pause the optimization process, save important information, 
and resume later. It also gives engineers the capability to distribute their computation over 
a cluster of machines, easing the workflow.

What’s more? It works with various support frameworks, including XGBoost, Pytorch, Tensorflow, 
and Keras.

On the downside, the documentation for Hyperopt is a bit slim.

### Keras Tuner

Built with the Keras API, [Keras Tuner](https://keras.io/keras_tuner/) is a scalable hyperparameter 
tuning framework that is simple and hassle-free to use. It allows you to easily configure 
your search space with a define-by-run syntax and uses search algorithms to find the best 
hyperparameter values for your models.

It comes with built-in Bayesian Optimization, Random Search, and Hyperband algorithms and is 
also designed to allow engineers to experiment with newer search algorithms. It can be used 
to get the best parameters for deep learning models and can promise some of the highest 
accuracies that can be achieved with the defined combinations.

### Ray-tune

[Ray-Tune](https://docs.ray.io/en/latest/) is another great Python library for hyperparameter 
tuning at any scale. It supports most machine learning frameworks (Scikit-learn, Keras, 
TensorFlow, Random Forest among others). It also runs some of the top learning algorithms 
such as Population Based Training, and Hyperband. It also integrates with a wide range of 
additional hyperparameter optimization tools.

When using Ray-Tune, you can also scale a hyperparameter search from a single machine to a 
large distributed cluster without having to change your code. To top it off, it’s also 
very fast.


## Conclusion

At this point, you may be a little overwhelmed with the amount of information that you’ve j
ust read. Understanding such complex concepts takes time, effort, and several tutorials 
to master properly. 

By now, however, you should at least have a better understanding of hyperparameter tuning, 
why it matters, the kinds of hyperparameters that you can optimize, the hyperparameter s
earch methods that you can use, as well as some of the python tools that can help you out 
along the way.

Still confused?

That’s alright. This article is meant to be a brief introduction and by no means as 
informative as a college degree. But if you’re looking for a more detailed tutorial on how 
to use these tools and optimize your set of hyperparameters, that’ll give you all the 
expertise needed, check out our online course 
[Hyperparameter Optimization for Machine Learning](https://www.trainindata.com/p/hyperparameter-optimization-for-machine-learning).

For a small fee, this course on hyperparameter optimization offers 10 hours of video 
tutorials, fun interactive activities like quizzes and assignments, a Q&A section, and 
lifetime access. By the end, you’ll be creating the best-performing machine learning models, 
and have a number of techniques to select the best hyperparameters at your disposal.

 If you don’t? There is a 30-day money-back guarantee. What more could you ask for?
 