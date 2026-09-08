---
layout: post
title: "Hyperparameter Tuning For Machine Learning"
author: sole
description: "Learn about grid, random search, and Bayesian optimization for hyperparameter tuning for machine learning, and their Python implementation."
excerpt: "Learn about grid, random search, and Bayesian optimization for hyperparameter tuning for machine learning, and their Python implementation."
categories: [Hyperparameter Optimization, Machine Learning]
image: assets/images/posts/hyperparameter-tuning-for-machine-learning/cover-4.gif
---

Hyperparameter optimization is the key to unlocking a machine learning model‘s full potential, ensuring it performs at its best on a given task.

Hyperparameters are user-defined configuration settings that guide the learning process and drive the model to peak performance.

Discover various techniques for finding the optimal hyperparameters, including grid search, random search, and Bayesian Optimization, as we delve into their strengths and limitations in this article.

For a comprehensive tutorial and Python code implementations on hyperparameter optimization for machine learning, explore our course [*Hyperparameter Optimization for Machine Learning*](https://www.trainindata.com/p/master-hyperparameter-optimization-for-tabular-learning)*.*

[![Hyperparameter optimization for machine learning course]({{ site.baseurl }}/assets/images/posts/hyperparameter-tuning-for-machine-learning/hyperparameter-tuning-course-1024x576.png)](https://www.trainindata.com/p/master-hyperparameter-optimization-for-tabular-learning)

## Model Parameters

Model parameters are the innate building blocks of a machine learning algorithm, derived from a dataset during the learning process. They embody the essence of a neural network, linear regression, or a decision tree.

Common examples of model parameters encompass:

- Linear and logistic regression model coefficients (weights).
- Weights and biases within a neural network.
- Cluster centroids in clustering.
- Partition points at each node of a decision tree.

## Hyperparameters

In contrast, hyperparameters serve as the architects of the model, dictating the training process with settings predefined by the user. They encapsulate vital characteristics of the model, such as its complexity or the learning rate.

Notable examples of hyperparameters include:

- Learning rate for optimization algorithms (e.g., gradient descent).
- Choice of optimization algorithm.
- The depth or number of hidden layers in deep neural networks.
- Drop-out rate and activation functions in neural networks.
- Number of training iterations (epochs) for a neural network.
- Regularization penalty in linear models (Lasso vs. Ridge).
- Decision tree depth.
- The count of decision trees in a random forest.
- Criteria for evaluating sample splits at each node (e.g., Gini or entropy).

## Parameters Vs. Hyperparameters

Model parameters are essential for making predictions. Model hyperparameters are necessary for controlling the learning process to optimize the model’s performance.

Parameters are specified or estimated during the training of the model. Hyperparameters are set before the beginning of the training.

Parameters are internal to the model, while hyperparameters are external.

The value of a parameter can be estimated by optimization algorithms such as gradient descent, while the hyperparameter values are set by the process known as hyperparameter tuning

## Hyperparameter Tuning for Machine Learning

To find and select the optimal set of hyperparameters for a machine learning model, we follow a process known as hyperparameter tuning or hyperparameter optimization.

This process involves trying various combinations of different hyperparameter values, training a model for each combination, and comparing the performance of the models on a validation set or test set. Ultimately, it leads us to the optimal hyperparameters for the final model.

A hyperparameters search consists of:

- Defining the hyperparameter space: the range of values to test for each hyperparameter
- Finding a method for sampling candidate hyperparameters (for example, grid search)
- A cross-validation scheme (usually hyperparameters are tested with cross-validation to obtain a measure of the dispersion error, obtained by assessing model performance over the data subset that was not used for model training)
- A performance metric to minimize (or maximize), like for example the ROC-AUC or the MSE

Various hyperparameter search algorithms are at our disposal, including:

- Manual Search
- Grid Search
- Random Search
- Bayesian Optimization

## Why is Hyperparameter Optimization Important?

Hyperparameter tuning is an essential part of controlling the machine learning model. Without it, the model parameters don’t produce the best results. This could mean higher errors for the model, or in other words, reduced performance, which is not what we want.

Hyperparameter optimization, thus, controls the behavior of a machine learning model. It ensures that the model gives us good, accurate results.

So how do you find the ideal set of hyperparameters? Read on to find out.

## Hyperparameter Optimization Methods

There are various algorithms that we can use for hyperparameter optimization in machine learning. We can select hyperparameters manually, using Grid or Random search or with Bayesian Optimization. Let’s dive into each one of these processes.

### Manual Search

One approach to finding the best set of hyperparameter values for an algorithm is to adjust them manually.

When we manually tune hyperparameters, we try different combinations of hyperparameters and select the one that returns the best model performance. Manual hyperparameter optimization consists in looping through different hyperparameter values, and evaluating the model performance for each combination. We would then select the combination of hyperparameters that returns the best performing model.

The idea here is first to take big jumps in the hyperparameter values, followed by small jumps to focus on a specific value that makes the model perform better.

As an example, let’s say we are trying to optimize the n_estimators and max_depth for an XBGClassifier model. We would start comparing the performance of models with 10, 100 and 1000 estimators, and None, 1 or 4 of depth. And then, we would narrow down the search to those values that returned the best-performing model so far. So, let’s say that the model with 100 estimators worked best; we would then narrow down the search to values between 100 and 200, and so on.

A manual search can be a long and tiresome process; however, it helps the data scientist get a better feeling and understanding of how the hyperparameters affect the model’s performance, and for simpler models, it is very easy to find the best hyperparameters manually, without the need to go into more computationally expensive methods.

### Grid Search

Grid Search is the most basic method for hyperparameter optimization. Once the machine learning engineer figures out which values of hyperparameters they want to assess, the grid search will compute all possible combinations of those values.

Back to the example of the XBGClassifier, if we want to evaluate the following values for n_estimator: `[10, 100, 500, 1000]` and for max_depth: `[None, 1, 2, 3]`, a Grid Search will create all possible combinations with these values (4×4=16 combinations), resulting in:

```
[10, None], [10, 1], [10, 2], [10, 3]

[100, None], [100, 1], [100, 2], [100, 3]

[500, None], [500, 1], [500, 2], [500, 3]

[1000, None], [1000, 1], [1000, 2], [1000, 3]

```

The search would then proceed to train 16 machine learning models with each of these combinations, determine the performance of each model, and select the combination of hyperparameter values that returns the best value for the performance metric.

### Random Search

Unlike a Grid Search, which tries all combinations of hyperparameters, a Random Search runs the model through random hyperparameter combinations. In addition, unlike in Grid Search, in Random Search the data scientist defines distributions for each hyperparameter instead of specific values.

So for example, for the hyperparameter max_depth, we could define a uniform distribution constrained between 1 and 5. For the hyperparameter n_estimators we could define a uniform distribution between 10 and 1500. For a learning rate we could define a log-uniform distribution between 0 and 1. For categorical hyperparameters like the method to evaluate the decrease in impurity, we would still need to define each value, for example Gini or entropy.

In Random Search, hyperparameter values will be sampled at random from these distributions. Given that, very often, similar values of hyperparameters return similarly performing models, it is not necessary to assess every possible value. Examining values at random will be more than enough to find the area or specific range of values that return the best performing models.

This decreases the time and complexity it takes to find the best hyperparameters, as well as the total number of combinations on which the models are trained. As a result, it’s usually the preferred model method when compared to Grid Search.

The Random Search method also ensures that we don’t end up with a biased model that relies on value sets chosen arbitrarily by users, as is the case with a manual search.

### Bayesian Hyperparameter Optimization

Bayesian Optimization aims to solve some of the drawbacks of Random Search. A Random Search may end up evaluating too many unsuitable combinations of hyperparameters, simply because it determines the combinations at random.

In Bayesian Optimization, the search is guided. The hyperparameter search would spend more time looking around hyperparameter values that have already shown promise. Thus, it reduces the amount of time it takes to sample the hyperparameter space.

## How Does Bayesian Optimization work?

When looking for the best hyperparameter combination, we usually define what is called an objective function. This objective function depends on the machine learning model we want to train, the metric we want to optimize (for example, the accuracy), the training set and the hyperparameters. The objective function is black box. This means that it is not possible to know its shape or distribution.

Bayesian optimization tries to “guess” or model the objective function using Gaussian processes or Tree-structured Parzen Estimators (TPE). We then use a follow up function to determine the space with the most promising hyperparameter values. These functions are called acquisition functions, and among them we find the expected improvement (ei) or the probability of improvement (pi).

In bayesian Optimization, we first perform a Random Search sampling at random a few model hyperparameter combinations. After these iterations, Bayesian Optimization applies a probabilistic function to select the hyperparameter values that works best for the machine learning model. In following iterations, the search will be carried out, one set of values at a time, selecting from values close to those that were deemed to be the best.

Hence, Bayesian Optimization will take fewer iterations to find the best hyperparameters, but because it is sequential, the overall wall clock time may not be reduced.

## Which hyperparameter tuning method should I use?

Bayesian optimization is suitable when searching for hyperparameter of computationally costly machine learning models, like deep neural networks, because overall, we end up training fewer models.

In Bayesian Optimization, the hyperparameter values are examined in sequence. We need to know the performance of the first sets of hyperparameter to inform the search for the subsequent values. Thus, while it reduces the computational cost, the time to carry out the search may be much longer.

Grid Search and Random Search, on the other hand, can run in parallel. That is, different hyperparameter combinations can be evaluated at the same time. Thus, the search for hyperparameters is usually faster, but we waste some computational resources evaluating useless hyperparameter values.

## Python Tools For Hyperparameter Optimization

There are several open-source Python tools to carry out hyperparameter tuning in machine learning. They implement different search algorithms as we discuss in the next paragraphs.

### Scikit-learn

Scikit-learn is the go-to Python library for machine learning, hosting a wide array of predictive algorithms. [Scikit-learn](https://scikit-learn.org/stable/modules/grid_search.html) is also the go-to library to optimize the hyperparameters of its own algorithms.

Scikit-learn supports Grid Search and Random Search with cross-validation through the GridSearchCV and RandomizedSearchCV. Scikit-learn also offers support for multi-fidelity hyperparameter optimization through successive halving, which we have not covered in this tutorial.

The entire Scikit-learn package provides a range of machine learning-related modules that can be applied with minimum effort. It comes with built-in with tasks that allow you to carry out any activity —from preprocessing to using the sklearn.model_selection module to select the best hyperparameters.

Scikit-learn, and other similar packages for hyperparameter tuning, such as scikit-optimize and scikit-hyperband, have a consistent interface across many implemented classes, providing a massive benefit to data scientists. Built on top of NumPy and SciPy it also provides tools for data analysis and data mining, making it one of the engineers’ favorite packages.

### Scikit-optimize

[Scikit-optimize](https://scikit-optimize.github.io/stable/) is built on top of Scikit-learn, and it extends its functionality by supporting Sequential model-based optimization in Python, that is, Bayesian Optimization. Through the class BayesianSearchCV, we can set up Bayesian optimization using Gaussian processes or Random forests to model the objective function, just as we would do with the GridSearchCV and RandomizedSearchCV from Scikit-learn.

Scikit-optimize also provides functions out-of-the-box to produce visualizations of the hyperparameter search, and better understand how the different hyperparameter values affect the model performance.

### Optuna

[Optuna](https://optuna.readthedocs.io/en/stable/index.html) is a Python library that’s best suited for automated hyperparameter optimization. It supports different hyperparameter search algorithms, such as Grid Search, Random Search, and Bayesian optimization using Tree-Structured Parzen Estimators to compute optimal hyperparameters for a machine learning model.

The beauty of Optuna is that it has a define-by-run way of setting up the hyperparameter search function. This means, pretty much, that you can optimize with ease almost any hyperparameter from your desired machine learning model. You are not constrained by the design of the model or the API you are using to develop it. Hence, Optuna is, in my opinion, the most versatile hyperparameter search library, and I expect it to be the most widely used.

With Optuna, it’s possible to create your own hyperparameter tuning method class. Optuna also integrates with other popular packages, such as Scikit-learn, to give you a smooth user experience. It provides easy scalability with little or no changes to the code and is easy to install since it has few requirements. What more could you ask for?

### Hyperopt

[Hyperopt](http://hyperopt.github.io/hyperopt/) is another Python library that uses Bayesian optimization with Tree-Structured Parzen Estimators (TPE) as well as other learning algorithms like Random Search and Simulated Annealing (SA).

The highlight of Hyperopt is that it allows you to create very complex parameter spaces as well as easily configure your search space. It was one of the first available libraries for hyperparameter optimization, hence the most popular back then and remaining very popular today as well.

Beyond that, Hyperopt allows you to pause the optimization process, save important information, and resume later. It also gives engineers the capability to distribute their computation over a cluster of machines, easing the workflow.

What’s more? It works with various support frameworks, including XGBoost, Pytorch, Tensorflow, and Keras.

On the downside, the documentation for Hyperopt is a bit slim.

### Keras Tuner

Built with the Keras API, [Keras Tuner](https://keras.io/keras_tuner/) is a scalable hyperparameter tuning framework that is simple and hassle-free to use. It allows you to easily configure your search space with a define-by-run syntax and uses search algorithms to find the best hyperparameter values for your models.

It comes with built-in Bayesian Optimization, Random Search, and Hyperband algorithms and is also designed to allow engineers to experiment with newer search algorithms. It can be used to get the best parameters for deep learning models and can promise some of the highest accuracies that can be achieved with the defined combinations.

### Ray-tune

[Ray-Tune](https://docs.ray.io/en/latest/) is another great Python library for hyperparameter tuning at any scale. It supports most machine learning frameworks (Scikit-learn, Keras, TensorFlow, Random Forest among others). It also runs some of the top learning algorithms such as Population Based Training, and Hyperband. It also integrates with a wide range of additional hyperparameter optimization tools.

When using Ray-Tune, you can also scale a hyperparameter search from a single machine to a large distributed cluster without having to change your code. To top it off, it’s also very fast.

## Conclusion

[Data preprocessing](https://www.blog.trainindata.com/mastering-data-preprocessing-techniques/), [feature selection](https://www.blog.trainindata.com/feature-selection-machine-learning-with-python/), model training and hyperparameter tuning are all common steps in any data science project.

I hope this article helped you better understand what hyperparameter tuning in machine learning is and why it matters. Hopefully, you also understand what the main hyperparameter search algorithms are about, as well as some of the python tools that can help you out along the way.

Learning to optimize the hyperparameters of your models takes time, effort, and several practical implementations to master properly.

If you’re looking for a more detailed tutorial on how to use these tools, check out our online course [Hyperparameter Optimization for Machine Learning](https://www.trainindata.com/p/master-hyperparameter-optimization-for-tabular-learning).
