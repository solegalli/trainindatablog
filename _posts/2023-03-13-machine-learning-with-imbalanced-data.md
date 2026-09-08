---
layout: post
title: "Dealing with Imbalanced Datasets in Machine Learning: Techniques and Best Practices"
author: sole
description: "Discover the techniques used to handle imbalanced datasets in machine learning, what they actually do, and how they compare to adjusting the classification threshold."
excerpt: "Discover the techniques used to handle imbalanced datasets in machine learning, what they actually do, and how they compare to adjusting the classification threshold."
categories: [Imbalanced Data, Machine Learning]
image: assets/images/posts/machine-learning-with-imbalanced-data/cover-7.gif
---

Imbalanced datasets are a familiar challenge data scientists and machine learning practitioners face. When the distribution of classes in a dataset is skewed, with one or more classes having significantly fewer samples than others, it can lead to trained models that make biased predictions and show poor overall performance.

But fear not, as various techniques and best practices can be employed to help us make cost-sensitive decisions when working with these datasets.

This article will explore the following:

- The nature of imbalanced data
- The challenges of imbalanced datasets
- How to make cost-sensitive decisions with imbalanced data
- The use of undersampling
- The use of oversampling
- Cost-sensitive learning, and
- Ensemble models for imbalanced datasets

We will also look at imbalanced-learn, an open-source Python package to tackle imbalanced datasets. So, if you are ready to tackle imbalanced data head-on and unlock the full potential of your machine-learning models, keep reading!

For a quick summary of current practices as of 2024, check out this video:

<div class="video-embed"><iframe src="https://www.youtube.com/embed/blcOOheXNoQ?feature=oembed" title="Working with Imbalanced Data in 2024 - Machine Learning with Imbalanced Data" width="560" height="315" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe></div>

For practical Python examples of how to work with imbalanced data, check out our book [Machine Learning with Imbalanced Data](https://www.trainindata.com/p/imbalanced-data-myths-mistakes-solutions-book).

[![Imbalanced Data: Myths, Mistakes and Modern Solutions - book by Soledad Galli]({{ site.baseurl }}/assets/images/imbalanced-data-book-cover.jpg)](https://www.trainindata.com/p/imbalanced-data-myths-mistakes-solutions-book)

## What Undersampling, Oversampling, SMOTE and Cost-Sensitive Learning Actually Do

Before we go through these techniques one by one, it's worth being precise about what they do. Undersampling, oversampling, SMOTE, and cost-sensitive learning do not make a model better at discriminating between classes. What they actually do is shift the model's decision boundary so that, at the default classification threshold of 0.5, we make more cost-sensitive decisions — that is, we correctly flag a larger proportion of the minority class, which is usually the class we care about the most.

Here's the important part: you can get this same effect by training the model on the original, unmodified data and simply adjusting the classification threshold afterward, without resampling anything or reweighting the loss function at all. We'll call this out again for each technique below.

## What is Imbalanced Data?

In imbalanced datasets, one class is significantly more represented than the other(s). In other words, imbalanced datasets have disproportionate numbers of observations in each category of the target variable, with one or more classes being extremely under-represented. This could make it difficult for machine-learning algorithms to learn how to discriminate between them.

![Definition and illustration of imbalanced datasets]({{ site.baseurl }}/assets/images/posts/machine-learning-with-imbalanced-data/imbalanced-data-1024x576.png)

Imbalanced datasets are common in the real world and often lead to biased predictions and poor overall performance of the machine learning model.

The degree of imbalance can vary significantly and may be caused by factors like natural unequal distribution or sampling bias in data collection. Understanding the characteristics and differences between binary and multiclass imbalanced data and minority and majority classes will help us address them better.

### Binary vs. Multiclass Imbalanced Data

Binary imbalanced data refers to datasets with only two classes: one class has a majority of samples and the other a minority.

In binary classification tasks, imbalanced datasets occur when one class has significantly fewer samples than another (e.g., 80% of samples belong to Class A while only 20% belong to Class B). This phenomenon is also known as “class imbalance.”

For example, in a credit card fraud detection dataset, the majority class would be non-fraudulent transactions, while the minority class would be fraudulent transactions.

On the other hand, multiclass imbalanced data refers to datasets with three or more classes, where one or more classes have significantly fewer samples than others.

A real-life example of a multiclass imbalanced dataset is a medical diagnosis dataset. There will be multiple diseases or different classes; one may have a lower occurrence rate than others.

### Minority Class vs. Majority Class

In both binary and multiclass imbalanced data, we’ll find:

- Minority classes: These are the classes with fewer samples than others. Minority classes are often the ones we are interested in predicting, as they are usually associated with rare events or anomalies.
- Majority classes: These are the most frequent classes in a dataset, having more samples than other classes.

### Importance of Defining Minority Class vs. Majority Class

In many real-world applications, the minority class is often the class of interest, and the goal is to identify and classify it correctly. In our previous fraud detection example, the minority class is the fraudulent transactions that must be addressed, while the majority class is the non-fraudulent transactions.

In the medical diagnosis example, the minority class is the rare disease that needs to be detected, while the majority class are the more common diseases.

The problem with imbalanced datasets is that the machine learning algorithms tend to be biased toward the majority class, as it has more data points from which to learn. But also, machine learning models optimize for balanced metrics by default. This can lead to poor model performance and misclassification of the minority class.

## Why is Imbalanced Data Challenging for Machine Learning

Imbalanced data is challenging for machine learning for several reasons.

### Biased Models

Machine learning models optimize balanced metrics or functions by default. Hence, imbalanced data can lead to biased trained models that tend to predict the majority class more accurately while neglecting the minority class. This can result in a high false negative rate, where the minority class is misclassified as the majority class.

### Skewed Class Distribution

Imbalanced data show a skewed class distribution, where the majority class dominates the dataset. This can lead to a lack of diversity in the data and make it difficult for the model to learn the underlying patterns of the minority class.

### Differing Cost of Misclassification Errors

Another challenge with imbalanced data is that misclassification errors can have varying costs depending on the application. For example, a false negative can be life-threatening in medical diagnoses, while a false positive may only cause inconvenience.

Considering the cost of misclassification errors when designing the model and selecting appropriate techniques to address the imbalance is essential.

## How to Handle Classification Problems in Imbalanced Data

Dealing with imbalanced data ensures that machine learning models can effectively learn and generalize to new, unseen data. Here are some of the most effective techniques to handle imbalanced data and make cost-sensitive decisions.

### Choosing Correct Evaluation Metrics

Choosing the right performance metrics is essential when working with imbalanced data. Popular evaluation metrics, like the accuracy, are misleading since they don’t consider the class imbalance and, hence, fail to measure performance accurately. In fact, it could well be that the model returns no true positives for the minority class, and yet the overall accuracy of the model is high because it classifies correctly most of the samples from the majority class, which are the overwhelming majority of data points.

Instead, other performance measures like precision, recall, F1 score, and Matthews correlation coefficient (MCC) should be used to evaluate model performance when dealing with imbalanced datasets.

In addition to using these metrics, confusion matrices can also help evaluate model performance. Confusion matrices provide a breakdown of predicted versus actual class labels, allowing for a more granular understanding of how the model performs on each class.

### Training Set Resampling

Resampling is a popular technique used to handle imbalanced data. It involves changing the distribution of the training set by either [oversampling](https://www.blog.trainindata.com/oversampling-techniques-for-imbalanced-data/) or [undersampling](https://www.blog.trainindata.com/undersampling-techniques-for-imbalanced-data/). We will say more about this towards the end of the article.

### Using Cross-Validation

Cross-validation lets you:

- Train the model on different subsets of data
- Test it with a held-out subset, and
- Validate results by repeating the process multiple times.

This approach helps to reduce bias when dealing with imbalanced datasets as it ensures that each set is evaluated using an unbiased metric.

### Adding more Data

One practical approach to overcome the model generalization issue is to add more or new data to your original dataset. However, if the class imbalance persists and the class separation is not clear, using conventional classifiers like logistic regression or random forest often results in misclassifying the rare class while generalizing.

### Adjusting the Threshold

Adjusting the classification threshold to achieve the optimal separation of two classes is essential when working with imbalanced datasets, where the default threshold of 0.5 may not be effective.

You can use ROC curves and Precision-Recall curves to determine the optimal threshold for the classifier. Also, the area under the curve (AUC) can be used as a metric to evaluate the performance of different thresholds. Find examples of how to calculate the roc_auc in our [GitHub code repository for imbalanced data](https://github.com/solegalli/machine-learning-imbalanced-data).

### Designing a Custom Classifier

Creating a custom classification model tailored explicitly to handle imbalanced classification can be very effective. This involves directly incorporating measures that address class imbalance into the model’s architecture, parameters, and hyperparameters. [Hyperparameter tuning and optimization](https://www.blog.trainindata.com/hyperparameter-tuning-for-machine-learning/) can help you significantly.

For example, support vector machines (SVMs) include a cost parameter (class_weight) that helps adjust the relative importance of different classes to handle class imbalance.

### What is the Most Effective Technique to Handle Classification Problems?

Unfortunately, there is no silver bullet. A good approach is experimenting with different techniques and evaluating their performance and tradeoffs using appropriate metrics such as precision, recall, and F1 score. By selecting the proper method, you can ensure that your model is not biased towards the majority class and can effectively generalize to new, unseen data.

We can make more cost-sensitive decisions at the default threshold by using sampling methods. Undersampling and oversampling balance the class distribution and, in doing so, shift the decision boundary toward the minority class. We could also implement [cost-sensitive learning](https://www.blog.trainindata.com/cost-sensitive-learning-for-imbalanced-data/), where we penalize harder the misclassification of the minority class — mathematically, this also shifts the effective decision boundary during training. And finally, we can also use bespoke ensemble methods, specifically designed to work with imbalanced datasets.

The following sections will overview undersampling, oversampling, cost-sensitive learning, and ensemble methods for imbalanced datasets.

## Overview of Undersampling

Undersampling, or downsampling, is a common technique to address the class imbalance in machine learning. This involves reducing the number of samples in the majority class to balance the class distribution.

![Diagram showing the process of random undersampling the majority class in an imbalanced dataset.]({{ site.baseurl }}/assets/images/posts/machine-learning-with-imbalanced-data/undersampling-1024x576.png)

Undersampling methods are divided into undersampling and cleaning methodologies. In undersampling, we remove samples until a certain class balance ratio is reached. In cleaning, we remove observations that are closer to the decision boundary.

You can learn about the different undersampling methods in more detail in our book “[Machine Learning with Imbalanced Data”](https://www.trainindata.com/p/imbalanced-data-myths-mistakes-solutions-book).” Here we highlight three of the most common ones.

### Common Undersampling Techniques

- Random undersampling: This method randomly removes samples from the majority class(es) until achieving a balanced class distribution.
- Tomek links: This method identifies samples closest to the decision boundary and removes them to improve class separation.
- Edited nearest neighbors: In this undersampling procedure, samples that do not agree enough with their neighborhood are removed from the dataset. In plain English, it looks at every observation from the majority class, and if the majority of its neighbors are not also from the majority class, then the observation is removed.

Let’s assess the pros and cons of undersampling methodologies:

### Pros

- Corrects imbalanced data to reduce the risk of skewing toward the majority class.
- Makes the minority class equal to the majority class for data analysis.
- Requires less storage and time for analysis, which can save businesses resources.
- Can improve run times for analyses and model training.

### Cons

- Loss of potentially essential data due to removing data from the majority class.
- Possibility of biased results if the sample of the majority class chosen is not representative of the real world.
- Thoughtful and informative undersampling techniques are needed to combat the loss of potentially important data.
- Possibility of requiring a combination of undersampling and oversampling to obtain the most accurate results.

## Overview of Oversampling

Oversampling is another commonly used technique to address imbalanced classes in machine learning. Oversampling involves increasing the number of samples in the minority class to balance the class distribution.

![Diagram showing the process of random oversampling a dataset with 3 classes where one is minority,]({{ site.baseurl }}/assets/images/posts/oversampling-techniques-for-imbalanced-data/oversampling-1024x576.png)

The most common method is random oversampling, which, in essence, just duplicates data points. To avoid data duplication, a group of data scientists developed SMOTE. Let’s find out what they are about.

### Common Oversampling Techniques

- Random oversampling: This method randomly duplicates samples from the minority class(es) until a balanced class distribution is achieved.
- SMOTE (Synthetic Minority Over-sampling Technique): It creates synthetic minority class samples by interpolating between existing minority class samples of the training dataset and their k-nearest neighbors. Like this, it avoids data duplication.

SMOTE is suitable only for continuous variables. There are therefore extensions of SMOTE that tackle datasets with both numerical and categorical variables, or just categorical variables. There are also other oversampling techniques like ADASYN, where samples in the feature space that are harder to learn are given more weight. You can learn more about alternative oversampling methods in our book “[Machine Learning with Imbalanced Data](https://www.trainindata.com/p/imbalanced-data-myths-mistakes-solutions-book).”

Let’s assess the pros and cons of oversampling methodologies:

### Pros

- Oversampling doesn’t lead to the loss of potentially critical data, as no samples are removed from the dataset.
- Oversampling shifts the model's decision boundary toward the minority class, so at the default 0.5 threshold, more minority class examples get correctly flagged. You can achieve the same effect by training on the original data and adjusting the classification threshold instead.

### Cons

- May cause overfitting due to the duplication of minority class samples.
- May generate synthetic data samples that are unrealistic or not representative of real-world distributions.

## Undersampling Or Oversampling: Which One to Choose?

When handling imbalanced datasets, choosing between undersampling and oversampling can be challenging. Oversampling creates synthetic examples or duplicates the minority class, while undersampling eliminates examples from the majority class.

Undersampling is a suitable option when dealing with large datasets that are challenging to analyze. By utilizing all rare class instances and randomly removing the majority class instances, the dataset can be transformed into a balanced one, with equal representation of both classes. This shifts the decision boundary toward the minority class.

Oversampling, on the other hand, can be useful when we have a limited amount of data. However, it’s crucial to be cautious when using oversampling, especially when the class imbalance is significant, as we will be introducing a lot of artificial data points or duplications.

So, as always, there is no silver bullet. Our choice depends on the datasets that we have, the computing resources that we have available, and a bit of trial and error.

## Overview of Cost-Sensitive Learning

[Cost-sensitive learning](https://www.blog.trainindata.com/cost-sensitive-learning-for-imbalanced-data/) is a machine learning approach that considers the costs associated with incorrect predictions. In traditional machine learning, the goal is to optimize the overall accuracy score of a model, but in cost-sensitive learning, the objective is to minimize the total cost of misclassification.

In cost-sensitive learning, the misclassification costs are defined based on the specific application or domain. For example, in fraud detection, a false negative (labeling a fraudulent transaction as legitimate) can result in a significant financial loss for a bank or credit card company.

In contrast, a false positive (labeling a legitimate transaction as fraudulent) can lead to customer frustration and decreased trust in the institution. The costs of these errors are unequal and must be considered when designing a cost-sensitive learning algorithm.

### Common Cost-Sensitive Learning Techniques

Cost-sensitive classification is a type of machine learning approach that takes into account the asymmetric costs of different types of classification errors. In traditional classification problems, the focus is on optimizing the overall accuracy, but in some scenarios, certain types of errors are more costly than others.

For example, in medical diagnosis, misclassifying a patient with a serious disease as healthy can be much more costly than misclassifying a healthy patient as having the disease. Similarly, in credit card fraud detection, falsely identifying a legitimate transaction as fraudulent can lead to customer dissatisfaction, while failing to detect a fraudulent transaction can lead to financial losses.

Cost-sensitive classification methods assign different misclassification costs to different classes and try to optimize a cost-sensitive metric such as the cost-weighted accuracy. The objective is to minimize the expected cost instead of the overall accuracy.

One common implementation of cost-sensitive algorithms is to use the class_weight parameter in the (Scikit-learn) model, which assigns a weight to each class in the training data to adjust the importance of each class during training.

### Pros

- Cost-sensitive learning can help businesses make better decisions by considering the costs associated with different types of errors.
- It allows us to avoid biasing the distribution of data through over or undersampling techniques. Instead, we can use traditional off-the-shelf algorithms with just a minor modification: adjusting the cost through techniques such as using the “class_weight” parameter.

### Cons

Cost-sensitive learning requires additional effort to identify and quantify the misclassification costs, which can be challenging in some applications.

## Ensemble Models for Imbalanced Data

Ensemble models are machine learning algorithms that combine the predictions of multiple models to improve their overall performance and accuracy. In an ensemble model, multiple models are trained on the same dataset, and their predictions are aggregated to produce a final prediction.

The classical ensemble models are random forests and gradient-boosting machines, which combine the output of several decision trees to make the final predictions. However, these models share the characteristics of all other traditional machine learning algorithms in that they optimize for a balanced metric, the overall accuracy.

Some scientists have, however, designed specific ensemble models that are geared to better classify imbalanced datasets, and therefore tackle the class problem out-of-the-box. Let’s explore some of them.

### Examples of Ensemble Models for Imbalanced Data

- Balanced Bagging (Bootstrap Aggregating): Several models are trained on different subsets of the training data, which are sampled randomly with replacement. The resampling rebalances the class distribution. These models are then combined through a weighted average to make a final prediction.
- RUSBoost: It combines data (re)sampling and boosting to improve classification performance for imbalanced training data.
- Balancing Cascade: A cascade of classifiers is trained, each of which learns to distinguish between a balanced subset of the majority class and the minority class. Misclassified majority class samples are discarded in each step to prevent them from dominating the classifier.

There are additional variations of bagging and boosting with resampling. To learn more about them, check out our book “[Machine Learning with Imbalanced Data](https://www.trainindata.com/p/imbalanced-data-myths-mistakes-solutions-book).”

### Pros

The advantage of ensemble methods for imbalanced datasets is that they handle imbalanced data effectively out-of-the-box without the need for any additional modifications or techniques.

### Cons

One drawback of these methods is that they are less well-known; there are no open-source implementations for all of them, which means that we need to code the algorithm ourselves, and some are computationally costly.

## Imbalanced-learn: The Open-Source Python Package for Working with Imbalanced Data

[Imbalanced-learn](https://imbalanced-learn.org/stable/), or Imblearn, is an open-source Python library designed to help address class imbalance problems in machine learning. The library provides tools and algorithms for resampling imbalanced datasets and shifting the decision boundary toward the minority class.

Imbalanced-learn offers a variety of oversampling and undersampling techniques, including:

- Random oversampling
- SMOTE (Synthetic Minority Over-sampling Technique) and SMOTE variations.
- ADASYN (Adaptive Synthetic Sampling)
- Tomek links
- Cluster centroids
- Random undersampling
- NearMiss
- Condensed Nearest Neighbor
- Edited Nearest Neighbor
- One Sided Selection

And much more.

The library also includes ensemble methods, such as Easy Ensemble, Balanced Bagging and RUSBoost, which can improve the classification accuracy of imbalanced data.

One of the critical benefits of Imbalanced-learn is its compatibility with popular machine learning libraries such as scikit-learn (sklearn), Keras, and PyTorch. This allows users to easily incorporate the imbalanced data handling techniques into their existing machine learning pipelines.

Imbalanced-learn shares sklearn functionality with methods fit() and resample() to learn the parameters from the data and then resample the datasets.

## Wrap-Up

Dealing with imbalanced data is a crucial aspect of machine learning and data science projects, and it requires effective techniques and tools to ensure accurate predictions.

Undersampling, oversampling, cost-sensitive learning, and using specific ensemble algorithms are all valuable approaches for addressing class imbalance. The Imbalanced-learn Python package provides a user-friendly and out-of-the-box solution to working with imbalanced datasets.

For tutorials about the different methods to work with imbalanced datasets, how to implement them in Python, and understanding how to better evaluate model performance, check our book [Machine Learning with Imbalanced Data](https://www.trainindata.com/p/imbalanced-data-myths-mistakes-solutions-book).
