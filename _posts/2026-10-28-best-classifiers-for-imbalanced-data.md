---
layout: post
title: "What Are the Best Classifiers for Imbalanced Data?"
author: sole
description: "Which classifiers work best for imbalanced data? I compared gradient boosting, random forests, logistic regression, KNN and decision trees across 35 imbalanced datasets."
excerpt: "Which classifiers work best for imbalanced data? I compared gradient boosting, random forests, logistic regression, KNN and decision trees across 35 imbalanced datasets."
categories: [Imbalanced Data, Machine Learning]
image: assets/images/posts/best-classifiers-for-imbalanced-data/best-classifiers-for-imbalanced-data-cover.png
---

When we work with imbalanced data, we usually ask how to fix the imbalance: should we undersample, oversample, or use SMOTE? A more useful question is which classifier to use in the first place.

In this article, I'll answer that question with the results of an experiment where I compared 9 classification algorithms across 35 imbalanced datasets.

**The short answer, up front:** the best classifiers for imbalanced data are gradient boosting machines, like XGBoost, LightGBM and CatBoost, and random forests. Logistic regression is surprisingly competitive, while k-nearest neighbors and single decision trees perform worst.

> In chapter 1 of my book [Imbalanced Data: Myths, Mistakes and Modern Solutions](https://www.trainindata.com/p/imbalanced-data-myths-mistakes-solutions-book), I discuss the factors that make imbalanced classification hard, and why the choice of model matters more than the class imbalance itself.
>
> [![Imbalanced Data: Myths, Mistakes and Modern Solutions - book by Soledad Galli]({{ site.baseurl }}/assets/images/imbalanced-data-book-cover.jpg){: width="240"}](https://www.trainindata.com/p/imbalanced-data-myths-mistakes-solutions-book)

## Why the Classifier Matters More Than the Imbalance

[Class imbalance](https://www.blog.trainindata.com/class-imbalance-in-machine-learning/) on its own is rarely the problem. What makes imbalanced classification hard is a small number of minority class examples, poor class separability, and a model that can't capture the pattern of the minority class.

If the minority class follows a complex pattern, for example, a non-linear relationship with the features, or a small region of the feature space, simpler models like logistic regression or a decision tree may not be able to separate the classes. More flexible models, like random forests and gradient boosting machines, often can.

That's why choosing the right classifier is the first thing to get right. Resampling methods like [undersampling](https://www.blog.trainindata.com/undersampling-techniques-for-imbalanced-data/) and [oversampling](https://www.blog.trainindata.com/oversampling-techniques-for-imbalanced-data/) were developed for weak classifiers, and they don't make strong classifiers better.

## Comparing Classifiers on Imbalanced Datasets

To find the best classifiers for imbalanced data, I trained 9 models on 35 imbalanced datasets, with sizes ranging from fewer than 200 to about 285,000 observations, and imbalance ratios from about 2:1 to almost 600:1:

- Random forests (rf)
- AdaBoost (ada)
- Scikit-learn's gradient boosting machine (gbm)
- CatBoost (cat)
- LightGBM (lgbm)
- XGBoost (xgb)
- Logistic regression (logit)
- K-nearest neighbors (knn)
- Decision tree (tree)

I tuned the hyperparameters of each model with successive halving, optimizing the ROC-AUC. Then, I evaluated each model on bootstrapped samples of a held-out test set, to estimate the variability of the performance, and compared the models with Friedman's test followed by Nemenyi post-hoc tests.

The code and results are available in my [resampling experiments repository](https://github.com/solegalli/resampling-experiments).

### Results Across 4 Imbalanced Datasets

The following figure shows the [ROC-AUC](https://www.blog.trainindata.com/auc-roc-analysis/) of the 9 classifiers in 4 of the 35 datasets. Dots show the mean ROC-AUC and whiskers one standard deviation. The pink band is the performance of the random forest, and the horizontal lines with asterisks mark models that are significantly worse than the best one.

![ROC-AUC of random forests, AdaBoost, gradient boosting, CatBoost, LightGBM, XGBoost, logistic regression, KNN and decision trees on four imbalanced datasets.]({{ site.baseurl }}/assets/images/posts/best-classifiers-for-imbalanced-data/classifiers-roc-auc-imbalanced-datasets.png)

The protein homology dataset has 111 majority observations for every minority one. Despite this extreme imbalance, all ensembles and logistic regression reach a ROC-AUC close to 0.99, and only KNN and the decision tree perform significantly worse.

In the poker dataset, with an imbalance ratio of 82:1, CatBoost and LightGBM reach a ROC-AUC of about 0.95, while logistic regression and KNN stay near 0.5, the level of random guessing. The difference comes from the complexity of the decision boundary rather than the class imbalance: the minority class follows a non-linear pattern that only gradient boosting captures.

In the us_crime dataset, logistic regression has the highest mean ROC-AUC. In the adult dataset, with a mild imbalance of about 3:1, XGBoost has the highest ROC-AUC, while logistic regression, KNN and the decision tree perform significantly worse.

## The Best Classifiers for Imbalanced Data

### Gradient Boosting Machines and Random Forests

Across the 35 datasets, the ensemble models separated the classes well in most cases. Gradient boosting machines tended to sit at or near the top in each dataset, but which implementation won varied from one dataset to another.

In the larger comparison of the ensembles across 37 datasets, XGBoost had the highest mean ROC-AUC in 10 datasets, random forests in 8, scikit-learn's gradient boosting and LightGBM in 6 each, AdaBoost in 4 and CatBoost in 3. Averaged across datasets, their ROC-AUC ranged from 0.895 for AdaBoost to 0.914 for random forests.

So there is no single best gradient boosting machine for imbalanced data. Try a few, and pick the one that works best on your data.

### Logistic Regression

Logistic regression was surprisingly competitive. It was significantly worse than the best model in only 12 of the 35 datasets, matched the ensembles in the remaining ones, and even had the highest mean ROC-AUC in 3 of them.

Logistic regression fails when the minority class follows a non-linear pattern, as in the poker dataset. But when the relationship between the features and the target is close to linear, it is a fast, simple and interpretable choice.

### K-Nearest Neighbors and Decision Trees

KNN was significantly worse than the best model in 22 datasets, although it had the highest mean ROC-AUC in 3 and matched the ensembles in 10 others.

The single decision tree performed worst. It was significantly worse than the best model in 27 of the 35 datasets, and never achieved the highest mean ROC-AUC.

## Choosing a Classifier: Speed and Calibration

### Training Time

When several models perform similarly, training time can be the tie breaker. In my experiments, XGBoost was the fastest ensemble, with a median training time of about 3 seconds per dataset, while CatBoost was the slowest, at about 37 seconds.

Random forests, scikit-learn's gradient boosting, AdaBoost and LightGBM fell in between.

### Probability Calibration

If you need accurate probabilities, for example, to estimate risk or to make cost-sensitive decisions, look at the Brier score too. Random forests and the gradient boosting machines returned well [calibrated probabilities](https://www.blog.trainindata.com/probability-calibration-in-machine-learning/), with an average Brier score of about 0.04.

AdaBoost returned poorly calibrated probabilities, with an average Brier score of 0.15, because it pushes the predicted probabilities away from 0 and 1.

## What About Classifiers Designed for Imbalanced Data?

Imbalanced-learn offers ensembles designed for imbalanced data, like balanced random forests, EasyEnsemble and RUSBoost, which undersample the data while building each tree.

In my experiments, none of them consistently beat random forests or gradient boosting machines trained on the original data, and they returned poorly calibrated probabilities. I discuss these results in [Should You Use Imbalanced-Learn in 2026?](https://www.blog.trainindata.com/should-you-use-imbalanced-learn-in-2025/).

## How to Get the Most Out of Your Classifier

Choosing a strong classifier is the first step. To get the most out of it:

1. Train it on the original data, without resampling.
2. Evaluate it with a threshold independent metric, like the ROC-AUC or the [PR-AUC](https://www.blog.trainindata.com/precision-recall-curves/).
3. Tune the classification threshold to make the decisions your application needs, instead of using the default 0.5.

I walk you through this approach in my guide to [machine learning with imbalanced data](https://www.blog.trainindata.com/machine-learning-with-imbalanced-data/).

## Conclusion

The best classifiers for imbalanced data are gradient boosting machines, like XGBoost, LightGBM and CatBoost, and random forests. They separated the classes well across most of the 35 datasets, without any resampling.

Logistic regression is a good, fast baseline when the relationship between the features and the target is close to linear. Avoid single decision trees and KNN, which performed worst.

If you want to learn more about working with imbalanced data, check out my book [Imbalanced Data: Myths, Mistakes and Modern Solutions](https://www.trainindata.com/p/imbalanced-data-myths-mistakes-solutions-book), and explore the results for all the datasets in the [resampling experiments repository](https://github.com/solegalli/resampling-experiments).

[![Imbalanced Data: Myths, Mistakes and Modern Solutions - book by Soledad Galli]({{ site.baseurl }}/assets/images/imbalanced-data-book-cover.jpg)](https://www.trainindata.com/p/imbalanced-data-myths-mistakes-solutions-book)

## References

- Chen, T. and Guestrin, C. 2016. [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754). arXiv.
- Galli, S. 2026. [Resampling experiments code repository](https://github.com/solegalli/resampling-experiments).
- Galli, S. 2026. [Imbalanced Data: Myths, Mistakes and Modern Solutions](https://www.trainindata.com/p/imbalanced-data-myths-mistakes-solutions-book).
- [Scikit-learn documentation: Successive halving](https://scikit-learn.org/stable/modules/grid_search.html#successive-halving-user-guide).
