---
layout: post
title: "Cost-Sensitive Learning: Beyond the Accuracy in Imbalanced Classification"
author: sole
description: "Find out what cost-sensitive learning is, why class weights are equivalent to shifting the decision threshold, and how to make cost-sensitive decisions in Python."
excerpt: "Cost-sensitive learning is threshold shifting. Find out when class weights make sense, when to adjust the threshold instead, and how to do both in Python."
categories: [Data Science, Imbalanced Data, Machine Learning]
image: assets/images/posts/cost-sensitive-learning-for-imbalanced-data/cover_csl.gif
math: true
---

Most classification algorithms are designed to minimize the error rate, the fraction of observations they classify incorrectly. They treat every mistake as equally costly, whether it is a missed fraud case or a false alarm.

In many real problems, that assumption is wrong. Missing a fraudulent transaction can cost thousands, while flagging a legitimate one costs a phone call to the customer. Cost-sensitive learning takes these different costs into account, and aims to minimize the total cost of the model's decisions instead of the error rate.

Cost-sensitive learning is commonly recommended to handle imbalanced data, by setting class weights proportional to the class imbalance. In this article, I'll show that class weights, sample weights and resampling all do the same thing: they shift the decision threshold of the classifier.

That has important practical consequences:

- Cost-sensitive learning does not make a model better at separating the classes.
- It makes sense when we know the real costs of each type of error.
- If the only problem is the class imbalance, we are better off training on the original data and adjusting the decision threshold.

In this article, we will cover:

- What cost-sensitive learning is and how a cost matrix works
- How costs determine the optimal decision threshold
- Why class weights and resampling are equivalent to shifting the threshold
- How to make cost-sensitive decisions in Python, evaluating every result with its standard deviation
- When to use class weights, and when to adjust the threshold instead

> This article is a short introduction to the topic. In chapter 4 of my book [Imbalanced Data: Myths, Mistakes and Modern Solutions](https://www.trainindata.com/p/imbalanced-data-myths-mistakes-solutions-book), I walk you through the mathematics that link costs, thresholds, class weights and resampling, step by step, and back them with experiments on dozens of datasets.
>
> [![Imbalanced Data: Myths, Mistakes and Modern Solutions - book by Soledad Galli]({{ site.baseurl }}/assets/images/imbalanced-data-book-cover.jpg){: width="240"}](https://www.trainindata.com/p/imbalanced-data-myths-mistakes-solutions-book)

## What Is Cost-Sensitive Learning?

Cost-sensitive learning is a group of methods that incorporate misclassification costs into the learning process, so that the model minimizes the total expected cost instead of the error rate. It rests on a simple premise: each observation should be assigned to the class with the lowest expected cost.

This means that the most probable class is not always the right decision. A bank may block a large card payment that is probably legitimate, and a doctor may order more tests for a patient who is probably healthy, because the cost of being wrong in the other direction is much higher.

## The Cost Matrix

In binary classification, every prediction has one of four outcomes: a true positive, a false positive, a true negative or a false negative. You can find these outcomes in the [confusion matrix](https://www.blog.trainindata.com/confusion-matrix-precision-and-recall/) of the model.

The first step in cost-sensitive learning is to assign a cost to each of these outcomes. We summarize them in a cost matrix, where $$C(i, j)$$ is the cost of predicting class $$i$$ when the true class is $$j$$:

| | Actual negative (0) | Actual positive (1) |
| --- | --- | --- |
| **Predicted negative (0)** | C(0, 0), true negative | C(0, 1), false negative |
| **Predicted positive (1)** | C(1, 0), false positive | C(1, 1), true positive |
{: .table .table-bordered .table-sm}

Correct predictions usually have a cost of zero. In fraud detection, $$C(0, 1)$$ would be the money lost by missing a fraudulent transaction, and $$C(1, 0)$$ the cost of investigating a legitimate one.

## From Costs to the Decision Threshold

Once we know the costs, how do we decide which class to predict? We compute the expected cost of each decision and pick the cheapest one. For an observation $$x$$, the expected cost of predicting class $$i$$ is:

$$
R(x, i) = \sum_j P(j \mid x) \, C(i, j)
$$

where $$P(j \mid x)$$ is the probability that the observation belongs to class $$j$$. In other words, we weight the cost of each possible outcome by its probability.

### The Cost-Sensitive Threshold

With a few lines of algebra, comparing the expected cost of predicting each class leads to a simple rule. When correct predictions cost nothing, we should predict the positive class whenever:

$$
P(1 \mid x) \geq p^* = \frac{C(1, 0)}{C(1, 0) + C(0, 1)}
$$

The costs determine the decision threshold $$p^*$$, and nothing else does. If we know the costs, and the model returns calibrated probabilities, we can make cost-sensitive decisions without changing the model at all.

Notice what happens when both errors cost the same. With $$C(1, 0) = C(0, 1) = 1$$, the threshold is $$1 / (1 + 1) = 0.5$$, the default threshold of most classifiers. A standard classifier is simply a cost-sensitive classifier that assumes all errors are equally costly.

### A Numerical Example

Suppose missing a fraudulent transaction costs 10 times more than investigating a legitimate one. With $$C(1, 0) = 1$$ and $$C(0, 1) = 10$$, the threshold becomes:

$$
p^* = \frac{1}{1 + 10} \approx 0.09
$$

We should flag a transaction as fraud when the model is only 9% sure, because missing a fraud is so expensive. Raising the cost of false negatives lowers the threshold, and the model predicts the positive class more readily.

> In chapter 4 of my [book](https://www.trainindata.com/p/imbalanced-data-myths-mistakes-solutions-book), I derive this threshold step by step from the expected cost, for any cost matrix, including the case where correct predictions also have a cost.

## Class Weights Are Equivalent to Shifting the Threshold

Adjusting the threshold is the simplest way to make a classifier cost-sensitive. In practice, though, most people reach for class weights or resampling instead.

### How Class Weights Enter the Loss Function

Class weights change the loss function that the algorithm minimizes. Logistic regression, for example, normally minimizes the log loss:

$$
L = -\sum_i \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]
$$

where $$\hat{p}_i$$ is the predicted probability of the positive class. With class weights, each term is multiplied by the weight of its class, so errors on the positive class count $$w_1 / w_0$$ times more:

$$
L_w = -\sum_i \left[ w_1 \, y_i \log(\hat{p}_i) + w_0 \, (1 - y_i) \log(1 - \hat{p}_i) \right]
$$

Decision trees and the ensembles built on them, like random forests and gradient boosting machines, use the weights in a similar way. They weight each observation when calculating the impurity of a split, such as the Gini index or the entropy, or the gradients of the loss.

### Weights, Resampling and Thresholds Lead to the Same Decisions

In 2001, Charles Elkan showed that changing the class balance of the training data produces a classifier that, at the default threshold of 0.5, makes the same decisions as a classifier trained on the original data and used at the threshold $$p^*$$. He also showed exactly how much we need to resample the data to reach a given threshold.

A couple of years later, Zadrozny and colleagues proved that weighting each observation by its misclassification cost turns an ordinary error-minimizing algorithm into one that minimizes the expected cost. That is the formal justification for class weights and sample weights.

Put together, these results tell us that class weights, sample weights, undersampling and oversampling are different ways of encoding the same cost matrix. They all move the decision boundary, so that the default threshold of 0.5 behaves like the threshold $$p^*$$.

> The full chain of equivalences, from the cost matrix to the threshold, to the resampling ratio, to the class weights, is the core of chapter 4 of my [book](https://www.trainindata.com/p/imbalanced-data-myths-mistakes-solutions-book). There, I go through Elkan's resampling formula and Zadrozny's translation theorem, and show what the empirical evidence says.

### What Class Weights Do Not Do

If class weights only shift the decision boundary, they should not make the model any better at separating the classes. Threshold-independent metrics like the ROC-AUC should stay the same.

They do, however, distort the predicted probabilities. A model trained with class weights behaves as if the minority class were more frequent than it really is, so it overestimates its probability, as we discuss in our article on [probability calibration](https://www.blog.trainindata.com/probability-calibration-in-machine-learning/).

We will see both effects in the Python demo below.

## Where Do the Costs Come From?

This is the hardest part of cost-sensitive learning. Real costs come from the consequences of the decisions: the money lost to a missed fraud, the price of an investigation, the clinical risk of a missed diagnosis, or the cost of a marketing call.

Finding them requires long conversations with domain experts and stakeholders. Some costs, like a financial loss, are easy to quantify, while others, like customer inconvenience or a patient's well-being, are much harder.

### Class Imbalance Is Not a Cost

A very common practice is to set the class weights to the inverse of the class frequencies, for example with `class_weight="balanced"` in scikit-learn. This assumes that the cost of each error is proportional to the imbalance ratio, which is rarely true.

Using the imbalance ratio as a cost is equivalent to lowering the threshold to the fraction of positives in the data. If that is all we want, we can adjust the threshold directly, without retraining the model or distorting its probabilities.

### Instance-Dependent Costs

Often, the cost of an error depends on the observation. Missing a fraudulent payment of $5,000 costs more than missing one of $50, and failing to contact a generous donor costs more than failing to contact a small one.

In these cases, there is no single threshold, and we can use sample weights, which assign a cost to each individual observation. We can also tune the threshold with a scoring function that uses the cost of each observation.

## Cost-Sensitive Learning in Python

Let's now put this into practice. We will check whether class weights improve a model, and then make cost-sensitive decisions with class weights and with threshold adjustment.

Every result will come with its standard deviation across cross-validation folds. A single train and test split often shows small differences between models that disappear once we account for the variability of the estimate.

### Loading the Imbalanced Dataset

We start by importing the libraries:

```
import numpy as np
import pandas as pd
from imblearn.datasets import fetch_datasets
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import (
    FixedThresholdClassifier,
    StratifiedKFold,
    TunedThresholdClassifierCV,
    cross_validate,
)
```

Next, we load the protein homology dataset from imbalanced-learn. The target takes the value 1 for the minority class and -1 for the majority class, so we convert it to 1 and 0:

```
data = fetch_datasets(filter_data=("protein_homo",))["protein_homo"]
X = data.data
y = (data.target == 1).astype(int)

print("Class counts:", np.bincount(y))
print(f"Fraction of positives: {y.mean():.4f}")
```

In the following output, we see that less than 1% of the observations belong to the positive class:

```
Class counts: [144455   1296]
Fraction of positives: 0.0089
```

We will evaluate all models with 5-fold stratified cross-validation, and use gradient boosting machines from scikit-learn. We write a small function to create them, with or without class weights:

```
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

def gbm(class_weight=None):
    return HistGradientBoostingClassifier(
        class_weight=class_weight, random_state=0,
    )
```

### Do Class Weights Improve the Model?

Let's train models without class weights, with balanced class weights, and with a weight of 10 for the positive class. We evaluate each one with the ROC-AUC, which measures how well the model separates the classes, and the Brier score, which measures how close the predicted probabilities are to the real outcomes, where lower is better:

```
results = {}
for name, class_weight in [
    ("No weights", None),
    ("Balanced", "balanced"),
    ("{0: 1, 1: 10}", {0: 1, 1: 10}),
]:
    cv_res = cross_validate(
        gbm(class_weight), X, y, cv=cv,
        scoring=["roc_auc", "neg_brier_score"],
    )
    roc = cv_res["test_roc_auc"]
    brier = -cv_res["test_neg_brier_score"]
    results[name] = {
        "ROC-AUC": f"{roc.mean():.3f} ± {roc.std():.3f}",
        "Brier score": f"{brier.mean():.4f} ± {brier.std():.4f}",
    }

print(pd.DataFrame(results).T)
```

In the following output, we see the mean and standard deviation of each metric across the 5 folds:

```
                     ROC-AUC      Brier score
No weights     0.993 ± 0.002  0.0023 ± 0.0003
Balanced       0.993 ± 0.002  0.0056 ± 0.0010
{0: 1, 1: 10}  0.993 ± 0.002  0.0021 ± 0.0002
```

The ROC-AUC is the same for the three models, so the class weights did not make the model better at separating the classes. Any difference you might see on a single split would be well within the variability of the estimate.

The Brier score, on the other hand, more than doubles with balanced class weights. The weights distorted the predicted probabilities, as the theory predicts.

### Making Cost-Sensitive Decisions With Real Costs

Now, let's assume that domain experts told us that missing a positive costs 20 times more than a false alarm. We write a function that calculates the total cost of the predictions, and turn it into a scorer that scikit-learn can use:

```
C_FP, C_FN = 1, 20

def total_cost(y_true, y_pred):
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return C_FP * fp + C_FN * fn

cost_scorer = make_scorer(total_cost, greater_is_better=False)

p_star = C_FP / (C_FP + C_FN)
print(f"p* = {p_star:.3f}")
```

The output shows the threshold derived from the costs:

```
p* = 0.048
```

We will compare four ways of making decisions:

- **Default threshold:** a model trained without weights, used at the threshold of 0.5, which ignores the costs.
- **Class weights:** a model trained with the costs as class weights, used at the threshold of 0.5.
- **Threshold p\*:** a model trained without weights, used at the cost-derived threshold, with scikit-learn's `FixedThresholdClassifier`.
- **Tuned threshold:** a model trained without weights, whose threshold is tuned with cross-validation to minimize the total cost, with `TunedThresholdClassifierCV`.

With very imbalanced data, the best threshold can be very small, so we let `TunedThresholdClassifierCV` search 200 thresholds between 0.0001 and 1 on a logarithmic scale:

```
models = {
    "Default threshold": gbm(),
    "Class weights": gbm({0: C_FP, 1: C_FN}),
    "Threshold p*": FixedThresholdClassifier(gbm(), threshold=p_star),
    "Tuned threshold": TunedThresholdClassifierCV(
        gbm(),
        scoring=cost_scorer,
        thresholds=np.logspace(-4, 0, 200),
        cv=5,
    ),
}

results = {}
for name, model in models.items():
    cv_res = cross_validate(
        model, X, y, cv=cv,
        scoring={"cost": cost_scorer, "recall": "recall", "precision": "precision"},
        return_estimator=True,
    )
    cost = -cv_res["test_cost"]
    results[name] = {
        "Cost": f"{cost.mean():.0f} ± {cost.std():.0f}",
        "Recall": f"{cv_res['test_recall'].mean():.3f} ± {cv_res['test_recall'].std():.3f}",
        "Precision": f"{cv_res['test_precision'].mean():.3f} ± {cv_res['test_precision'].std():.3f}",
    }
    if name == "Tuned threshold":
        thresholds = [est.best_threshold_ for est in cv_res["estimator"]]

print(pd.DataFrame(results).T)
print(f"Tuned threshold: {np.mean(thresholds):.3f} ± {np.std(thresholds):.3f}")
```

In the following output, we see the total cost per fold, together with the recall and precision of each approach:

```
                         Cost         Recall      Precision
Default threshold  1213 ± 149  0.769 ± 0.029  0.923 ± 0.020
Class weights       865 ± 153  0.843 ± 0.031  0.816 ± 0.018
Threshold p*        911 ± 134  0.836 ± 0.026  0.788 ± 0.030
Tuned threshold      721 ± 84  0.913 ± 0.017  0.490 ± 0.094
Tuned threshold: 0.006 ± 0.002
```

Ignoring the costs is clearly expensive: the default threshold produces the highest cost. Training with class weights and using the cost-derived threshold $$p^*$$ on a model without weights give the same cost, within one standard deviation, and very similar recall and precision.

That is the equivalence between class weights and threshold shifting, in practice. The two approaches make essentially the same decisions, but the threshold approach did not need to retrain the model or distort its probabilities.

The tuned threshold gives the lowest cost. The theoretical threshold $$p^*$$ assumes calibrated probabilities, and the probabilities of this gradient boosting machine are not perfectly calibrated, so the best threshold, around 0.006, is lower than 0.048. Tuning the threshold on data finds the best operating point regardless of the calibration.

> In chapter 4 of my [book](https://www.trainindata.com/p/imbalanced-data-myths-mistakes-solutions-book), I go through a full profit maximization example, with instance-dependent costs from a real fundraising campaign. You'll see how sample weights and threshold tuning compare when we account for uncertainty, and how to pass the cost of each observation to `TunedThresholdClassifierCV` with scikit-learn's metadata routing.

### When Only the Class Imbalance Matters: Adjust the Threshold

What if we don't have real costs, and we just want a model that performs well on both classes? A common choice is to train with balanced class weights and evaluate the balanced accuracy, which is the average of the recall of each class.

Let's compare that with a model trained on the original data, whose threshold is tuned to maximize the [balanced accuracy](https://www.blog.trainindata.com/a-data-scientists-guide-to-balanced-accuracy/):

```
models = {
    "Balanced class weights": gbm("balanced"),
    "Tuned threshold": TunedThresholdClassifierCV(
        gbm(),
        scoring="balanced_accuracy",
        thresholds=np.logspace(-4, 0, 200),
        cv=5,
    ),
}

results = {}
for name, model in models.items():
    cv_res = cross_validate(
        model, X, y, cv=cv, scoring="balanced_accuracy", return_estimator=True,
    )
    ba = cv_res["test_score"]
    results[name] = {"Balanced accuracy": f"{ba.mean():.3f} ± {ba.std():.3f}"}
    if name == "Tuned threshold":
        thresholds = [est.best_threshold_ for est in cv_res["estimator"]]

print(f"Tuned threshold: {np.mean(thresholds):.3f} ± {np.std(thresholds):.3f}")
print(pd.DataFrame(results).T)
```

The output shows the threshold selected in each fold, followed by the balanced accuracy of each model:

```
Tuned threshold: 0.001 ± 0.000
                       Balanced accuracy
Balanced class weights     0.942 ± 0.014
Tuned threshold            0.958 ± 0.014
```

The model trained on the original data with a tuned threshold reaches the same or a slightly higher balanced accuracy, within one standard deviation. It does so at a much lower threshold than 0.5, which is exactly what the class weights were doing implicitly.

So, when the class imbalance is the only issue, train on the original data and adjust the threshold. You get the same decisions, and you keep probabilities that you can trust.

### Using Sample Weights for Instance-Dependent Costs

When the cost of an error depends on each observation, we can pass a vector of costs to the `sample_weight` parameter of the `fit()` method. For example, if missing a fraudulent transaction costs its amount, and investigating any transaction costs a fixed fee, the weights could look like this:

```
sample_weight = np.where(y_train == 1, amount_train, investigation_cost)

model = gbm()
model.fit(X_train, y_train, sample_weight=sample_weight)
```

Here, `amount_train` holds the amount of each transaction in the training set, and `investigation_cost` the cost of reviewing one. Most scikit-learn models support `sample_weight`.

This is where cost-sensitive learning shines, because the costs reflect real consequences. Even so, the same decisions can be reached by training on the original data and tuning the threshold with a scorer that uses the cost of each observation.

## Other Ways to Make Classifiers Cost-Sensitive

Resampling the training data is equivalent to class weights. [Undersampling](https://www.blog.trainindata.com/undersampling-techniques-for-imbalanced-data/) the majority class, or [oversampling](https://www.blog.trainindata.com/oversampling-techniques-for-imbalanced-data/) the minority class with methods like [SMOTE](https://www.blog.trainindata.com/smote-in-python-a-guide-to-balanced-datasets/) or [ADASYN](https://www.blog.trainindata.com/adasyn-adaptive-synthetic-sampling-for-imbalanced-datasets/), also shift the decision boundary, with the added risk of losing information or creating unrealistic data.

When an algorithm does not support weights, MetaCost, proposed by Pedro Domingos in 1999, makes it cost-sensitive by relabeling the training data according to the costs and the estimated probabilities. In practice, threshold adjustment is a much simpler alternative.

## Myths, Mistakes and Modern Solutions

**Myth:** class weights and resampling improve model performance. In practice, they shift the decision boundary so that the default threshold of 0.5 makes cost-sensitive decisions, which is equivalent to adjusting the threshold of a model trained on the original data.

**Mistake:** comparing models with a single number from a single train and test split. Small differences in ROC-AUC or other metrics often fall within the variability of the estimate, so always report the standard deviation across folds or bootstrap samples.

**Modern solution:** train on the original class distribution and tune the threshold to the metric or cost you care about, for example with `TunedThresholdClassifierCV`. Reach for class weights or sample weights when you have real costs and they make the implementation easier, knowing that they will distort the probabilities.

## Conclusion

Cost-sensitive learning minimizes the expected cost of a model's decisions instead of its error rate. The costs determine the decision threshold, and class weights, sample weights and resampling are different ways of moving the decision boundary to that threshold.

Cost-sensitive learning makes sense when we know the real costs of each type of error, especially when they vary from one observation to another. If the only problem is the class imbalance, we are better off training on the original data and adjusting the threshold, which gives the same decisions and keeps the probabilities calibrated.

If you want to understand why, step by step, from the expected cost to the resampling ratio and the translation theorem, and see the evidence from experiments on many datasets, check out chapter 4 of my book [Imbalanced Data: Myths, Mistakes and Modern Solutions](https://www.trainindata.com/p/imbalanced-data-myths-mistakes-solutions-book). You can also read our article on [machine learning with imbalanced data](https://www.blog.trainindata.com/machine-learning-with-imbalanced-data/).

[![Imbalanced Data: Myths, Mistakes and Modern Solutions - book by Soledad Galli]({{ site.baseurl }}/assets/images/imbalanced-data-book-cover.jpg)](https://www.trainindata.com/p/imbalanced-data-myths-mistakes-solutions-book)

## Further Reading and Additional Resources

- Elkan. 2001. [The Foundations of Cost-Sensitive Learning](https://cseweb.ucsd.edu/~elkan/rescale.pdf). Proceedings of the 17th International Joint Conference on Artificial Intelligence.
- Zadrozny, Langford and Abe. 2003. [Cost-Sensitive Learning by Cost-Proportionate Example Weighting](https://doi.org/10.1109/ICDM.2003.1250950). Third IEEE International Conference on Data Mining.
- Domingos. 1999. [MetaCost: A General Method for Making Classifiers Cost-Sensitive](https://dl.acm.org/doi/10.1145/312129.312220). Proceedings of the Fifth International Conference on Knowledge Discovery and Data Mining.
- Ling and Sheng. 2011. [Cost-Sensitive Learning](https://doi.org/10.1007/978-0-387-30164-8_181). Encyclopedia of Machine Learning. Springer.
- Scikit-learn documentation: [post-tuning the decision threshold for cost-sensitive learning](https://scikit-learn.org/stable/auto_examples/model_selection/plot_cost_sensitive_learning.html).
