---
layout: post
title: "Should You Use Imbalanced-Learn in 2026?"
author: sole
description: "I discuss the latest evidence on undersampling, SMOTE and resampling ensembles for imbalanced data, and whether imbalanced-learn (imblearn) is still useful."
excerpt: "I discuss the latest evidence on undersampling, SMOTE and resampling ensembles for imbalanced data, and whether imbalanced-learn (imblearn) is still useful."
categories: [Data Preprocessing, Imbalanced Data]
image: assets/images/posts/should-you-use-imbalanced-learn-in-2025/imbalance-learn-2025.png
---

Imbalanced-learn, or imblearn, is a Python open-source library that supports methods that have long been proposed as effective to tackle alleged poor performance of machine learning models trained using imbalanced datasets.

Among the methods supported by imbalanced-learn, we find oversampling and data generating techniques like SMOTE; undersampling and data cleaning methods, like Edited Nearest Neighbors and Condensed Nearest Neighbors; and bagging and boosting algorithms specifically designed to perform better with imbalanced datasets, like EasyEnsemble and RusBoost.

Imbalanced-learn also supports specific model evaluation metrics designed to, in theory, better determine a machine learning model‘s performance that was trained on imbalanced data.

But, are these methods and tools really effective to tackle imbalanced data? That’s what we’ll discuss throughout this article.

> In my book [Imbalanced Data: Myths, Mistakes and Modern Solutions](https://www.trainindata.com/p/imbalanced-data-myths-mistakes-solutions-book), I review the scientific literature behind undersampling, oversampling and cost-sensitive learning, and test these methods with modern classifiers on dozens of datasets. This article summarizes what I found.
>
> [![Imbalanced Data: Myths, Mistakes and Modern Solutions - book by Soledad Galli]({{ site.baseurl }}/assets/images/imbalanced-data-book-cover.jpg){: width="240"}](https://www.trainindata.com/p/imbalanced-data-myths-mistakes-solutions-book)

**The short answer, up front:** undersampling, oversampling, SMOTE, and cost-sensitive learning don't make a model better at discriminating between classes. What they do is shift the decision boundary so that, at the default classification threshold of 0.5, we correctly flag a larger proportion of the minority class, which is usually the class we care about most.

You can get that exact same effect by training on the original, unmodified data and adjusting the classification threshold afterward, without touching imbalanced-learn at all. The rest of this article walks through the evidence for that claim.

## Imbalanced datasets: what are they?

Imbalanced datasets, datasets where one class significantly outnumbers the others, are a common occurrence in real-world applications, such as fraud detection or disease diagnosis, among others.

When classes are imbalanced, machine learning algorithms are said to favor the majority class, leading to poor predictions for the minority class. That is certainly true for the so called “weak learners”, like decision trees and support vector machines. Strong classifiers have been shown to be effective at predicting the minority class in spite of the imbalance (read on).

## Methods to tackle class imbalance

To address this “class imbalance problem”, various methods have been described and later on implemented in the Python library [Imbalanced-Learn](https://github.com/scikit-learn-contrib/imbalanced-learn). These methods are said to enhance model performance and ensure that minority classes are accurately represented in predictions.

Among the methods designed to tackle the alleged classification problems said to occur when using imbalanced datasets in machine learning, we find:

- [Oversampling](https://www.blog.trainindata.com/oversampling-techniques-for-imbalanced-data/) and data augmentation methods, like random oversampling and SMOTE and its variants.
- [Undersampling](https://www.blog.trainindata.com/undersampling-techniques-for-imbalanced-data/) and data cleaning techniques, like random undersampling, Edited Nearest Neighbors and the Neighborhood Cleaning Rule.
- Bagging and Boosting algorithms that include under- or oversampling while generating the bootstrapped datasets that are used to train each tree in the ensemble, like EasyEnsemble.

But… do these methods really work?

## The hype of SMOTE

[SMOTE](https://www.blog.trainindata.com/smote-in-python-a-guide-to-balanced-datasets/) is a data generating method, said to create datapoints that look like those from the minority class. By adding more minority class-look like examples to the dataset, we remove the imbalance and hence, shift the decision boundary toward the minority class. Or so the story goes.

In 2022, a scientific article came out, making a systematic comparison of the performance of various machine learning models, including weak and strong learners, trained to classify various imbalanced datasets, with and without the use of random oversampling and SMOTE.

![To SMOTE or Not to SMOTE]({{ site.baseurl }}/assets/images/posts/should-you-use-imbalanced-learn-in-2025/to-smote-or-not-to-smote-paper.png)

Weak learners included decision trees, support vector machines and adaboost, and strong learners included xgboost and catboost.

To evaluate model performance they used threshold dependent metrics, like precision and recall, and threshold independent metrics like the ROC-AUC.

For the threshold dependent metrics, they compared the model performance returned when the metric was calculated using the “classical” 0.5 probability threshold, or a more suitable threshold for imbalanced classes, that will typically be somewhere below 0.5.

What they saw, is that random oversampling and SMOTE did indeed show an increase in model performance **WHEN**, the threshold dependent metrics were calculated using a probability threshold of 0.5. **HOWEVER**, and this is the important bit, the same effect could be obtained when calculating the metric using a tuned threshold, **WITHOUT** oversampling or SMOTE.

What does this mean? that the whole SMOTE hype was based on metrics that were calculated wrongly, by not adjusting the probability threshold, something that is common practice in data science, when working with imbalanced classes.

In fact, if you want to see these evidence in action, check out Guillaume Lamaitre’s (one of the imbalanced-learn developers) talk, where he discusses this issue and demonstrates it with Python code using scikit-learn.

<div class="video-embed"><iframe src="https://www.youtube.com/embed/6YnhoCfArQo?feature=oembed" title="EuroSciPy 2023 - Get the best from your scikit-learn classifier" width="560" height="315" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe></div>

In fact, glemaitre goes on to discourage the use of SMOTE, [ADASYN](https://www.blog.trainindata.com/adasyn-adaptive-synthetic-sampling-for-imbalanced-datasets/) and TOMEK Links (a cleaning method), and instead tune the probability threshold when working with imbalanced datasets.

## So, does oversampling not work?

The authors of the 2022 article “[To SMOTE, or not to SMOTE?](https://arxiv.org/abs/2201.08528)”, concluded that if we use strong classifiers like xgboost, then oversampling or SMOTE do not really help improve the model’s performance.

They identified, however, a few scenarios where oversampling and SMOTE might be useful, and they are:

- SMOTE-like methods could improve the performance of “weak” learners like multilayer perceptrons, decision trees, support vector machines, and, interestingly, lightGBMs.
- SMOTE-like methods could improve the performance of models that do not return a probability as output, and hence, we can’t optimize the threshold to calculate the threshold dependent metrics.

However, on the later, they mentioned that random oversampling returned similar results to SMOTE, and is yet a simpler technique, so random oversampling is recommended over SMOTE.

**So, should you use imbalanced-learn in 2026?**

Well, for oversampling methods, only if you are training weak learners. But the recommendation is, of course, to use strong classifiers like xgboost and tune the probability threshold.

## Undersampling methods for imbalanced datasets

Undersampling methods reduce the number of samples of the majority class. There are various methods, typically divided into cleaning or fixed undersampling.

Fixed undersampling methods reduce the number of samples up to a certain extent, defined by the user. For example, they reduce the number of observations until we have equal number of samples of the majority and minority class. The classical method is random undersampling, while the instance hardness threshold is another fixed undersampling technique.

Cleaning undersampling methods, instead, remove “problematic” observations. Problematic observations are usually those that are hard to classify, for one reason or another. For example, they’d remove observations from the majority class if they are surrounded mostly by observations of the minority class. All cleaning methods involve the use of k-nearest neighbors (knn) to find those observations to remove, and hence “clean” the data.

### What the literature says about undersampling

Most undersampling methods were developed between the late 1960s and the early 2000s, when k-nearest neighbors and decision trees were the go-to classifiers. The original studies evaluated them with those weak learners, and with threshold dependent metrics like accuracy, error rate, recall or the geometric mean.

The classification threshold was rarely reported, so we can assume that these metrics were calculated at the default threshold of 0.5. And even then, the improvements were not consistent across datasets and classifiers.

In other words, undersampling methods have only looked successful when models were evaluated with threshold dependent metrics and the threshold was not adjusted. That is exactly the same flaw we found behind the SMOTE hype.

### Undersampling with strong classifiers

What about modern classifiers like random forests and gradient boosting machines? Only a handful of studies evaluated undersampling with strong classifiers and threshold independent metrics, like the ROC-AUC or the PR-AUC, and most of them found that undersampling gave little or no improvement.

For example, [Wainer (2023)](https://arxiv.org/abs/1810.07168) found no statistically significant difference in ROC-AUC between models trained with or without undersampling across 58 datasets. [Kim et al. (2022)](https://doi.org/10.1371/journal.pone.0271260) found that undersampling decreased, rather than increased, the performance of the models across 31 datasets.

The only convincing improvement I found in the literature comes from [Hasanin et al. (2019)](https://doi.org/10.1186/s40537-019-0274-4), on a dataset with 1.5 million observations and only 0.27% positive cases. So undersampling might help in very specific scenarios, like huge datasets with extreme imbalance, where discarding observations still leaves plenty of data to train the model.

### My experiments: undersampling does not improve performance

Given the scarce evidence, I ran the experiments myself. I combined random undersampling, the instance hardness threshold and the cleaning methods (ENN, RENN, AllKNN, Tomek links, CNN, NCR and OSS) with random forests, AdaBoost, scikit-learn's gradient boosting machine, CatBoost, XGBoost and LightGBM, and evaluated them across 37 imbalanced datasets.

The code and results are available in my [resampling experiments repository](https://github.com/solegalli/resampling-experiments). Each model was evaluated with cross-validation, and I report the mean and standard deviation of each metric, together with statistical tests to determine whether the differences were significant.

The following figure shows the ROC-AUC (left) and the Brier score (right) of the models trained on the original data (blue) and after random undersampling (green) for 5 of the 37 datasets. The red band shows the performance of a random forest trained on the original data, and the horizontal brackets mark statistically significant differences.

![Random undersampling does not improve the ROC-AUC and worsens the Brier score of random forests, AdaBoost, gradient boosting, CatBoost, LightGBM and XGBoost.]({{ site.baseurl }}/assets/images/posts/should-you-use-imbalanced-learn-in-2025/random-undersampling-roc-auc-brier-score.png)

In no case did a model trained after random undersampling outperform the same model trained on the original data. The results were consistent for the remaining 32 datasets.

On top of that, undersampling consistently increased the Brier score. Since the ROC-AUC barely changed, this means that undersampling made the predicted probabilities less accurate, in other words, it [miscalibrated the model](https://www.blog.trainindata.com/probability-calibration-in-machine-learning/).

### Undersampling shifts the threshold

What about threshold dependent metrics? The following figure shows the balanced accuracy of each model (left), and the threshold that maximized the balanced accuracy (right).

![Balanced accuracy of models trained on the original or undersampled data, and the threshold that maximizes it.]({{ site.baseurl }}/assets/images/posts/should-you-use-imbalanced-learn-in-2025/random-undersampling-balanced-accuracy-threshold.png)

When the balanced accuracy was calculated at its optimal threshold, random undersampling did not improve it. Notice how the models trained on the original data reach their best balanced accuracy at thresholds below 0.5, while models trained after undersampling reach it at higher thresholds.

This is the key point: undersampling changes the threshold at which we should classify an observation as the minority class. It does not make the model better at ranking observations.

The cleaning methods told a similar story. None of them improved the performance of the models consistently across datasets, and in some cases, like ENN with LightGBM on the adult dataset, they degraded it substantially.

## So, does undersampling work?

No. Undersampling does not improve the performance of modern classifiers. At best, it shifts the decision boundary so that we get good values of threshold dependent metrics at the default threshold of 0.5, and we can obtain the same result by training on the original data and lowering the threshold.

The only compelling reason to undersample is computational. If your dataset is so big that training becomes prohibitively slow, random undersampling of the majority class can be a shortcut to reduce training time. Keep in mind that cleaning methods based on knn are slow and don't scale, so they won't help you there.

> In chapter 5 of my book [Imbalanced Data: Myths, Mistakes and Modern Solutions](https://www.trainindata.com/p/imbalanced-data-myths-mistakes-solutions-book), I review the studies behind each undersampling method in detail, and discuss the full results of these experiments.

## Bagging and Boosting algorithms for imbalanced datasets

Throughout the years, various bagging and boosting methods that add a step of under- or oversampling while creating the bootstrapped datasets to train the model have been described. Imbalanced-learn supports the following ones:

- EasyEnsemble
- RUSBoost
- Balanced Random Forests
- Balanced Bagging (it balances the data to bag any classifier)

Do we really need them? If undersampling does not improve the performance of a model, adding it to the bagging or boosting procedure should not improve it either. Each model in the ensemble is still trained on resampled data, so the ensemble inherits the same threshold shift, and the same miscalibration.

### Resampling ensembles vs standard ensembles

I tested this in the same [resampling experiments repository](https://github.com/solegalli/resampling-experiments). I compared EasyEnsemble, RUSBoost and Balanced Random Forests with random forests, AdaBoost, gradient boosting, CatBoost, LightGBM and XGBoost trained on the original data, across the same 37 datasets.

The following figure shows the ROC-AUC of each model for 8 of those datasets. As before, the red band shows the performance of a random forest trained on the original data, and the horizontal brackets mark statistically significant differences.

![ROC-AUC of RUSBoost, EasyEnsemble and Balanced Random Forests compared to random forests, AdaBoost, gradient boosting, CatBoost, LightGBM and XGBoost trained on the original data.]({{ site.baseurl }}/assets/images/posts/should-you-use-imbalanced-learn-in-2025/resampling-ensembles-vs-standard-ensembles-roc-auc.png)

None of the resampling ensembles consistently beat the best standard ensemble. In every dataset, there was always a model trained on the original data that performed as well as, if not better than, the resampling ensembles. EasyEnsemble, in particular, was often among the worst performing models.

Averaged across the 37 datasets, the ROC-AUC of Balanced Random Forests (0.908), RUSBoost (0.906) and EasyEnsemble (0.888) was at or below that of the standard ensembles (0.895 to 0.914). The same pattern held for the average precision, recall, F1-score, Matthews correlation coefficient and balanced accuracy.

### Resampling ensembles and calibration

The biggest difference showed up in the Brier score. The resampling ensembles returned Brier scores between 0.11 and 0.16, while random forests and gradient boosting models trained on the original data stayed around 0.04.

This means that the resampling ensembles return poorly calibrated probabilities, which is a problem whenever we need the probabilities themselves, for example, to estimate risk or to make cost-sensitive decisions.

So the answer is no, we don't need resampling ensembles. A gradient boosting machine trained on the original data, with a tuned threshold, does the job, and it is faster to train too.

## So, should we use imbalanced-learn in 2026?

As the latest evidence suggests, the advice when working with imbalanced data goes as follows:

1. Use strong classifiers when possible, like xgboost, lightGBM or catboost.
2. Train them on the original data, without resampling.
3. Evaluate performance by combining a threshold independent metric, like the [ROC-AUC](https://www.blog.trainindata.com/auc-roc-analysis/) or the [PR-AUC](https://www.blog.trainindata.com/precision-recall-curves/), with a threshold dependent metric, like precision or recall.
4. For threshold dependent metrics, optimize the threshold instead of using the default 0.5.

### What about cost-sensitive learning?

[Cost-sensitive learning](https://www.blog.trainindata.com/cost-sensitive-learning-for-imbalanced-data/) is often suggested as the alternative to resampling. But when the class weights are set from the class imbalance, cost-sensitive learning does not improve model performance either. Class weights, resampling and threshold tuning are equivalent: all three shift the decision threshold.

Having said this, if for some reason you need to make decisions at the default threshold of 0.5, class weights are a better option than resampling. They produce the same shift in the decision boundary, without discarding data or creating synthetic examples, and without adding an extra step to your pipeline.

Even better, if you know the real costs of each type of error for your business, for example, how much a missed fraudulent transaction costs compared to a false alarm, then cost-sensitive learning is the way to go. You can derive the optimal threshold from those costs, or use sample weights when the costs vary from one observation to another.

If you want to dig deeper, in chapter 4 of my book [Imbalanced Data: Myths, Mistakes and Modern Solutions](https://www.trainindata.com/p/imbalanced-data-myths-mistakes-solutions-book), I discuss everything you need to know about cost-sensitive learning for imbalanced data, from the cost matrix and the optimal threshold to class weights and instance-dependent costs.

### When could resampling still be useful?

If you are, for some reason, using weaker learners, or models that do not return probabilities, oversampling may still help. In that case, keep in mind that random oversampling tends to give similar results to more complex methods like SMOTE, so there is no need to over-complicate things.

And if your dataset is huge, random undersampling can reduce training time, as discussed earlier. For random under- or oversampling, there is no need to use imbalanced-learn, you can resample your dataset with mainstream libraries like pandas or numpy.

### Why I still show you how to use imblearn

You will still come across imblearn in plenty of legacy code, older tutorials and Kaggle notebooks. And it is a handy tool if you want to run your own experiments and confirm that resampling does not improve the performance of your models.

So, for the rest of this article, I'll show you how to install and use imbalanced-learn. Treat these demos as a reference to read and maintain existing code, or to run your own comparisons, and not as a recommendation to resample your data.

## Imblearn: How to Use the Imbalanced-Learn Python Library

Imbalanced-learn is an open-source Python library that implements resampling methods for imbalanced datasets. It follows the scikit-learn API, and we import it in Python as `imblearn`.

The library includes methods for oversampling, undersampling, and hybrid sampling, as well as the resampling ensembles we discussed earlier.

The installation of imbalanced-learn is straightforward:

```
pip install imbalanced-learn
```

### Techniques for Balancing Data in Imbalanced-Learn

Imbalanced-Learn offers several techniques for balancing datasets. The three primary methods are oversampling, undersampling, and hybrid sampling. Let’s look at each one of them.

#### 1. Oversampling

Oversampling techniques increase the representation of the minority class by generating synthetic data points or duplicating existing samples. This shifts the decision boundary toward the minority class, so at the default threshold, fewer of its examples get missed.

**Random Oversampling**: This basic technique randomly duplicates instances of the minority class to balance the dataset. It’s simple to implement but can lead to overfitting, because we are in essence, duplicating data points.

Here’s how to apply random oversampling with imblearn:

```
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler

# Step 1: Create a synthetic imbalanced dataset
X, y = make_classification(
    n_samples=500,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    weights=[0.9, 0.1],
    flip_y=0,
    random_state=42
)

# Step 2: Visualize the original dataset before oversampling
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label="Class 0", alpha=0.5, edgecolor="k")
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label="Class 1", alpha=0.5, edgecolor="k")
plt.title("Original Imbalanced Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()

# Step 3: Apply RandomOverSampler to balance the dataset
oversample = RandomOverSampler(sampling_strategy='minority', random_state=42)
X_over, y_over = oversample.fit_resample(X, y)

# Step 4: Visualize the dataset after oversampling
plt.subplot(1, 2, 2)
plt.scatter(X_over[y_over == 0][:, 0], X_over[y_over == 0][:, 1], label="Class 0", alpha=0.5, edgecolor="k")
plt.scatter(X_over[y_over == 1][:, 0], X_over[y_over == 1][:, 1], label="Class 1", alpha=0.5, edgecolor="k")
plt.title("Dataset After Random Oversampling")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()

plt.tight_layout()
plt.show()
```

In the following image, we compare the original classes distribution, with the data distribution after applying random oversampling. Note that we see exactly the same data points for the minority class, just in darker orange, due to the overlay of points on themselves:

![Image comparing the class distribution before and after applying random oversampling with imbalanced learn.]({{ site.baseurl }}/assets/images/posts/should-you-use-imbalanced-learn-in-2025/random-oversampling-before-after-distribution.png)

**Synthetic Minority Over-sampling Technique (SMOTE)**: SMOTE generates new data resembling the minority class by interpolating between existing instances of the minority class.

Here’s how to implement it with imblearn:

```
from imblearn.over_sampling import SMOTE

smote = SMOTE()
X_smote, y_smote = smote.fit_resample(X, y)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label="Class 0", alpha=0.5, edgecolor="k")
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label="Class 1", alpha=0.5, edgecolor="k")
plt.title("Original Imbalanced Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(X_smote[y_smote == 0][:, 0], X_smote[y_smote == 0][:, 1], label="Class 0", alpha=0.5, edgecolor="k")
plt.scatter(X_smote[y_smote == 1][:, 0], X_smote[y_smote == 1][:, 1], label="Class 1", alpha=0.5, edgecolor="k")
plt.title("Dataset After SMOTE")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()

plt.tight_layout()
plt.show()
```

In the following image, we compare the distribution of the minority class after creating synthetic examples. Note that we are not simply duplicating points, there are newer data examples:

![Image comparing the class distribution before and after applying SMOTE with imbalanced learn.]({{ site.baseurl }}/assets/images/posts/should-you-use-imbalanced-learn-in-2025/smote-before-after-distribution.png)

#### 2. Undersampling

Undersampling reduces the size of the majority class, balancing the dataset by removing excess samples. While undersampling can result in data loss, it can be useful when dealing with large majority classes that may lead to computational inefficiencies.

**Random Undersampling**: This technique randomly removes samples from the majority class to achieve balance. It’s simple but can risk losing valuable information from the majority class.

Here’s how to implement it with imblearn:

```
from imblearn.under_sampling import RandomUnderSampler

undersample = RandomUnderSampler()
X_under, y_under = undersample.fit_resample(X, y)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label="Class 0", alpha=0.5, edgecolor="k")
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label="Class 1", alpha=0.5, edgecolor="k")
plt.title("Original Imbalanced Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(X_under[y_under == 0][:, 0], X_under[y_under == 0][:, 1], label="Class 0", alpha=0.5, edgecolor="k")
plt.scatter(X_under[y_under == 1][:, 0], X_under[y_under == 1][:, 1], label="Class 1", alpha=0.5, edgecolor="k")
plt.title("Dataset After Random Undersampling")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()

plt.tight_layout()
plt.show()
```

In the following image, we compare the class distributions before and after undersampling the majority class:

![Image comparing the class distribution before and after applying random undersampling with imbalanced learn.]({{ site.baseurl }}/assets/images/posts/should-you-use-imbalanced-learn-in-2025/random-undersampling-before-after-distribution.png)

**Edited Nearest Neighbours (ENN)**: ENN is an undersampling method that removes instances in the majority class if their nearest neighbors belong to a different class. This technique shifts the decision boundary by cleaning up majority class samples near it.

Here’s how to implement it with imblearn:

```
from imblearn.under_sampling import EditedNearestNeighbours

enn = EditedNearestNeighbours()
X_en, y_en = enn.fit_resample(X, y)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label="Class 0", alpha=0.5, edgecolor="k")
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label="Class 1", alpha=0.5, edgecolor="k")
plt.title("Original Imbalanced Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()


plt.subplot(1, 2, 2)
plt.scatter(X_en[y_en == 0][:, 0], X_en[y_en == 0][:, 1], label="Class 0", alpha=0.5, edgecolor="k")
plt.scatter(X_en[y_en == 1][:, 0], X_en[y_en == 1][:, 1], label="Class 1", alpha=0.5, edgecolor="k")
plt.title("Dataset After ENN")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()

plt.tight_layout()
plt.show()
```

In the following image, we compare the class distributions before and after applying the cleaning method edited nearest neighbor. Note how the method has removed observations from the majority class close to those of the minority class:

![Image comparing the class distribution before and after applying edited nearest neighbours with imbalanced learn.]({{ site.baseurl }}/assets/images/posts/should-you-use-imbalanced-learn-in-2025/enn-before-after-distribution.png)

#### 3. Hybrid Sampling

Hybrid methods combine oversampling and undersampling techniques, for example, SMOTE followed by ENN or Tomek links. They aim to balance the dataset while avoiding the overfitting of oversampling and the data loss of undersampling.

The python implementation is similar to those of under- or oversampling techniques: we just import the resampler and apply it with `fit_resample`.

### Evaluation Metrics for Imbalanced Data

After resampling, it’s essential to assess the effectiveness of the approach. Standard evaluation metrics like accuracy can be misleading for imbalanced datasets, so alternative metrics like precision, recall, F1-score, and the Area Under the Receiver Operating Characteristic Curve (AUC-ROC) are recommended.

- **Precision**: Measures the proportion of true positive predictions among all positive predictions.
- **Recall**: Indicates the model’s ability to capture all positive instances.
- **F1-Score**: The harmonic mean of precision and recall, balancing both metrics to provide a more reliable evaluation for imbalanced data.
- **ROC-AUC**: The area under the ROC curve, which provides insight into the model’s sensitivity across different thresholds.

Comparing threshold dependent metrics before and after resampling shows how the decision boundary shifted at the default threshold. Remember, the same shift can be achieved by tuning the threshold on the model trained on the original data.

### Integrating Imblearn with Scikit-Learn Pipelines

Imbalanced-learn integrates with scikit-learn. We can add imblearn's resampling methods to a pipeline, together with data preprocessing, feature selection, and model training steps.

For example, it’s possible to integrate SMOTE within a pipeline to balance the classes during the preprocessing stage:

```
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
   ('smote', SMOTE()),
   ('classifier', RandomForestClassifier())
])

pipeline.fit(X, y)
```

Below, the evidence that the pipeline worked ;)

![Pipeline combining resampling with imbalanced-learn with training a classifier from scikit-learn.]({{ site.baseurl }}/assets/images/posts/should-you-use-imbalanced-learn-in-2025/imbalanced-learn-resampling-pipeline.png)

### Using Imblearn to Check Whether Resampling Improves Performance

Because the resampler sits inside the pipeline, it is applied only to the training folds during cross-validation, and the model is always evaluated on untouched data. This makes the imblearn pipeline a handy tool to check, on your own data, whether resampling improves the performance of your model.

Let's compare a random forest trained on the original data with the same model trained after SMOTE or random undersampling. We'll use a harder dataset with 20 features and 5% of observations in the minority class, and evaluate each model with repeated cross-validation, so that we can report the mean and the standard deviation of each metric:

```
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

X, y = make_classification(
    n_samples=5000,
    n_features=20,
    n_informative=5,
    weights=[0.95, 0.05],
    flip_y=0.01,
    random_state=0,
)

models = {
    "original": RandomForestClassifier(random_state=0),
    "smote": Pipeline([
        ("smote", SMOTE(random_state=0)),
        ("classifier", RandomForestClassifier(random_state=0)),
    ]),
    "undersampling": Pipeline([
        ("rus", RandomUnderSampler(random_state=0)),
        ("classifier", RandomForestClassifier(random_state=0)),
    ]),
}

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)

results = {}
for name, model in models.items():
    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=["roc_auc", "average_precision", "neg_brier_score"],
        n_jobs=-1,
    )
    results[name] = {
        "roc_auc": f"{scores['test_roc_auc'].mean():.3f} ± {scores['test_roc_auc'].std():.3f}",
        "average_precision": f"{scores['test_average_precision'].mean():.3f} ± {scores['test_average_precision'].std():.3f}",
        "brier_score": f"{-scores['test_neg_brier_score'].mean():.3f} ± {scores['test_neg_brier_score'].std():.3f}",
    }

print(pd.DataFrame(results).T)
```

The code returns the following results:

```
                     roc_auc average_precision    brier_score
original       0.918 ± 0.030     0.763 ± 0.058  0.024 ± 0.003
smote          0.935 ± 0.034     0.726 ± 0.044  0.030 ± 0.002
undersampling  0.919 ± 0.032     0.660 ± 0.044  0.093 ± 0.007
```

The ROC-AUC of the three models is within one standard deviation of each other, so neither SMOTE nor random undersampling made the random forest better at separating the classes. The average precision, on the other hand, dropped after resampling, and the Brier score increased, particularly after undersampling, which means that the predicted probabilities became less accurate.

This is the same pattern I observed across the 37 datasets in my experiments. Resampling changes the probabilities returned by the model, while its ability to separate the classes stays the same.

## Conclusion

Imbalanced-learn was a great initiative that aimed to democratize and make easily accessible methods to tackle imbalanced datasets, as soon as these methods were described.

Since then, stronger classifiers like xgboost came to the field, and more thorough comparisons of these methods were made. The evidence from the literature, and from my own experiments across 37 datasets, shows that undersampling, oversampling and resampling ensembles do not improve the performance of modern classifiers. They shift the decision threshold and, in the process, distort the predicted probabilities.

So, should you use imbalanced-learn in 2026? For most projects, no. Train a strong classifier on the original data and tune the threshold, or use [cost-sensitive learning](https://www.blog.trainindata.com/cost-sensitive-learning-for-imbalanced-data/) when you know the real costs of each error. Keep imblearn around to maintain legacy code, or to run your own experiments.

If you want to learn more about working with imbalanced datasets, check out my book [Imbalanced Data: Myths, Mistakes and Modern Solutions](https://www.trainindata.com/p/imbalanced-data-myths-mistakes-solutions-book).

[![Imbalanced Data: Myths, Mistakes and Modern Solutions - book by Soledad Galli]({{ site.baseurl }}/assets/images/imbalanced-data-book-cover.jpg)](https://www.trainindata.com/p/imbalanced-data-myths-mistakes-solutions-book)
