---
layout: post
title: "SMOTE in Python and whether you should still use it in 2026"
author: sole
description: "Learn how to implement SMOTE in Python and whether you should still use it to work with imbalanced datasets in 2026."
excerpt: "Learn how to implement SMOTE in Python and whether you should still use it to work with imbalanced datasets in 2026."
categories: [Imbalanced Data]
image: assets/images/posts/smote-in-python-a-guide-to-balanced-datasets/blog_banner.png
---

SMOTE (Synthetic Minority Over-sampling Technique) is often presented as a powerful tool for handling imbalanced data in machine learning. In this article, I’ll challenge that reputation and show why using SMOTE in machine learning pipelines may offer less benefit than you’ve been led to believe.

To give you a bit of context, SMOTE emerged when tree-based ensemble methods were still gaining traction. Since then, machine learning has changed considerably, with gradient boosting becoming a popular choice for tabular data. These models can often distinguish between classes effectively without synthetic oversampling. Yet the recommendation to use SMOTE has persisted, even when its benefits for the model and dataset at hand have not been established.

Let's debunk the myth.

## Imbalanced Data

In most real-world scenarios, data is imbalanced, meaning that one class (the majority class) has many more samples than the other one (the minority class). Although you will read a lot that class imbalance makes it difficult for algorithms to classify the classes correctly, that is not necessarily the case. In fact, if the classes are well separated, that is, there is a clear separation boundary among them, the algorithms will work just fine. But when the class separability is not that clear, then things start getting difficult.

> Check out my book [Imbalanced Data: Myths, Mistakes and Modern Solutions](https://www.trainindata.com/p/imbalanced-data-myths-mistakes-solutions-book), to learn how to tackle imbalanced data in the modern world.

[![Imbalanced Data: Myths, Mistakes and Modern Solutions - book by Soledad Galli]({{ site.baseurl }}/assets/images/imbalanced-data-book-cover.jpg)](https://www.trainindata.com/p/imbalanced-data-myths-mistakes-solutions-book)


## Is a Balanced Dataset Important?

Class imbalance isn't a problem. How you handle it is. The problems attributed to class imbalance are normally related to one of three issues: insufficient data, class overlap, or the use of a wrong model or wrong evaluation metric.

More than 20 years ago, some machine learning models like decision trees, support vector machines and k-nearest neighbors, might have struggled to discriminate among classes when the datasets were imbalanced.

Today, we use more powerful machine learning models, like gradient boosting machines, including XGBoost and LightGBM. These models tend to work equally well in balanced and imbalanced datasets. So, when training these models, there isn’t a need to balance the data.

When training weaker classifiers, if the classes are not well separated, increasing the number of samples of the minority class, might help the model find proper boundaries and increase its performance. SMOTE, in fact, was designed and tested using weaker classifiers.

## SMOTE in Machine Learning

SMOTE, which stands for Synthetic Minority Over-sampling Technique, was designed to increase the representation of the minority class in an imbalanced dataset. That makes SMOTE an oversampling method.

SMOTE generates synthetic samples for the minority class to balance the dataset, so that we have an equal number of majority and minority class samples. In other words, SMOTE creates synthetic, that is, artificial new data points.

SMOTE does that by interpolating between existing minority class examples. That is, SMOTE creates new data points in between 2 samples of the minority class.

> **A quick note before we start:** SMOTE, like all other undersampling or oversampling methods, does not make a model better at discriminating between classes. What it does is shift the model's decision boundary so that, at the default classification threshold of 0.5, we make cost-sensitive decisions. That is, we correctly flag a larger proportion of the minority class, which is usually the class we care about the most. Note, however, that you can achieve this same effect by training the model on the original, unmodified data and simply adjusting the classification threshold afterward, without generating any synthetic samples at all. We'll come back to this point throughout the article.

Now let's see how SMOTE actually works.

### Step 1: Finding the Nearest Neighbors

SMOTE looks only at the minority class. It first selects a minority class data point and then finds its nearest neighbors. These neighbors are selected from within the minority class as well. For each data point in the minority class:

- SMOTE looks at the features (attributes) of the current minority class sample.
- It finds the k-nearest neighbors of this sample among other minority class samples.

Here, k is the number of neighbors selected. Typically, SMOTE uses the Euclidean distance to measure similarity between samples, but other distance metrics can also be used.

**Example:** Suppose you have a dataset with features representing different characteristics of people (like age, income, etc.). If you’re trying to classify rare cases of a disease (minority class), SMOTE will first identify a few people in the minority class who have similar features.

![Figure showing nearest neighbors of minority class sample in the dataset ]({{ site.baseurl }}/assets/images/posts/smote-in-python-a-guide-to-balanced-datasets/Finding_nearest_neighbors-1024x644.png)

### Step 2: Generating Synthetic Samples

After finding the k-nearest neighbors, SMOTE randomly selects one neighbor from this group. It then creates a new synthetic sample by interpolating between the current sample and the selected neighbor. This means generating a new data point that lies somewhere between the two existing data points in the feature space.

![Figure showing generation of a new synthetic sample by using SMOTE]({{ site.baseurl }}/assets/images/posts/smote-in-python-a-guide-to-balanced-datasets/New_sample_generation-1024x644.png)

The interpolation is done using the following formula:

![interpolation formula used to generate new data with SMOTE]({{ site.baseurl }}/assets/images/posts/smote-in-python-a-guide-to-balanced-datasets/interpolation_formula.png)

Where:

- x_original is the feature vector of the current minority class sample.
- x_neighbor is the feature vector of one of its nearest neighbors.
- λ is a random number between 0 and 1, so the new point will lie somewhere between the two samples.

**Example:**

Let’s say you have two examples in the minority class:

- Example 1: (Age: 25, Income: $30,000)
- Example 2: (Age: 30, Income: $35,000)

SMOTE might generate a synthetic example by taking a random value between these two points. If λ=0.5, the new synthetic sample will be:

- (Age: 27.5, Income: $32,500)


### Step 3: Repeat Until the Dataset Is Balanced

SMOTE repeats the process explained in the previous step for each sample in the minority class, creating synthetic samples until the minority class has the same number of samples as the majority class (or reaches a predefined ratio).

### Example of How SMOTE Balances a Dataset

**Original Dataset:**

- Majority Class: 100 samples
- Minority Class: 20 samples

If we apply SMOTE with a sampling ratio of 1:1, the algorithm will generate 80 synthetic minority samples, so the new dataset will have:

- Majority Class: 100 samples
- Minority Class: 100 samples (20 original + 80 synthetic)

Now we have a balanced dataset.

## Implementing SMOTE in Python

Let’s take a look at how we can implement the SMOTE algorithm in Python. We will first train a classifier on the imbalanced dataset to have the baseline performance, and then train it on a balanced dataset created by applying SMOTE. We will then compare the performance of both models and draw conclusions about SMOTE.

> For this demo, we will use SMOTE for binary classification, though it can also be applied to multi-class problems.

### Load the Dataset

We’ll use the [credit card fraud dataset](https://www.openml.org/search?type=data&id=1597) from OpenML, originally published on [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download). It consists of numerical features representing credit card transactions. This dataset is highly imbalanced, with the majority of transactions being legitimate and only a small fraction being fraudulent. For this binary classification problem we have:

- The majority class (label 0) represents legitimate transactions.
- The minority class (label 1) represents fraudulent transactions.

Let’s go ahead and load the dataset:

```
import pandas as pd
from sklearn.datasets import fetch_openml

# Load the dataset as pandas dataframe
df = fetch_openml(name='creditcard', version=1, as_frame=True).frame
df['Class'] = df['Class'].astype(int)

# Display the class distribution
print(df['Class'].value_counts())
```

In the following image we see the number of observations for each class:

![Figure showing imbalanced distribution of data points among both classes ]({{ site.baseurl }}/assets/images/posts/smote-in-python-a-guide-to-balanced-datasets/imbalanced_data_distribution.png)

### Data Preprocessing

Let's split the data into features (X) and target (y), and after that, into training set and test set. To do this, we will utilize scikit-learn:

```
from sklearn.model_selection import train_test_split

# Separate features (X) and target (y)
X = df.drop('Class', axis=1)
y = df['Class']

# Split the dataset into training and testing sets
# (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
```

### Train a Classifier on the Imbalanced Dataset

Now, let's train a classification model on the imbalanced dataset and evaluate its performance. We’ll use a Random Forest classifier:

```
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Train a Random Forest classifier on the imbalanced dataset
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
print("Classification Report (Before SMOTE):")
print(classification_report(y_test, y_pred))
```

The following figure shows the classification report of the random forest classifier trained with the imbalanced dataset. For the majority class, the precision, recall and the F1 score is 1.00. However, for the minority class, we see the recall is 0.76.

![Classification report of the trained model before applying smote on the dataset ]({{ site.baseurl }}/assets/images/posts/smote-in-python-a-guide-to-balanced-datasets/classificationrep_before_smote.png)

We see that the model performs better on the majority class compared to the minority class. Now, we will move on to applying SMOTE to our imbalanced dataset to create more examples of the fraudulent transactions, and see if that improves the model’s performance.

### Apply SMOTE to Balance the Dataset

We will use the Python open-source library [imbalanced-learn](https://imbalanced-learn.org/stable/) to apply SMOTE. With SMOTE, we aim to create synthetic samples for the minority class and balance the dataset.

With imbalanced-learn, we have the flexibility to adjust the number of minority class samples that we want to create, by modifying the **sampling_strategy** parameter, which specifies the desired ratio of the minority class relative to the majority class. It can have a float value from 0 to 1.

For example, a value of 0.5 means the minority class will have half as many samples as the majority class after resampling. By default its value is set to `"auto"`, which means that the minority class will have the same number of samples as the majority class after the oversampling.

```
from imblearn.over_sampling import SMOTE

# Apply SMOTE to the training data
smote = SMOTE( sampling_strategy = "auto", random_state=42)
X_resampled, y_resampled = smote.fit_resample(
    X_train, y_train)

# Check the class distribution after applying SMOTE
print(pd.Series(y_resampled).value_counts())
```

In the following figure, we can see that both classes have an equal number of samples, i.e. 199020, after applying SMOTE.

![Figure showing an equal number of samples for both classes ]({{ site.baseurl }}/assets/images/posts/smote-in-python-a-guide-to-balanced-datasets/balanced_data_distribution.png)

### Train the Classifier on the SMOTE-Augmented Dataset

We’ll train a Random Forest classifier using the SMOTE-balanced training data so that we can analyze the difference.

```
# Train the classifier on the SMOTE-balanced dataset
clf_smote = RandomForestClassifier(random_state=42)
clf_smote.fit(X_resampled, y_resampled)

# Predict on the test set
y_pred_smote = clf_smote.predict(X_test)

# Evaluate the model
print("Classification Report (After SMOTE):")
print(
    classification_report(y_test, y_pred_smote))
```

The following figure shows the classification report of the model trained with the balanced dataset. We can see that the recall for the minority class has improved from 0.76 to 0.80.

![Classification report of the trained model after applying smote on the dataset ]({{ site.baseurl }}/assets/images/posts/smote-in-python-a-guide-to-balanced-datasets/classificationrep_after_smote.png)

### SMOTE Worked! Right?

Do you notice something wrong with how I evaluated the effectiveness of SMOTE?

Precision, recall, and F1-score are metrics that depend on the threshold, and here I am using the default classification threshold of 0.5, which is not suitable for imbalanced datasets. Not at least, if we care most about the minority class. In fact, if we changed the threshold, we’d see different values for these metrics.

Since SMOTE generates synthetic samples, it might seem like it improves the model’s performance. But if we don’t carefully choose the right threshold for classifying the minority class, the perceived improvements could be exaggerated.

Instead of relying on the default threshold (0.5), we can adjust the threshold based on the specific use case. For example, if missing a minority class prediction is very costly like in our fraud detection example, we may want to lower the threshold to capture more instances of the minority class, and that would increase the value of recall (at the expense of precision, of course). That is exactly the trade-off we saw with SMOTE as well.

For a thorough discussion on the value of SMOTE, check out our [YouTube video](https://www.youtube.com/watch?v=blcOOheXNoQ) (and **subscribe** to stay up to date!):

<div class="video-embed"><iframe src="https://www.youtube.com/embed/blcOOheXNoQ?feature=oembed" title="Working with Imbalanced Data in 2024 - Machine Learning with Imbalanced Data" width="560" height="315" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe></div>

If you prefer reading, then check out our [imbalanced datasets](https://www.blog.trainindata.com/class-imbalance-in-machine-learning/) article.

### Comparing the Classification Reports

Here is the comparison between the classification reports, before and after SMOTE, at the default threshold of 0.5:

**Before SMOTE:**

- Precision (Class 1, Fraudulent): 0.96
- Recall (Class 1, Fraudulent): 0.76
- F1-Score (Class 1, Fraudulent): 0.85

In this case, while precision is high, the recall is lower (0.76), indicating that the model struggles to identify a large portion of actual fraudulent transactions. This can be problematic in scenarios like fraud detection, where missing fraudulent cases is costly.

**After SMOTE:**

- Precision (Class 1, Fraudulent): 0.87
- Recall (Class 1, Fraudulent): 0.80
- F1-Score (Class 1, Fraudulent): 0.83

Comparing the two reports directly: recall rose four percentage points, from 0.76 to 0.80, precision fell nine percentage points, from 0.96 to 0.87, and F1 fell two percentage points, from 0.85 to 0.83. Basically, there was a trade-off between precision and recall: SMOTE shifted the decision boundary so that more of the minority class gets flagged, catching more fraud (higher recall) at the cost of more false positives (lower precision).

This is how the success of SMOTE has often been described: a model that appears to get better at catching fraud. But the gain in recall came paired with a drop in precision, exactly the kind of trade-off you would expect from moving the decision threshold, not from a model that got better at telling fraud apart from legitimate transactions. Let's check that directly: can we get the same trade-off by training on the original data and simply adjusting the threshold?

### Adjusting the Threshold Instead of Resampling

Let's put that claim to the test. Scikit-learn has a class built exactly for this, `TunedThresholdClassifierCV`, which searches for the decision threshold that optimizes a chosen metric.

We take `clf`, the Random Forest we already trained on the original, imbalanced data, and let `TunedThresholdClassifierCV` find the threshold that maximizes the F1 score. It uses cross-validation internally to choose that threshold, so we are not tuning it on the same data we then evaluate on:

```
from sklearn.model_selection import TunedThresholdClassifierCV

tuned_clf = TunedThresholdClassifierCV(clf, scoring='f1')
tuned_clf.fit(X_train, y_train)

print("Best threshold:", tuned_clf.best_threshold_)

y_pred_tuned = tuned_clf.predict(X_test)
print("Classification Report (Original data, tuned threshold):")
print(classification_report(y_test, y_pred_tuned))
```

This is what we get:

```
Best threshold: 0.4444444444444445
Classification Report (Original data, tuned threshold):
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     85295
           1       0.95      0.78      0.86       148

    accuracy                           1.00     85443
   macro avg       0.98      0.89      0.93     85443
weighted avg       1.00      1.00      1.00     85443
```

The best threshold, 0.444, lower than 0.5. For the minority class, precision, recall and F1 are 0.95, 0.78 and 0.86. Compare that to SMOTE's 0.87, 0.80 and 0.83. Adjusting the threshold on the original model gets almost the same recall as SMOTE, 0.78 versus 0.80. If we take the threshold a little lower, we'll match the precision obtained by SMOTE (try it out!).

In other words, we did not need any synthetic data at all. Training the model once on the original, imbalanced data, and then choosing a better threshold, got us a comparable, and on this dataset an even better, trade-off between precision and recall than SMOTE did.

### Overall Impact

- **More Fraud Caught at the Default Threshold:** The model flags more fraudulent transactions after SMOTE, because the decision boundary shifted toward the minority class, not because the model got better at telling fraud apart from legitimate transactions.
- **Slight Trade-off:** There’s a small increase in false positives (legitimate transactions incorrectly flagged as fraud), but this is often acceptable in scenarios like fraud detection, where catching fraud is critical.

As we just showed directly, the classification report is threshold dependent, so we could achieve the same effect shown with SMOTE, and here an even better one, simply by changing the classification threshold used on the model trained on the original, imbalanced dataset.

### Does SMOTE Improve Machine Learning Model Performance?

A better way to evaluate if SMOTE improved the performance of a machine learning model, is to use a metric that directly assesses a model's discrimination ability, like the ROC curve and ROC-AUC.

Let's check then whether SMOTE improves ROC-AUC on 3 different, real-world imbalanced datasets: `ecoli`, `thyroid_sick`, and `arrhythmia`.

This time, we'll also quantify the uncertainty, so that we can tell whether any difference we see is likely real or just noise. We'll use cross-validation to get the ROC-AUC and its standard deviation on the training set, and bootstrapping to get the ROC-AUC and its standard deviation on the test set. We'll keep the bootstrapping to just 5 samples, to keep the demo quick.

Let's make the imports:

```
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
from sklearn.pipeline import Pipeline

from imblearn.datasets import fetch_datasets
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
```

SMOTE, and the scaler that it needs (since it relies on distances between neighbors), must only be fit on the training data used within each cross-validation fold. So we combine both into a pipeline, which fits and resamples only during training:

```
datasets_ls = ['ecoli', 'thyroid_sick', 'arrhythmia']

def bootstrap_roc_auc(model, X_test, y_test, n_boots=5):
    scores = []
    for i in range(n_boots):
        X_bs, y_bs = resample(X_test, y_test, random_state=i)
        pred = model.predict_proba(X_bs)[:, 1]
        scores.append(roc_auc_score(y_bs, pred))
    return np.mean(scores), np.std(scores)
```

Now, let's loop over the datasets, and for each, obtain the ROC-AUC and its standard error with 5-fold cross-validation on the train set, and with 5 bootstrap samples on the test set:

```
for dataset in datasets_ls:

    data = fetch_datasets()[dataset]
    # stratify: these datasets have very few minority samples, so an
    # unstratified split can easily leave an unrepresentative test set
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, 
        test_size=0.3, random_state=0, 
        stratify=data.target
    )

    baseline_pipe = Pipeline([
        ('scaler', MinMaxScaler()),
        ('rf', RandomForestClassifier(n_estimators=100, 
            random_state=39, max_depth=2, n_jobs=4)),
    ])
    smote_pipe = ImbPipeline([
        ('scaler', MinMaxScaler()),
        ('smote', SMOTE(random_state=0)),
        ('rf', RandomForestClassifier(n_estimators=100, 
            random_state=39, max_depth=2, n_jobs=4)),
    ])

    cv_baseline = cross_val_score(baseline_pipe, X_train, y_train, cv=5, scoring='roc_auc')
    cv_smote = cross_val_score(smote_pipe, X_train, y_train, cv=5, scoring='roc_auc')

    baseline_pipe.fit(X_train, y_train)
    smote_pipe.fit(X_train, y_train)

    test_baseline_mean, test_baseline_se = bootstrap_roc_auc(baseline_pipe, X_test, y_test)
    test_smote_mean, test_smote_se = bootstrap_roc_auc(smote_pipe, X_test, y_test)

    print(dataset)
    print(f"CV ROC-AUC (baseline): {cv_baseline.mean():.4f} +/- {cv_baseline.std()/np.sqrt(5):.4f}")
    print(f"CV ROC-AUC (SMOTE):    {cv_smote.mean():.4f} +/- {cv_smote.std()/np.sqrt(5):.4f}")
    print(f"Test ROC-AUC (baseline): {test_baseline_mean:.4f} +/- {test_baseline_se:.4f}")
    print(f"Test ROC-AUC (SMOTE):    {test_smote_mean:.4f} +/- {test_smote_se:.4f}")
    print()
```

This is what we get:

```
ecoli
CV ROC-AUC (baseline): 0.9135 +/- 0.0160
CV ROC-AUC (SMOTE):    0.9207 +/- 0.0253
Test ROC-AUC (baseline): 0.9442 +/- 0.0215
Test ROC-AUC (SMOTE):    0.9480 +/- 0.0243

thyroid_sick
CV ROC-AUC (baseline): 0.9507 +/- 0.0221
CV ROC-AUC (SMOTE):    0.9312 +/- 0.0221
Test ROC-AUC (baseline): 0.9612 +/- 0.0097
Test ROC-AUC (SMOTE):    0.9310 +/- 0.0142

arrhythmia
CV ROC-AUC (baseline): 0.8923 +/- 0.0369
CV ROC-AUC (SMOTE):    0.8619 +/- 0.0342
Test ROC-AUC (baseline): 0.9274 +/- 0.0226
Test ROC-AUC (SMOTE):    0.8426 +/- 0.0894
```

On `ecoli`, cross-validation and the test set agree: there’s no meaningful difference between the baseline and SMOTE, in either case (0.914 vs 0.921 in cross-validation, 0.944 vs 0.948 on the test set, well within one standard error of each other both times). On `thyroid_sick`, both evaluations point the same way, with the baseline scoring somewhat higher than SMOTE, though the gap is modest relative to the standard error in both cases. On `arrhythmia`, cross-validation again shows no clear difference, and while the test set numbers still favor the baseline by a wide margin, the standard error on the SMOTE estimate is now large enough (driven by there being only 8 minority examples in that test set) that we can't call this a significant difference either.

In other words, once we evaluate this properly, we find no convincing evidence that SMOTE improves ROC-AUC on any of these 3 datasets.


## Bottom Line

- When working with imbalanced datasets, use strong classifiers like XGBoost and LightGBM.
- Always adjust the probability threshold used to classify an observation as a member of the minority class.
- Always determine the metrics with their standard deviation to corroborate that the differences you see are significant.

## What SMOTE Really Does in Machine Learning

1. **Shifts the Decision Boundary Toward the Minority Class:** By balancing the classes, the prediction model flags more of the minority class at the default threshold, the same trade-off you'd get by adjusting the classification threshold on the original, imbalanced data.
2. **Avoids Data Duplication:** Unlike random oversampling, where we duplicate existing minority samples, SMOTE generates synthetic samples. This helps avoid overfitting because the new samples are not exact copies of the original samples.
3. **Might Improve Discrimination Ability of Weak Classifiers:** SMOTE can be combined with some machine learning algorithms (such as decision trees or SVM) and has been shown to have a bigger effect on threshold-dependent metrics for these weaker models than for strong classifiers like XGBoost.

## Limitations of SMOTE in Machine Learning

1. **Synthetic Samples May Not Always Be Meaningful:** SMOTE generates new data points based on linear interpolation between existing points. If the feature space has complex relationships, the synthetic samples might not represent realistic or meaningful data.
2. **Overlapping Classes:** If the boundary between classes is unclear (i.e., if some minority class points are too close to the majority class points), SMOTE might generate synthetic samples that actually belong to the majority class region, leading to misclassification. To address this issue we can use variations of SMOTE, such as Borderline SMOTE or ADASYN, although only if you are using weaker classifiers.
3. **Not Suitable for Categorical Data:** SMOTE works by interpolating numerical features. It doesn’t handle categorical data well. Variations of SMOTE, such as SMOTE-NC (for Nominal and Continuous data), have been developed to address this issue.
4. **Computational Cost:** SMOTE relies on the k-nearest neighbors algorithm to find neighbors for each minority sample, which doesn't scale well to large datasets and can become slow when there are many minority observations.
5. **Sensitivity to the Choice of k:** The number of neighbors (`k_neighbors`) used to generate synthetic samples can noticeably affect the quality of the synthetic data. Tuning it with cross-validation adds to the computational cost, since it means running the nearest-neighbors search repeatedly.


## Other Ways to Work with Imbalanced Datasets

Among data resampling methods, we discussed SMOTE in machine learning, which is an oversampling method. The alternative is to use [undersampling](https://www.blog.trainindata.com/undersampling-techniques-for-imbalanced-data/) methods to remove excessive number of majority examples. However, [undersampling does not improve model performance](https://tech.trainindata.com/posts/undersampling-does-not-improve-model-performance), contrary to popular belief.

To avoid rebalancing, we could use [cost-sensitive learning](https://www.blog.trainindata.com/cost-sensitive-learning-for-imbalanced-data/). But be aware that [class weights are also designed to shift the decision boundary](https://tech.trainindata.com/posts/class-weights-do-not-improve-model-performance), rather than improving model performance.

## Citations

- N. V. Chawla, K. W. Bowyer, L. O. Hall, W. P. Kegelmeyer, “[SMOTE: synthetic minority over-sampling technique](https://arxiv.org/abs/1106.1813),” Journal of artificial intelligence research, 321-357, 2002.
- Y. Elor, H. Averbuch-Elor, “[To SMOTE, or not to SMOTE?](https://arxiv.org/html/2201.08528v3)” arXiv:2201.08528, 2022.
- Galli. 2026. [Undersampling experiments Code Repository](https://github.com/solegalli/resampling-experiments).
- Galli. 2026 [Imbalanced Data: Myths, Mistakes and Modern Solutions](https://www.trainindata.com/p/imbalanced-data-myths-mistakes-solutions-book).
- H. Han, W. Wen-Yuan, M. Bing-Huan, “Borderline-SMOTE: a new over-sampling method in imbalanced data sets learning,” Advances in intelligent computing, 878-887, 2005.
- He, Haibo, Yang Bai, Edwardo A. Garcia, and Shutao Li. “ADASYN: Adaptive synthetic sampling approach for imbalanced learning,” In IEEE International Joint Conference on Neural Networks (IEEE World Congress on Computational Intelligence), pp. 1322-1328, 2008.
