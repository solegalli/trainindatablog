---
layout: post
title: "ADASYN: Adaptive Synthetic Sampling for Imbalanced Datasets"
author: shri
description: "Find out why you should NOT use ADASYN to handle data imbalance, what the hype was, and what to do instead to make cost-sensitive decisions."
excerpt: "Find out why you should NOT use ADASYN to handle data imbalance, what the hype was, and what to do instead to make cost-sensitive decisions."
categories: [Data Science, Imbalanced Data, Machine Learning]
image: assets/images/posts/adasyn-adaptive-synthetic-sampling-for-imbalanced-datasets/adayasn_imbalced_datasets.jpg
---

In machine learning, data imbalance is common, and depending on the nature of the data and the machine learning model, it may affect model performance. ADASYN is an oversampling method that has been proposed as a solution to the problem of imbalanced data. But does this problem really exist?

Machine learning models, such as XGBoost and LightGBM, generally perform well, including when trained with imbalanced datasets. Weaker learners like decision trees or support vector machines, on the other hand, can lead to biased models when trained on imbalanced datasets. These models will output accurate predictions for the majority class but poor ones for the minority class.

When training weak learners, oversampling the minority class has been shown to shift the decision boundary at the default classification threshold, so more minority class examples get flagged — the same trade-off you'd get by adjusting the threshold on the original data. The most popular oversampling method is SMOTE, and you can learn more about [SMOTE’s advantages and limitations](https://www.blog.trainindata.com/smote-in-python-a-guide-to-balanced-datasets/) in our previous article. Here, we will explore another oversampling technique called ADASYN.

This article will provide an insightful read on how ADASYN works. We’ll show how to implement ADASYN in Python. More importantly, we’ll discuss what we need to do **before** attempting any resampling method.

To master ADASYN these and other resampling methods, check out our book [Machine Learning with Imbalanced Data](https://www.trainindata.com/p/imbalanced-data-myths-mistakes-solutions-book).

[![Imbalanced Data: Myths, Mistakes and Modern Solutions - book by Soledad Galli]({{ site.baseurl }}/assets/images/imbalanced-data-book-cover.jpg)](https://www.trainindata.com/p/imbalanced-data-myths-mistakes-solutions-book)

**A quick note before we start:** ADASYN, like SMOTE and other resampling methods, does not make a model better at discriminating between classes. What it does is shift the model's decision boundary so that, at the default classification threshold of 0.5, we make more cost-sensitive decisions — that is, we correctly flag a larger proportion of the minority class, which is usually the class we care about the most. You can get this exact same effect by training on the original, unmodified data and simply adjusting the classification threshold afterward, without generating any synthetic samples at all. We'll come back to this point throughout the article.

## **What is ADASYN?**

**ADASYN** (Adaptive Synthetic Sampling Approach for Imbalanced Learning) was proposed by Haibo He, Yang Bai and Edwardo A. Garcia in their 2008 article titled **“Learning from Imbalanced Data”**. It is a data augmentation technique designed to address the class imbalance problem.

ADASYN is an extension of the Synthetic Minority Over-sampling Technique (SMOTE). SMOTE creates synthetic observations using all minority class examples as templates. ADASYN, instead, focuses on generating synthetic samples in areas where the minority class is hardest to learn, that is, where minority class is sparsely represented. Like this, ADASYN should shift the decision boundary further towards the hardest-to-learn minority class examples, or at least, so the theory goes.

## **How Does ADASYN Work?**

The ADASYN algorithm relies on k-nearest neighbors (k-NN) to identify regions in the feature space where the minority class is underrepresented. Below is a step-by-step outline of how ADASYN works:

### **1. Compute the Class Imbalance**

ADASYN first calculates the degree of imbalance in the dataset. This is the ratio between the number of majority and minority class samples.

The following image illustrates class imbalance, where yellow dots represent the majority class, while green and purple dots represent the minority class.

![Figure showing distribution of an imbalanced dataset]({{ site.baseurl }}/assets/images/posts/adasyn-adaptive-synthetic-sampling-for-imbalanced-datasets/adasyn_class_imbalance_image.png)

Imbalanced dataset class distribution

### **2. Identify Difficult-to-Learn Instances**

For each minority class sample, ADASYN finds the number of nearest neighbors that belong to the majority class. A higher number of closest neighbors from the majority class suggests that the instance is harder to classify correctly.

The dots circled in red are perfect examples of difficult to learn instances, because they are surrounded by a higher number of majority class instances.

![Figure showing datapoints that will be difficult to train]({{ site.baseurl }}/assets/images/posts/adasyn-adaptive-synthetic-sampling-for-imbalanced-datasets/adasyn_datapoints_example.png)

### **3. Compute Sampling Distribution**

ADASYN calculates the weight for each minority class instance based on the ratio of majority class neighbors in its local neighborhood. Specifically, for each minority sample *xi​*, the weight *wi*​ is computed as the number of majority class samples among its k-nearest neighbors, divided by k (the total number of neighbors considered).

This ratio reflects the local imbalance around *xi*​. The weights are then normalized so that they sum to 1, forming a probability distribution. Harder-to-learn instances (those with more majority class neighbors) receive higher weights, ensuring that more synthetic samples are generated in their vicinity.

In mathematical terms:

- Let *ri* be the number of majority class samples among the k-nearest neighbors of *xi*​.
- The weight *wi* is calculated as wi=ri /k​​.
- The weights are normalized as wi′=wi / ∑wj​​, where j varies from 1 to the total number of minority class samples.

This normalized weight wi′​ determines the proportion of synthetic samples to be generated for each minority instance.

### **4. Balance the Dataset**

For each minority class instance *xi​*, ADASYN generates synthetic samples based on its normalized weight *wi′*​. The number of synthetic samples to be created for *xi​* is proportional to *wi′​*. To create a new synthetic sample:

- Randomly select one of the k-nearest neighbors of *xi​*, denoted as *xzi​*.
- Compute the difference vector between *xzi​* and *xi*: diff=xzi−xi​.
- Multiply this difference vector by a random number λ between 0 and 1: .
- Add this scaled difference vector to *xi* to create the new synthetic sample: *xnew=xi+lambda diff​*

This process ensures that synthetic samples are generated along the line between *xi​* and its neighbors, focusing more on regions where minority class instances are sparse or harder to classify. The result is an adaptive oversampling technique that improves class balance by emphasizing difficult-to-learn areas.

## **Advantages of ADASYN**

**Focused Sampling**: Unlike SMOTE which generates synthetic data evenly across the minority class, ADASYN prioritizes the generation of data in regions that are more challenging for classification. This is said to help to fine-tune the decision boundary between the majority and minority classes.

**Shifts the Decision Boundary Further:** By focusing on the most difficult-to-classify regions, ADASYN shifts the decision boundary more assertively toward the minority class than plain SMOTE — the same kind of trade-off achievable by tuning the classification threshold, just with a different emphasis on which minority class examples get prioritized.

> Unsure whether SMOTE or ADASYN are the right methods for your project? Read my “[7 Takes on Working with Imbalanced Data](https://www.trainindata.com/p/7-takes-on-working-with-imbalanced-data)“, where I discuss 3 recent articles that change the conversation around resampling. It’s free.

[![7 takes on working with imbalanced data, free booklet.]({{ site.baseurl }}/assets/images/posts/should-you-use-imbalanced-learn-in-2025/MLID-booklet-presentation.png)](https://www.trainindata.com/p/7-takes-on-working-with-imbalanced-data)

## **ADASYN in Python**

This section demonstrates how to handle an imbalanced dataset, specifically the **Thyroid Sick dataset** (which can be imported from imbalanced learn), by using **ADASYN** to oversample the minority class.

### **Part 1: Importing Libraries and Data Preparation**

The first step is to import the required libraries to fetch the dataset, build classification models, and apply ADASYN.

```
# Import required libraries
import numpy as np

from imblearn.datasets import fetch_datasets
from imblearn.over_sampling import ADASYN

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, roc_auc_score, precision_recall_curve
```

Now let’s load the dataset from `imblearn.datasets`. This dataset consists of a binary classification problem, where the target (y) indicates if a patient is sick (minority class) or healthy (majority class). In this case, the target (y) has value of -1 for negative class and 1 for the positive class (that is, the minority class). We’ll convert this to the 0 and 1 binary format.

As a preprocessing step, the dataset is split into training and testing sets using `train_test_split`. Setting `stratify=y` ensures that the function preserves the proportions of each class in the target variable `y` across both the training and test sets.

```
# Load an imbalanced dataset
data = fetch_datasets()['thyroid_sick']
X, y = data.data, data.target

# Convert to binary classification: Class 1 = positive, others = negative
y = (y == 1).astype(int)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
```

Let’s check the distribution of the classes both before and after splitting the dataset.

```

# Original class distribution
unique, counts = np.unique(y, return_counts=True)
print("Original class distribution:")
for label, count in zip(unique, counts):
    print(f"Class {label}: {count} samples")

# Distribution in train/test sets
train_unique, train_counts = np.unique(y_train, return_counts=True)
test_unique, test_counts = np.unique(y_test, return_counts=True)

print("\nTrain set class distribution:")
for label, count in zip(train_unique, train_counts):
    print(f"Class {label}: {count} samples")

print("\nTest set class distribution:")
for label, count in zip(test_unique, test_counts):
    print(f"Class {label}: {count} samples")
```

The output:

`Original class distribution:
Class 0: 3541 samples
Class 1: 231 samples`

`Train set class distribution:
Class 0: 2656 samples
Class 1: 173 samples`

`Test set class distribution:
Class 0: 885 samples
Class 1: 58 samples`

We can see that the dataset is highly imbalanced, with the minority class present only in 6% of the observations in the dataset.

### **Part 2: Baseline Model on the Imbalanced Dataset**

Now, let’s train a **Random Forest classifier** using the original dataset, which is imbalanced. We’ll use the trained model to generate predictions on the test dataset we set aside. We’ll evaluate the performance on the test set by computing metrics like Recall and ROC-AUC score.

```
# Train Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions and probabilities
y_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)

# Metrics
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"ROC AUC:   {roc_auc:.4f}")
```

The output:

`Precision: 0.9273`

`Recall: 0.8793`

`ROC AUC: 0.9980`

From the output, we can see that the Recall is 87.9%. The ‘Recall’ metric measures how many of all sick patients were identified by the model, which is around 87.9%. This indicates the model is **missing about 12% of the sick cases** (false negatives).

The ROC-AUC score is a measure of the Area Under the ROC Curve, which represents how well the model can **separate “sick” from “not sick”** across all possible thresholds. It is common for imbalanced datasets to see high ROC-AUC values.

### **Part 3:** Apply ADASYN to the Training Set

Next, let’s apply the ADASYN method to balance the dataset and train another random forest model using the balanced training set to see if there is any improvement in precision and recall for the minority class. We can apply ADASYN using the function we imported from the imblearn Python library.

```
# Apply ADASYN on the training set
adasyn = ADASYN(random_state=42)
X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)

```

```
print("\nOriginal training class distribution:")
for cls, count in zip(train_unique, train_counts):
    print(f"Class {cls}: {count} samples")
```

The output:

```
Original training class distribution:
Class 0: 2656 samples
Class 1: 173 samples
```

Let’s check how the class distribution has changed after applying ADASYN in the training dataset.

```
# Show new distribution after ADASYN
unique_res, counts_res = np.unique(y_resampled, return_counts=True)
print("\nTraining class distribution after ADASYN:")
for cls, count in zip(unique_res, counts_res):
    print(f"Class {cls}: {count} samples")
```

The output:

Training class distribution after ADASYN:

`Class 0: 2656 samples`

`Class 1: 2657 samples`

We can see that the resampled dataset has almost a 50-50 class distribution. Let’s use this balanced training dataset to train a random forest model and evaluate its performance through the ROC-AUC, precision and recall metrics on the test set (which contains the original distribution, and this is key).

```

# Train Random Forest on resampled data
model = RandomForestClassifier(random_state=42)
model.fit(X_resampled, y_resampled)

# Predict and evaluate
y_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"ROC AUC:   {roc_auc:.4f}")
```

The output:

`Precision: 0.8833`

`Recall: 0.9138`

`ROC AUC: 0.9979`

The Recall has improved from 87% to 91% after balancing the dataset using ADASYN, capturing more sick patients. In the use cases of ML algorithms in healthcare, avoiding false negatives is more critical, as patients should not be denied early treatment. So, in this scenario, ADASYN has shown a positive effect.

But wait a sec…

### **Part 4:** Optimizing the Threshold instead of Oversampling

In the previous section, we saw that oversampling helped improve Recall on the imbalanced dataset. But, is oversampling always essential?

To evaluate machine learning model performance, we often use threshold-dependent metrics. By default, scikit-learn uses 0.5 to determine which observations belong to the minority class. However, 0.5 is hardly a good threshold when working with imbalanced data and can lead to a bad assessment of the performance of the model.

In this section, let’s tune the classification threshold to better evaluate our model’s performance on imbalanced data. Instead of relying on the default threshold of 0.5, let’s evaluate multiple thresholds by calculating precision and recall at each probability cut-point using `precision_recall_curve()` function from scikit-learn. We can use the predicted probabilities from the baseline model to pass as input to the precision_recall_curve().

```
# Get predicted probabilities from the original imbalanced model
y_scores = model.predict_proba(X_test)[:, 1]  # model is from Section 1

# Use precision-recall curve to find best threshold
precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)

# Calculate F1 scores and get threshold that gives max F1
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

# Apply new threshold
y_pred_tuned = (y_scores >= best_threshold).astype(int)

# Recalculate precision and recall
precision_tuned = precision_score(y_test, y_pred_tuned)
recall_tuned = recall_score(y_test, y_pred_tuned)
f1_tuned = f1_scores[best_idx]

print(f"\nBest threshold by F1: {best_threshold:.4f}")
print(f"Precision at best threshold: {precision_tuned:.4f}")
print(f"Recall at best threshold:    {recall_tuned:.4f}")
print(f"F1-Score at best threshold:  {f1_tuned:.4f}")
```

The output:

`Best Threshold by F1: 0.4500`

`Precision at Best Threshold: 0.9153`

`Recall at Best Threshold: 0.9310`

`F1-Score at Best Threshold: 0.9231`

The best threshold identified here is 0.45, at which both precision and recall are balanced. The Recall is 93%, which is better than 91% we got by applying the oversampling technique while also not reducing precision a lot.

Hence, instead of applying techniques like SMOTE or ADASYN to balance the dataset, adjusting the decision threshold is a simpler step that almost always leads to better results. This adjustment makes the model more effective without the added complexity of resampling.

### **Part 5:** Automatic Threshold Tuning using Sklearn

Instead of manually tuning the decision threshold, as we did in the previous section, we can also use the `TunedThresholdClassifierCV` from `scikit-learn` to automate the process. This tool wraps any base classifier and tunes the threshold during cross-validation to maximize a selected scoring metric, such as Recall or F1-score.

In this section, let’s train a random forest on the original imbalanced dataset and use the `TunedThresholdClassifierCV` to find the optimal threshold.

```
from sklearn.model_selection import TunedThresholdClassifierCV

# Initialize the base classifier
base_model = RandomForestClassifier(random_state=42)

# Wrap with TunedThresholdClassifierCV to auto-tune threshold using F1-score
tuned_model = TunedThresholdClassifierCV(estimator=base_model, scoring='f1', cv=5)

# Fit on original (imbalanced) training data
tuned_model.fit(X_train, y_train)

# Predict on test set
y_pred_auto = tuned_model.predict(X_test)                  # Label predictions (threshold tuned)
y_proba_auto = tuned_model.predict_proba(X_test)[:, 1]     # Probability predictions

# Evaluate metrics
precision_auto = precision_score(y_test, y_pred_auto)
recall_auto = recall_score(y_test, y_pred_auto)
roc_auc_auto = roc_auc_score(y_test, y_proba_auto)

# Print results
print(f"Precision (auto): {precision_auto:.4f}")
print(f"Recall (auto):    {recall_auto:.4f}")
print(f"ROC AUC:          {roc_auc_auto:.4f}")
```

```
Precision (auto): 0.8852
 Recall (auto):    0.9310
 ROC AUC:          0.9980
```

We can observe that recall is 93%, the same or better than the results from manual threshold tuning and oversampling methods. Without applying any oversampling like ADASYN, the model has achieved a well-balanced trade-off between precision and recall. Hence, fine-tuning thresholds can be a simpler way to achieve better results without oversampling whenever possible.

### **ADASYN vs SMOTE**

While ADASYN is closely related to SMOTE, let’s see what the differences are between them:

- **Adaptive Nature**: SMOTE generates synthetic samples uniformly, which can lead to over-representation of certain regions in the feature space. ADASYN, on the other hand, adaptively generates more samples in regions where the minority class is underrepresented, leading to a better learning process.
- **Focus on Minority-Class Complexity**: ADASYN emphasizes generating samples where the minority class is hardest to classify, making it particularly effective when the minority class has complex patterns or overlaps significantly with the majority class. For more information, you can check this article on [SMOTE](https://www.blog.trainindata.com/smote-in-python-a-guide-to-balanced-datasets/).

### **Challenges and Limitations**

Despite its advantages, ADASYN has some limitations:

- **Synthetic Data Quality**: The synthetic samples generated may not always capture the true underlying distribution of the minority class, especially if the minority class itself is not well-represented or contains noise.
- **Risk of Overfitting**: While ADASYN mitigates overfitting compared to traditional oversampling methods, there is still a risk that the model may overfit to the synthetic data, especially if too many samples are generated in a small region of the feature space.
- **Computational Complexity**: ADASYN requires calculating k-nearest neighbors, which can be computationally expensive for large datasets. This can become a bottleneck when dealing with high-dimensional data.

## **Conclusion**

ADASYN was introduced as a way of dealing with imbalanced datasets, especially when the minority class is complex and difficult to learn, by focusing synthetic samples on the most challenging areas of the minority class distribution. But as we saw in Parts 4 and 5, that shift in the decision boundary is not unique to ADASYN — the same trade-off between recall and precision can be reached by tuning the classification threshold on a model trained on the original data, without generating any synthetic samples at all.

So our recommendation is to try threshold tuning first. It's simpler, faster, and doesn't risk introducing unrealistic synthetic data. Only reach for ADASYN, SMOTE, or other resampling methods if threshold tuning genuinely isn't enough for your use case — and even then, test whether it actually helps on your specific dataset and model rather than assuming it will.

## **Additional Resources**

To learn more about Adasyn, check the paper published in the 2008 IEEE International Joint Conference on Neural Networks (IEEE World Congress on Computational Intelligence), [here](https://ieeexplore.ieee.org/document/4633969).

To learn more about Working with Imbalanced Data, what modeling techniques are used in the industry to solve data imbalance, and much more, check out our book [Machine Learning with Imbalanced Data](https://www.trainindata.com/p/imbalanced-data-myths-mistakes-solutions-book).
