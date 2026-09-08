---
layout: post
title: "Class Imbalance in Machine Learning"
author: gurjinder
description: "Contrary to what you'll read online or get from ChatGPT, class imbalance is NOT the problem. How you handle it is."
excerpt: "Contrary to what you'll read online or get from ChatGPT, class imbalance is NOT the problem. How you handle it is."
categories: [Imbalanced Data, Machine Learning]
image: assets/images/posts/class-imbalance-in-machine-learning/class-imbalance-overview-blog.png
---

Class imbalance isn’t a problem—how you handle it is. Contrary to popular belief, imbalanced data does not inherently harm model performance—poor methodology does.

SMOTE, once a go-to solution, is frequently misapplied, introducing bias rather than solving it. Similarly, the use of a default 0.5 probability threshold to calculate evaluation metrics persists despite its misalignment with real-world class distribution.

In this article, I challenge outdated practices and provide rigorous alternatives.

> To find the latest discussions and tools to work with imbalanced data, check out our course [Machine Learning with Imbalanced Data](https://www.trainindata.com/p/machine-learning-with-imbalanced-data).

## Class imbalance is not the problem

When the classes in a dataset are not uniformly distributed, it is known as class imbalance. Considering a simple case of a binary classification problem, class imbalance occurs when the number of instances representing a “rare” class (also termed a **minority class**) is far less than that representing the other class (i.e. **majority class**).

Datasets with uneven class distributions are referred to as **imbalanced datasets**. Imbalanced classes also occur in multi-class classification.

Data imbalance can make it difficult for the machine learning models to generalize well. No, **that’s not true**. You’ll read that all around the web, but what makes classification a challenge has seldom to do with class imbalance.

The key factors affecting classification accuracy include:

– The size of the dataset
– The class separability
– The machine learning model

### Data size

As you probably know, model performance typically improves with more data, but only up to a point, beyond which additional data is unlikely to result in an increase in performance.

With imbalanced datasets, that is also the case: if a machine learning model is not exposed to enough examples of the minority class during training, it may develop a bias towards the majority class. As a result, the model may perform well for the majority class but poorly for the minority class. This can have alarming consequences especially when the stakes associated with miss-classifying a minority class observation are high.

### Class Separability

The separability of classes significantly influences a classifier’s ability to accurately predict the minority class. If the classes are well separated, most classifiers will be able to correctly predict the classes of most instances.

### The machine learning model

Using the right algorithm is key to correctly identify the decision boundaries between classes. Using linear models when the boundaries are not linear, will lead to wrong conclusions. Similarly, using simple models, like decision trees, to separate complex boundaries will also lead to wrong classification.

This might sound trivial, but most over- and under-sampling techniques, were introduced and discussed in the context of simpler machine learning models, like linear models, or decision trees. In fact, in a 2022 article, it’s shown that over-sampling does not really improve the performance of stronger classifiers like gradient boosting machines.

![To SMOTE or Not to SMOTE]({{ site.baseurl }}/assets/images/posts/should-you-use-imbalanced-learn-in-2025/to-smote-or-not-to-smote-paper.png)

> Check out our views on this and 2 other recent articles that change the discussion around resampling for imbalanced datasets, in our free booklet “[7 Takes on Working with Imbalanced Data](https://www.trainindata.com/p/7-takes-on-working-with-imbalanced-data)“.

## Class Imbalance in the Real World

Some examples of real-world scenarios where class imbalance is prominent include:

- **Fraud detection**, where fraudulent transactions are far less common than legitimate ones,
- **Disease diagnosis**, where certain diseases are rare compared to the overall population, e.g. cancer prediction.
- **Customer churn** prediction, where the majority of customers are not likely to churn.
- **Spam detection**, where the ratio of spam emails to regular emails is generally low, etc.

In the above examples, the frequency of the **minority class,** which is most often the focus of our work, is much smaller than that of the **majority class**. In these scenarios, failing to accurately identify the sensitive, minority class (e.g., fraudulent transactions or rare diseases) can have devastating consequences.

## Challenges with Imbalanced Datasets

To understand the challenges posed by using imblanced datasets, let’s walk through an example related to credit card fraud detection. We’ll use a dataset that can be downloaded from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data).

We’ll start by training a machine learning model without data preprocessing to handle the class imbalance.

Let’s set up a Python notebook and import the libraries.

```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from imblearn.over_sampling import SMOTE
```

Before training our classifier, let’s do some basic data analysis:

```
df = pd.read_csv('creditcard.csv')
print(df.shape)
print(df.columns)
```

In the following output, we see the size of the dataset and the name of the variables:

```
(284807, 31)

Index(['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount',
'Class'],       dtype='object')
```

The dataset consists of 284807 rows and 31 columns, meaning there are 284807 data points and 31 features (out of which 28 features i.e., V1 to V28 are anonymized). The ‘Class’ variable is the target feature that specifies whether or not the given data point (presumably a credit card transaction) is either fraud (denoted as class value ‘1’) or non-fraud (denoted as class value ‘0’).

Let’s look at the distribution of the ‘Class’ variable.

```
df['Class'].value_counts()
```

In the following output we see that the non-fraudulent transactions massively outnumber the fraudulent ones:

```
Class
0    284315
1       492
Name: count, dtype: int64
```

Let’s do a plot instead:

```
bars = plt.bar(df['Class'].unique(), df['Class'].value_counts())

# Add text on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{int(height)}', ha='center', va='bottom')

# Customize the rest of the chart
plt.xticks([0, 1])
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.title('Frequency of each Class')
plt.show()
```

In the following bar plot we clearly see the class imbalance:

![Bar chart showing the class distribution of the dataset]({{ site.baseurl }}/assets/images/posts/class-imbalance-in-machine-learning/class_dist_plot-1024x778.png)

It is evident from the above chart that this dataset is suffering from a class imbalance problem, with only 492 data points representing the ‘Class = 1’ (fraud) which is just 0.001% of the total data points. Therefore, we call class ‘1’ the minority class and class ‘0’ the majority class.

### Creating a fraud detection model

The credit card company is interested in identifying whether a transaction is a regular (non-fraud) transaction or a fraudulent transaction based on various characteristics. Hence, its data science team builds a classification system that would alert the business as soon as it detects a potentially fraudulent transaction.

Classifying a fraudulent transaction as a safe one (a “false negative”) can have significant financial consequences for the credit card company. By treating a fraudulent transaction as regular, the company may suffer financial losses, which is highly undesirable.

Given this context and the business objective, the goal of the data science team is to implement a fraud detection system that minimizes false negatives to ensure no fraudulent transaction goes unattended.

As we now have a basic idea of the problem, let’s start implementing a simple machine learning model for classification. Before proceeding with model training, we will split our dataset into training and test sets using Scikit-learn’s `train_test_split` method, with 30% of data points going in the test set as follows.

```
X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

print(X_train.shape, X_test.shape)
```

In the following output, we see the sizes of training and testing sets:

```

(199364, 30) (85443, 30)
```

Note that we’ve set `‘stratify=y’` to enable a stratified train-test split. This ensures that the proportion of the target variable ‘y’ in the training dataset and test dataset is similar to that in the original dataset. Let’s confirm it by running the following:

```
# percentage of class=1 in train and test set
print(y_train.value_counts(normalize=True))
print(y_test.value_counts(normalize=True))
```

In the following output we see that the proportion of fraudulent transactions is similar in training and test sets:

```
Class
0    0.998275
1    0.001725
Name: proportion, dtype: float64
Class
0    0.998268
1    0.001732
Name: proportion, dtype: float64
```

If we don’t do a stratified split, especially for imbalanced datasets, we may end with a training set having all data points belonging to the majority class. This will cause the machine learning algorithm to generalize poorly and not be able to learn about the minority class.

We will now train a logistic regression classifier using the training set and evaluate its predictions for the test set as follows.

```

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

Here, `y_pred` represents the predictions of the logistic regression model for the test set. We need to compute the performance metrics to evaluate how good these predictions are, and Scikit-learn offers a wide range of metrics to choose from.

Undoubtedly, one would reach out for the accuracy (i.e., the ratio of correctly classified data points to the total number of data points) to measure the model’s performance. Let’s look at the accuracy of our test set predictions:

```
print(f'Accuracy: {accuracy_score(y_test, y_pred):.3f}')
```

In the following output we see the accuracy of the model evaluated on the test set:

```
Accuracy: 0.999
```

An accuracy of 99.9% might make us feel that our model is performing exceptionally well and ready for production. However, it’s important to pause and understand that accuracy isn’t always the best metric for evaluating a model’s quality, especially when dealing with highly imbalanced data.

## Why should we not use accuracy?

Accuracy measures the proportion of correct predictions, but it doesn’t account for the importance or sensitivity of each class. In cases where one class dominates, even a high accuracy can be misleading, as it may overlook critical errors in the minority class, like fraudulent transactions in our example.

So for imbalanced datasets, it is encouraged to consider important tools such as confusion matrix, precision, recall, F1 score, ROC curve or, even the [balanced accuracy](https://www.blog.trainindata.com/a-data-scientists-guide-to-balanced-accuracy/), which contemplate the model across the different classes, including the minority.

> Master the use of performance metrics for imbalanced data with our course [Machine Learning with Imbalanced Data](https://www.trainindata.com/p/machine-learning-with-imbalanced-data).

Let’s take a look at these evaluation metrics one by one.

## Confusion matrix

The confusion matrix for binary classification is a 2×2 matrix where the rows represent actual labels and the columns represent the predicted labels. It records the count of data points that were correctly classified by the model (these are shown along the diagonal) and the count of data points for which the model was ‘confused’ and either classified a positive class as negative class (false negative) or vice-versa (false positive).

We can visualize our confusion matrix as a heatmap plot using Seaborn as follows:

```
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
```

![Confusion matrix showing actual values and predictions on the test set]({{ site.baseurl }}/assets/images/posts/class-imbalance-in-machine-learning/confusion_matrix_1.png)

The above confusion matrix tells us that:

- 85259 data points were correctly classified as class ‘0’ (non-fraudulent transactions). These are referred to as **True Negatives**.
- 36 data points were incorrectly classified as class ‘0’, but they actually belong to class ‘1’ (fraudulent transactions). These are known as **False Negatives**.
- 57 data points were incorrectly classified as class ‘1’ (fraudulent transactions), when in fact, they belong to class ‘0’. These are referred to as **False Positives**.
- 91 data points were correctly classified as class ‘1’ (fraudulent transactions). These are known as **True Positives**.

From the confusion matrix, we can see that the model performs well for class ‘0’ (non-fraudulent transactions), but struggles with class ‘1’ (fraudulent transactions).

To get a further idea of the performance of our model, we can also look at other evaluation metrics as well, such as precision and recall.

## Precision

Precision measures the proportion of correctly predicted positive instances out of all instances predicted as positive. For our example, it gives answer to the question – *“Of all the transactions the model predicted to be fraudulent, how many were actually fraudulent?”*

```
print(f'Precision: {precision_score(y_test, y_pred):.3f}')
```

The precision for this model is:

```
Precision: 0.717
```

A precision of 0.717 means that 71.7% of the transactions predicted as fraudulent were actually fraudulent, while the remaining 28.3% were false positives.

## Recall

Recall measures the proportion of actual positive instances that were correctly identified by the model. It answers the question –*“Of all the actual fraudulent transactions, how many were correctly classified by the model?”* From the context of our problem, recall seems to be a suitable measure to optimize since we don’t want to miss any fraudulent transactions.

```
print(f'Recall: {recall_score(y_test, y_pred):.3f}')
```

The recall of this model is:

```
Recall: 0.615
```

A recall of 0.615 means that only 61.5% of the total fraudulent transactions were classified correctly, while the remaining 38.5% went undetected.

These values of precision and recall were calculated using the **default probability threshold of 0.5** This is, however, rarely a suitable threshold when working with imbalanced datasets.

## Adjusting the Probability Threshold Value

Before moving any further, the first thing to do, and this is particularly important when working with imbalanced datasets, is to adjust the threshold of probability used to classify an observation as class 1 or 0.

Scikit-learn uses a threshold of 0.5 by default. But this value, if anything else, is only suitable for balanced datasets and perfectly calibrated classifiers, which is almost always not what we have at hand.

For imbalanced datasets, we will most likely need lower thresholds. Let’s see how the values of the examined metrics change when we use a different value instead of 0.5.

`FixedThresholdClassifier` from Scikit-learn allows us to manually specify the threshold for classification. By testing different threshold values (e.g., 0.3, 0.4, 0.6, etc.), we can observe how the precision and recall scores change and determine which threshold provides the best trade-off between the two.

```
from sklearn.model_selection import FixedThresholdClassifier

thresh_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

precision_scores = []
recall_scores = []

for threshold in thresh_values:
    model_fixed_threshold = FixedThresholdClassifier(estimator=model, threshold=threshold)
    model_fixed_threshold.fit(X_train, y_train)
    y_pred = model_fixed_threshold.predict(X_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision_scores.append(precision)
    recall_scores.append(recall)

plt.plot(thresh_values, precision_scores, label='Precision')
plt.plot(thresh_values, recall_scores, label='Recall')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision and recall at different thresholds')
plt.legend()
plt.show()
```

In the following plot, we see that as the threshold decreases, the model becomes more likely to predict positive class, which improves the recall but comes at the expense of precision:

![Precision and recall at different thresholds]({{ site.baseurl }}/assets/images/posts/class-imbalance-in-machine-learning/precision_and_recall.png)

This plot highlights the trade-off between the two metrics. We see that a better threshold for precision and recall is located between 0.3 and 0.4, where the lines of precision and recall meet.

However, in our example, recall is more critical, so we can prioritize it, and reduce the threshold even further to ensure a high recall value.

> Master the use of precision, recall and tuning the classification threshold with our course [Machine Learning with Imbalanced Data](https://www.trainindata.com/p/machine-learning-with-imbalanced-data).

## Cost-sensitive Learning

Cost-sensitive learning is a straightforward approach to address class imbalance by making the model more sensitive to the minority class.

Cost-sensitive learning works by penalizing the model more heavily for misclassifying the minority class during training. The idea is to assign higher weights to the minority class such that misclassification of data samples from the minority class incurs a higher penalty, so the model puts more effort into minimizing errors in that class.

In Scikit-learn, we can implement cost sensitive learning through the `class_weight` parameter in prediction models such as logistic regression, decision trees, random forests and gradient boosting machines. When `class_weight` is set to `balanced`, then each class gets assigned a weight inversely proportional to its class frequency, also known as imbalance ratio. This means that the minority class will have a higher weight than the majority class.

Let’s implement this and look at the performance of the model.

```
cost_sensitive_model = LogisticRegression(class_weight='balanced')
cost_sensitive_model.fit(X_train, y_train)

y_pred = cost_sensitive_model.predict(X_test)

print(f'Accuracy: {accuracy_score(y_test, y_pred):.3f}')
print(f'Precision: {precision_score(y_test, y_pred):.3f}')
print(f'Recall: {recall_score(y_test, y_pred):.3f}')
```

We can see that the value of recall has improved to 0.885 which means that the model is better able to detect the positive class:

```
Accuracy: 0.965
Precision: 0.042
Recall: 0.885
```

We can also try and specify custom class weights to achieve the desired performance. In fact, the class weight could be one of the many hyperparameters to optimize in our machine learning model.

> To better assess the performance of the cost-sensitive learning trained model, we should also **tune the probability threshold** to determine precision and recall, as we did in the previous section.

## Handling Class Imbalance

If adjusting probability thresholds or specifying class weights don’t resolve the issue, we can try using random **over-sampling** or random **under-sampling** techniques.

[Over-sampling](https://www.blog.trainindata.com/oversampling-techniques-for-imbalanced-data/) involves increasing the number of examples from the minority class while [under-sampling](https://www.blog.trainindata.com/undersampling-techniques-for-imbalanced-data/) reduces the number of examples from the majority class. These techniques balance the dataset and are said to improve the model’s ability to detect minority class instances

Undersampling is suitable when we have huge datasets. By removing observations from the majority class, we can speed up training of the model. For smaller datasets, undersampling risks loss of information.

Oversampling has also its sets of problems. Random oversampling simply duplicates data points. To avoid this, methods that “create” data points similar to those of the minority class, have been suggested to improve model performance on class imbalance. The classical method to create synthetic data is [**SMOTE**](https://www.blog.trainindata.com/overcoming-class-imbalance-with-smote/) (**S**ynthetic **M**inority **O**ver-sampling **TE**chnique).

Imblearn is a Python package that supports many over- and undersampling methods for class imbalance. Whether we [should still be using imbalanced-learn](https://www.blog.trainindata.com/should-you-use-imbalanced-learn-in-2025/) is becoming a subject of heated debate.

As I said, with SMOTE, instead of simply duplicating the data points by sampling with replacement, we generate **new** synthetic data points. It does so by selecting a data point from the minority class and identifying its k-nearest neighbors (KNN). It then creates new data points by interpolating between the selected data point and one of its neighbors, creating synthetic samples that lie along the line segments between them. This approach increases the size of the minority class in a more meaningful way, improving the model’s ability to learn from dominated data without overfitting.

Let’s implement SMOTE to generate the new dataset as follows:

```
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
```

We will only use the training dataset to create new examples using SMOTE and use that to fit the model, and evaluate it on the non-resampled test set. Sometimes SMOTE may generate synthetic data that may not be a true representative of the given data, so this step ensures that we are testing the model’s performance on the data with the original class distribution.

Let’s look at the class distribution in the resampled training data:

```
y_train_resampled.value_counts()

```

We get the following output:

```
Class
0    199020
1    199020
```

The resulting dataset is now perfectly balanced, with an equal number of samples for each class. With this balanced dataset, let’s train the model and look at the performance metrics.

```
# train the classification model (logistic regression)
model = LogisticRegression(random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# predict on test set
y_pred = model.predict(X_test)

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - SMOTE')
```

This gives us the following output:

![Confusion matrix of test set predictions after applying SMOTE]({{ site.baseurl }}/assets/images/posts/class-imbalance-in-machine-learning/confusion_matrix_smote.png)

From the confusion matrix, we can see that both the true positive rate and the true negative rate have improved, as indicated by the values along the diagonal. This means the model is performing better in correctly identifying both fraudulent and non-fraudulent transactions. Now, let’s also take a closer look at other performance metrics to understand its overall performance.

```

print(f'Accuracy: {accuracy_score(y_test, y_pred):.3f}')
print(f'Precision: {precision_score(y_test, y_pred):.3f}')
print(f'Recall: {recall_score(y_test, y_pred):.3f}')
```

The model now achieves a recall of 0.885, meaning that 88.5% of fraudulent transactions are successfully detected, with 11.5% being misclassified as shown in the output below:

```
Accuracy: 0.979
Precision: 0.068
Recall: 0.885
```

In this example, SMOTE has improved the values of the recall **compared to our model without cost-sensitive** learning and **without adjusting the probability cut-off**. And this is something to keep in mind.

## SMOTE is not the silver-bullet

Much of the SMOTE success was shown on weak learners, like support vector machines (SVMs) and random forests, when optimizing threshold dependent metrics like precision and recall, without, and this is key, adjusting the probability threshold. However, a recent article showed that if we adjust the probability threshold, then we don’t need to use SMOTE any more.

## DOs when working with class imbalance

So in short, when working with imbalanced data:

- Use strong classifiers like xgboost and catboost whenever possible.
- Adjust the probability threshold used to assign an observation to a class.
- If adjusting probability thresholds is not enough, try cost-sensitive learning.
- If cost-sensitive learning is not enough, only then try SMOTE (or other resampling methods), but be extra mindful of analysing the quality of the synthetic data.

## **Conclusion**

Class imbalance is a common issue in machine learning, particularly in classification tasks where classes are not evenly distributed. This often occurs in real-world scenarios like disease diagnosis (fewer rare disease cases compared to healthy ones) or fraud detection (fewer fraud cases than legitimate transactions). If not handled well, class imbalance can cause the machine learning algorithm to struggle in detecting these critical, minority cases, leading to serious consequences—sometimes even life-threatening.

Before using imbalanced data for training a model, it’s important to address this issue. Some methods to handle the problem of class imbalance include:

- **Lowering the threshold:** Lowering the decision threshold can improve model performance on imbalanced datasets by helping detect more minority class instances. The default threshold of 0.5 may not work well for such cases, as it can miss positive instances of the minority class.
- **Cost-sensitive learning:** In this approach, we penalize the loss function more if the model misclassifies a minority class example. This encourages the model to give more attention to the underrepresented class during training.

- **Over-sampling**: This increases the number of minority class examples, either by duplicating them or generating new ones. SMOTE is an example of an over-sampling technique that generates synthetic data points from the minority class.

- **Under-sampling**: This reduces the number of majority class examples to balance the dataset, although this might lead to the loss of important information.

Besides handling the class imbalance, the optimization of the right performance metrics is also critical. Relying only on accuracy can be misleading in such cases. It’s crucial to consider other performance metrics like precision, recall, and the confusion matrix for proper model validation.

## Additional Resources

If you want to learn more about class imbalance, check out our course [Machine Learning with Imbalanced Data.](https://www.trainindata.com/p/machine-learning-with-imbalanced-data)

[![Online course Machine Learning with Imbalanced data.]({{ site.baseurl }}/assets/images/posts/class-imbalance-in-machine-learning/imbalanced-data-course-1024x576.png)](https://www.trainindata.com/p/machine-learning-with-imbalanced-data)
