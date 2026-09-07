---
layout: post
title: "A Data Scientist’s Guide to Balanced Accuracy"
author: cmcouto
description: "Discover the balanced accuracy's advantages over traditional accuracy and learn how to implement it in Python."
excerpt: "Discover the balanced accuracy's advantages over traditional accuracy and learn how to implement it in Python."
categories: [Data Science, Imbalanced Data, Machine Learning]
image: assets/images/posts/a-data-scientists-guide-to-balanced-accuracy/balanced_accuracy_featured_image.png
---

In classification problems, evaluating the performance of your models is crucial. When working with imbalanced datasets, traditional accuracy might give misleading results, making you think your model is performing well when, in reality, it’s just favoring the majority class. This is where **balanced accuracy** comes into play.

Balanced accuracy provides a more insightful measure by accounting for both your model’s sensitivity (true positive rate) and specificity (true negative rate). This makes it particularly valuable in real-world scenarios where imbalanced data is common, and the minority class is usually more important.

In this article, we will explore balanced accuracy, why it’s important, and how to implement it in Python.

> Get your **free** copy of our “[7 Takes on Working with Imbalanced Data](https://www.trainindata.com/p/7-takes-on-working-with-imbalanced-data)“, where we discuss 3 recent articles that change the conversation around resampling.

## Understanding Imbalanced Datasets

Before we delve into balanced accuracy, let’s first understand what [imbalanced datasets](https://www.blog.trainindata.com/machine-learning-with-imbalanced-data/) are. In classification problems, an imbalanced dataset is one where the classes are not represented equally. For example, in a binary classification problem, if 90% of the samples belong to class A and only 10% belong to class B, we have an imbalanced dataset.

Imbalanced datasets are common in real-world scenarios, such as:

- Fraud detection (few fraudulent transactions among many legitimate ones)
- Medical diagnosis (rare diseases in a population)
- Predictive maintenance (infrequent equipment failures)

These scenarios challenge traditional machine learning algorithms and evaluation metrics, such as accuracy, which often assume balanced class distributions.

## The Pitfall of Accuracy

Accuracy is defined as the ratio of correct predictions to the total number of predictions:

```
Accuracy = (True Positives + True Negatives) / Total Samples
```

While accuracy is intuitive and easy to understand, it can be highly misleading when dealing with imbalanced datasets. Let’s illustrate this with a Python example.

The following code creates an imbalanced dataset with a binary target. The simulated data will contain only 10% of the categories belonging to target category 1. Then, we’ll simulate predictions of 0 for the whole dataset and compute the accuracy to assess the output:

```
import numpy as np
from sklearn.metrics import accuracy_score

# Simulating an imbalanced dataset
y_true = np.array([0] * 900 + [1] * 100)  # 90% class 0, 10% class 1

# A "dummy" classifier that always predicts the majority class
y_pred = np.zeros_like(y_true)

# Compute the accuracy metric
acc = accuracy_score(y_true, y_pred)
print(f"Accuracy: {acc:.2%}")
```

As a result, our dummy classifier achieved an impressive 90% of accuracy:

```
Accuracy: 90.00%
```

Despite predicting 0 for all instances, the “dummy” classifier achieved a great accuracy score. However, this classifier is utterly useless for identifying samples of class 1, which might be the class of interest (e.g., fraud detection). It’s “useless” because this model can’t predict the other category (the one we’re interested in). Therefore, there’s no intelligence and no action to be taken.

> To master performance metrics suitable for imbalanced datasets, check out our course [Machine Learning with Imbalanced Data](https://www.trainindata.com/p/machine-learning-with-imbalanced-data).

This scenario demonstrates why accuracy alone can be deceptive when working with imbalanced datasets.

## Introducing Balanced Accuracy

Balanced accuracy addresses the limitations of traditional accuracy by taking into account the performance of the model on both classes, regardless of their proportions in the dataset. It is defined as the average of recall obtained in each class:

```
Balanced Accuracy = (Sensitivity + Specificity) / 2
```

Where:

- Sensitivity (True Positive Rate) = True Positives / (True Positives + False Negatives)
- Specificity (True Negative Rate) = True Negatives / (True Negatives + False Positives)

In other words, it’s the average recall obtained on each class.

Balanced accuracy provides a more reliable metric for imbalanced datasets by giving equal weight to the performance of both classes.

### Implementing Balanced Accuracy in Python

Let’s implement balanced accuracy from scratch and compare it with scikit-learn’s implementation.

First, let’s create a function to compute the balanced accuracy using only `numpy`:

```
def balanced_accuracy(y_true, y_pred):
    """Compute balance accuracy using numpy"""
    # Convert inputs to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Get unique classes
    classes = np.unique(y_true)

    # Calculate recall for each class
    recalls = []
    for cls in classes:
        true_positives = np.sum((y_true == cls) & (y_pred == cls))
        actual_positives = np.sum(y_true == cls)
        recall = true_positives / actual_positives
        recalls.append(recall)

    # Return the arithmetic mean
    return np.mean(recalls)
```

This function computes the recall average for each class (i.e., balanced accuracy). Now, let’s create sample labels and predictions so we can compare the output from our manual function with scikit-learn’s implementation:

```
import numpy as np
from sklearn.metrics import balanced_accuracy_score

# Example labels & predictions
y_true = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])

# Compute balanced accuracy using both functions
custom_ba = balanced_accuracy(y_true, y_pred)          # our function
sklearn_ba = balanced_accuracy_score(y_true, y_pred)   # scikit-learn's function

# Print results for both functions
print(f"Custom Balanced Accuracy: {custom_ba:.2%}")
print(f"Scikit-learn Balanced Accuracy: {sklearn_ba:.2%}")
```

In the following output, we can see that both approaches – from scratch and with `scikit-learn` – yields the same balanced accuracy score:

```
Custom Balanced Accuracy: 75.00%
Scikit-learn Balanced Accuracy: 75.00%
```

In this section, we’ve implemented balanced accuracy from scratch and compared it with scikit-learn’s implementation. Both methods yield the same result (average between the recalls: 1 for class 0 and 0.5 for class 1), confirming the correctness of our implementation.

> Master the use of performance metrics in Python with our course [Machine Learning with Imbalanced Data](https://www.trainindata.com/p/machine-learning-with-imbalanced-data). Straight to the point, full Python code implementations.

## Comparing Balanced Accuracy with other Performance Metrics

In this section, we’ll explore balanced accuracy further by comparing it with other performance metrics using an imbalanced dataset:

```
# Functions to simulate & split dataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Creating a highly imbalanced dataset
X, y = make_classification(
    n_samples=10_000, n_features=10, n_classes=2, n_informative=5,
    weights=[0.9, 0.1], flip_y=0, random_state=42
)

# Splitting the dataset into training testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
```

In this example, we created an imbalanced dataset with a 90:10 class distribution and split it into training and testing sets. Since we have two classes, we are working on a task for a binary classifier.

Next, we’ll use two models to compute and compare the performance metrics on this dataset:

- A dummy model to predict the most frequent class
- A simple ensemble model (Random Forest) to predict according to the data patterns

```
# Import classifiers
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier

# Training a dummy classifier
dummy_model = DummyClassifier(strategy='most_frequent')
dummy_model.fit(X_train, y_train)

# Training a simple RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Compute prediction for each model
y_pred_dummy = dummy_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)
```

In this example, we’ve trained both models using the training set and computed the predictions using the test set.

Let’s validate the performance of each model, starting with the dummy classifier:

```
# Metrics to assess the classifier performance
from sklearn.metrics import balanced_accuracy_score, classification_report

# Compute balanced accuracy for the dummy model
bal_acc_dummy = balanced_accuracy_score(y_test, y_pred_dummy)

# Overall model validation
print('Dummy Classifier')
print(classification_report(y_test, y_pred_dummy))
print(f'Balanced Accuracy: {bal_acc_dummy:.2f}')
```

In the following output, we can see the [classification report](https://www.blog.trainindata.com/confusion-matrix-precision-and-recall/) for the dummy classifier. According to accuracy, our dummy model identifies 90% of the observations correctly. However, note that the recall is 0 because the model correctly classifies observations belonging to class 0 but always miss-classifies the observations belonging to class 1. That’s why the balanced accuracy is 0.5 (50%). It gets 100% correct for class 1 and 100% wrong for the other.

```
Dummy Classifier
              precision    recall  f1-score   support

           0       0.90      1.00      0.95      2713
           1       0.00      0.00      0.00       287

    accuracy                           0.90      3000
   macro avg       0.45      0.50      0.47      3000
weighted avg       0.82      0.90      0.86      3000

Balanced Accuracy: 0.50
```

Now, let’s calculate the same performance metrics for the Random Forest:

```
# Compute balanced accuracy for random forest
bal_acc_rf = balanced_accuracy_score(y_test, y_pred_rf)

# Overall model validation
print('Random Forest')
print(classification_report(y_test, y_pred_rf))
print(f'Balanced Accuracy: {bal_acc_rf:.2f}')
```

The following output shows the classification report for the random forest classifier. Unlike the dummy classifier, this model actually works with pattern recognition. Random forest achieved a higher accuracy (96%) and could identify both classes, with a recall of 99% and 70% for classes 0 and 1, respectively. The average between these two values (recalls) is 0.85, our balanced accuracy score.

```
Random Forest
              precision    recall  f1-score   support

           0       0.97      0.99      0.98      2713
           1       0.92      0.70      0.79       287

    accuracy                           0.96      3000
   macro avg       0.94      0.85      0.89      3000
weighted avg       0.96      0.96      0.96      3000

Balanced Accuracy: 0.85
```

It’s also worth noting that the balanced accuracy, despite being developed initially to evaluate the results of a binary classifier, can also be applied to multiclass classification tasks. It computes the same value as the “macro avg” output from the sklearn’s classification report for recall.

## When to Use Balanced Accuracy

Balanced accuracy is particularly useful in the following scenarios:

- **Imbalanced datasets**: When one class significantly outnumbers the other(s), balanced accuracy provides a more reliable performance metric.

- **Model comparison**: When comparing different models on imbalanced datasets, balanced accuracy offers a fairer comparison by considering the performance of all classes.
- **Threshold optimization**: Balanced accuracy can optimize the decision threshold for binary classifiers, especially when dealing with imbalanced datasets.

> **Note:** Balanced accuracy is the default metric used to optimize the decision threshold for the classifiers using [TunedThresholdClassifierCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TunedThresholdClassifierCV.html), which was recently introduced on scikit-learn (added in version 1.5).

## Limitations and Considerations

While balanced accuracy is a valuable metric for imbalanced datasets, it’s important to be aware of its limitations:

- **Insensitivity to class proportions**: Balanced accuracy gives equal weight to all classes, regardless of their proportions in the dataset. This may not always be desirable, especially if the class imbalance reflects the true distribution of the population.
- **Potential for high variance**: In cases of extreme imbalance or small sample sizes, balanced accuracy can have high variance due to the performance of the minority class having a large impact on the overall score.
- **May hide overall performance**: By focusing on class-wise performance, balanced accuracy might obscure the overall predictive power of the model.

To address these limitations, it’s often beneficial to use balanced accuracy in conjunction with other metrics and visualizations, such as:

- Precision-Recall curves
- ROC curves and its AUC
- F1-score
- Confusion matrices

## Conclusion

Balanced accuracy is a powerful metric for evaluating classification models on imbalanced datasets. Giving equal weight to the performance in all classes provides a more reliable assessment of the performance of your machine learning models than traditional accuracy.

In this article, we’ve explored the concept of balanced accuracy, implemented it from scratch, and demonstrated its application in various scenarios using Python. We’ve seen how it can reveal model shortcomings that might be hidden by traditional accuracy, especially when dealing with imbalanced datasets.

While balanced accuracy is not a silver bullet, it is an essential tool in the data scientist’s toolkit for handling imbalanced classification problems. By combining balanced accuracy with other metrics and techniques, such as threshold optimization and resampling methods, you can develop more robust and reliable models for real-world imbalanced datasets.

Remember, the choice of evaluation metric should always be guided by the specific requirements of your problem and the consequences of different types of misclassifications. Balanced accuracy shines in situations where performance for all classes is equally important, regardless of their proportions in the dataset.

As you continue your journey in data science and machine learning, consider balanced accuracy when tackling imbalanced datasets. It might be the key to unlocking better model performance and more meaningful insights from your data.

## Master Classification Metrics for Imbalanced Datasets

To master classification metrics for imbalanced datasets in Python, check out our course [Machine Learning with Imbalanced Data](https://www.trainindata.com/p/machine-learning-with-imbalanced-data).

[![Online course Machine Learning with Imbalanced data.]({{ site.baseurl }}/assets/images/posts/should-you-use-imbalanced-learn-in-2025/imbalanced-data-course.png)](https://www.trainindata.com/p/machine-learning-with-imbalanced-data)
