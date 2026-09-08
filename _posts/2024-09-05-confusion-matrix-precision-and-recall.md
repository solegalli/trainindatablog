---
layout: post
title: "Confusion Matrix, Precision, and Recall"
author: priyansh
description: "Find out what the confusion matrix is and how it relates to other classification metrics like precision, recall and f1-score."
excerpt: "Find out what the confusion matrix is and how it relates to other classification metrics like precision, recall and f1-score."
categories: [Machine Learning]
image: assets/images/posts/confusion-matrix-precision-and-recall/confusion-matrix-precision-recall.png
---

In machine learning, assessing model performance is crucial for building reliable and effective systems. The metrics used to evaluate a model’s performance are known as Evaluation Metrics. These metrics vary depending on the type of problem being solved — Regression, Classification, Clustering, etc.

For example, when we deal with **Regression**, the evaluation metric is the “***Error”*** made by our model algorithm, which is simply the difference between predicted values and the actual values. Similarly, for **Classification tasks**, a commonly used evaluation metric is “***Accuracy”***, which indicates how well the model correctly predicts the class labels. For instance, in a binary classification problem, accuracy reflects how accurately the model distinguishes between two classes.

In this article we are going to discuss some of the essential tools and key metrics we use to evaluate Classification Models. One of the most fundamental tools for this purpose is the **Confusion** **Matrix.**

> Advance your data science and machine learning skills with our [comprehensive expert led courses](https://www.trainindata.com/courses).

[![Advance your data science and machine learning skills with our comprehensive and expert led courses.]({{ site.baseurl }}/assets/images/posts/complete-guide-to-platt-scaling/Rectangle-7.png)](https://www.trainindata.com/courses)

## Confusion Matrix

A confusion matrix is a simple table that provides a comprehensive view of how well a classification model is performing. It shows the correct and incorrect predictions made by our model, broken down into four key components — **True Positives (TP), False Positives (FP), True Negatives (TN),** and **False Negatives (FN).**

A typical Confusion Matrix looks like this:
![Confusion matrix showing the real values and the predictions made by the model]({{ site.baseurl }}/assets/images/posts/confusion-matrix-precision-and-recall/confusion-matrix-1.png)

The y-axis represents the **Actual Labels**, and the x-axis represents the **Predicted Labels**. True Positives and True Negatives are correct predictions made by the model. False Positives and False Negatives are incorrect predictions made by the model.

Let’s understand this with the help of a simple example of a Spam Classification Model. The idea is simple — we have to evaluate a machine-learning model that classifies email as spam and not spam.

Assume we have a dataset of 300 emails, where 90 emails are labeled as spam (class 1 or Positive class) and 210 emails are labeled as not spam (class 0 or Negative class).

After training a [machine learning](https://www.blog.trainindata.com/machine-learning-fundamentals/) model, the following predictions are made:

- **True Positives (TP)**: The model correctly identified 70 emails as spam—these 70 emails were actually spam and were predicted as spam. Truly Spam – True Positive.
- **False Positives (FP)**: The model incorrectly identified 20 emails as spam—these 20 emails were actually not spam but were predicted as spam. This type of error is called a False Positive, also known as a **Type I error**, where the model falsely predicts something as positive when it’s not.
- **True Negatives (TN)**: The model correctly identified 180 emails as not-spam—these 180 emails were actually not-spam and were predicted as not-spam. Truly Not-Spam – True Negative.
- **False Negatives (FN)**: The model incorrectly identified 30 emails as not spam—these 30 emails were actually spam but were predicted as not spam. This type of error is called a False Negative, also known as a **Type II error**, where the model fails to detect something that is actually positive.

These numbers, when plotted in a table with True Labels on the y-axis and Predicted Labels on the x-axis, form the Confusion Matrix for our spam classification model:

![Confusion matrix showing the real value of the target and the predictions made by the model]({{ site.baseurl }}/assets/images/posts/confusion-matrix-precision-and-recall/confusion-matrix-2.png)

A confusion matrix can be plotted for both binary-class and multi-class classification models. Using the Confusion Matrix, we can calculate other key metrics that provide deeper insights into the model’s performance. Two such most important metrics derived from the Confusion Matrix are **Precision** and **Recall**.

> Advance your data science and machine learning skills with our [comprehensive expert led courses](https://www.trainindata.com/courses).

[![Advance your data science and machine learning skills with our comprehensive and expert led courses.]({{ site.baseurl }}/assets/images/posts/complete-guide-to-platt-scaling/Rectangle-7.png)](https://www.trainindata.com/courses)

## Precision

Precision is a metric that answers the question: “Of all the positive predictions made by the model, how many were actually correct?”. It is a ratio of true positive predictions out of all positive predictions made by the model.
In our spam classification model, Precision indicates : “When the model predicts something as spam, how often is it right?”. It is simply the ratio of true spam emails correctly identified out of all the emails that were predicted as spam by the model.

Precision can also be viewed as the **accuracy of positive predictions**.

#### Formula for Precision

From the confusion matrix, we can see that correctly predicted positives are called True Positives (TP) and incorrectly predicted positives are called False Positives (FP). So, the total positives predicted by the model would be **TP + FP**. Therefore, the ratio of actual positives out of total positives predicted by the model would be:

![Equation of the precision metric]({{ site.baseurl }}/assets/images/posts/confusion-matrix-precision-and-recall/precision-formula.png)

Therefore, for our spam classification model, the Precision would be 70/90 ≈ 0.78 or 78%. This means that the classifier is 78% accurate in identifying an email as spam.

#### Importance of Precision

Precision is especially important in scenarios where the cost of a False Positive is high or when incorrectly predicting something as positive can have significant consequences. For example, in medical diagnosis, predicting a disease when it’s not actually present (a False Positive) can lead to unnecessary stress, additional testing, or even harmful treatments for the patient. Similarly, in fraud detection, falsely labeling a legitimate transaction as fraudulent can result in inconvenience to customers, damage reputations, and cause financial losses. In our spam detection example, if the model frequently marks important emails as spam (False Positives), users may miss out on crucial information.

A high Precision ensures that when the model predicts something as positive—whether it’s spam, a disease, or fraudulent activity—it is likely to be correct. This reduces the risks associated with False Positives, making Precision a critical metric in high-stakes domains like medical diagnosis, financial and fraud detection, security and surveillance, pharmaceutical and drug development, etc.

## Recall

Recall, also known as **Sensitivity** and **True Positive Rate**, answers the question: “Of all the actual positive cases, how many did the model correctly identify?”. It is the ratio of true positive predictions out of all actual positive instances. In other words, it tells us how well the model finds all positive instances.

In our spam classification model, Recall indicates: “When there is a spam email, how often does the model correctly identify it as spam?” It’s the ratio of true spam emails correctly identified by the model out of all the actual spam emails present in the dataset.

Recall can also be seen as the ability of the model to **capture all possible positive cases.**

#### Formula for Recall

True Positives (TP) are instances that were actually positive and were predicted positives. False Negatives (FN) are instances that were actually positive but were predicted as negative. So, the total number of actual positive instances in the dataset would be **TP + FN.** Therefore, the ratio of total positive instances out of all the actual positive instances would be:

![Equation of recall]({{ site.baseurl }}/assets/images/posts/confusion-matrix-precision-and-recall/recal-equation.png)

Therefore, for our spam classification model, the Recall would be 70/100 ≈ 0.70 or 70%. This means that the classifier correctly identifies 70% of all actual spam emails.

#### Importance of Recall

Recall is very important in scenarios where the cost of a False Negative is very high or when wrongly predicting a positive case as negative has serious implications. For example, failing to diagnose a patient with cancer (False Negative) can have very serious implications in the healthcare industry.

Some areas where recall is considered very critical are security and emergency response systems, where the cost of missing a genuine threat (false negative) is far more than a false alarm (false positives), medical diagnosis, defect detection in manufacturing, lead generation, etc.

## Precision Vs Recall: Which is more Important?

Precision and Recall often exist in tension, meaning that improving one typically lowers the other. This is because of the way these metrics are defined and how they interact with the decision threshold of a classification model.

For instance, in binary classification, a model assigns a score or probability to each data point, and a decision threshold determines whether the data point is classified as positive (class 1) or negative (class 0). When this threshold is lowered, the model will classify more data points as positive, which can increase the Recall (since the number of true positives captured will increase), but may decrease precision (as the model can now predict more data points as False Positives). Similarly, when we raise the threshold, the model becomes strict and reduces the number of instances classified as positive, which can increase precision (since fewer false positives are included) but may decrease recall (as more true positives are missed).

Having a trade-off between precision and recall is essential for optimizing model performance based on the problem. For example, in medical diagnosis, failing to identify a patient with a disease (a False Negative) is more critical than incorrectly diagnosing a healthy patient as sick (a False Positive). In this case, having **high Recall** is crucial to ensure that as many cases as possible are caught, even if it means dealing with some false positives. Conversely, in situations like Spam classification, it is more important to avoid misclassifying legitimate emails as spam (False Positives) and therefore maintaining **high Precision** is important.

F1 Score and Precision-Recall curves are used to balance precision and recall, providing a more comprehensive view of the model’s effectiveness.

> Advance your data science and machine learning skills with our [comprehensive expert led courses](https://www.trainindata.com/courses).

[![Advance your data science and machine learning skills with our comprehensive and expert led courses.]({{ site.baseurl }}/assets/images/posts/complete-guide-to-platt-scaling/Rectangle-7.png)](https://www.trainindata.com/courses)

### F1-Score

The F1 Score is the harmonic mean of Precision and Recall, offering a single measure that accounts for both metrics. It combines the strengths of Precision and Recall into a single number, making it easier to compare models and choose the best one based on overall performance.

The formula for F1-score is:

![Equation for the f1-score]({{ site.baseurl }}/assets/images/posts/confusion-matrix-precision-and-recall/equaion-f1-score.png)

The F1 score is especially useful when dealing with imbalanced datasets, where one class significantly outnumbers the other. As a harmonic mean, it gives equal weight to both metrics and ensures that the model’s performance is not skewed by the dominant class.

### Precision-Recall Curves

Precision-Recall (PR) curve is a powerful tool for evaluating the performance of binary classification models, particularly in scenarios where class imbalance is a significant concern. The curve plots precision against recall at various thresholds, helping to find an optimal balance between these two metrics.

By adjusting the threshold, you can observe how Precision and Recall trade off against each other. A curve that is close to the top-right corner of the plot indicates high Precision and high Recall, signifying better model performance. Lowering the threshold typically increases Recall as more positive instances are captured, but it may also decrease Precision due to an increase in False Positives. Conversely, raising the threshold tends to improve Precision by reducing False Positives but may lower Recall, as fewer true positives are identified.

The below image shows a typical precision-recall curve:

![Precision-Recall curve]({{ site.baseurl }}/assets/images/posts/confusion-matrix-precision-and-recall/precision-recall-curve.png)

The shape of the PR curve offers insights into a model’s performance. A curve that is close to the top-right corner of the plot indicates high Precision and high Recall, signifying better model performance. The area under the PR curve (**PR AUC**) summarizes this performance: a higher PR AUC reflects a model that achieves a good balance between Precision and Recall. This is particularly useful in imbalanced datasets, where traditional metrics like Accuracy can be misleading.

Other performance metrics and representations that can help assess a classification model’s strengths and weaknesses include:

- **Specificity** — True Negative Rate
- **F2 score, F-beta score** — variants of F1 score
- **AUC–ROC curve** — Area Under the Receiver Operating Characteristic Curve, which plots the True Positive Rate (Recall) against the False Positive Rate (**FPR** or 1-Specificity) at various threshold settings
- **Log Loss** — Measures the uncertainty of the predictions, typically used in logistic regression.
- **Gini Coefficient** — Derived from the ROC curve

All the above metrics can be effortlessly computed and evaluated in Python using the **sklearn.metrics** module from the scikit-learn library.

Together with the Confusion Matrix, Precision, and Recall, these metrics provide a comprehensive understanding of the model’s strengths and weaknesses. By carefully selecting and interpreting these metrics, you can optimize the model to better meet the specific goals of your application.

## Additional resources

Master these and other metrics for classification, in the context of imbalanced datasets with our book [Machine Learning with Imbalanced Data](https://www.trainindata.com/p/imbalanced-data-myths-mistakes-solutions-book).

[![Imbalanced Data: Myths, Mistakes and Modern Solutions - book by Soledad Galli]({{ site.baseurl }}/assets/images/imbalanced-data-book-cover.jpg)](https://www.trainindata.com/p/imbalanced-data-myths-mistakes-solutions-book)
