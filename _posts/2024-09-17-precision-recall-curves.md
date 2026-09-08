---
layout: post
title: "Precision Recall Curves"
author: noor
description: "The ultimate guide to Precision-Recall curves—what they are, when to use them, and how to plot and interpret them with Python."
excerpt: "The ultimate guide to Precision-Recall curves—what they are, when to use them, and how to plot and interpret them with Python."
categories: [Data Science, Imbalanced Data, Machine Learning]
image: assets/images/posts/precision-recall-curves/featured-image.jpg
---

When working with machine learning models, a key aspect consists of evaluating how well the model performs. For classification problems, there are several tools that we can use, like the ROC curve, gain charts, lift charts and the Precision-Recall curve.

Here, I’ll discuss the Precision-Recall curve and show how to create this plot through a Python example.

## **Understanding Precision and Recall**

Before moving on to Precision-Recall curve, we need to know what precision, recall and confusion matrix are. Let’s dive into it.

### **Confusion Matrix**

A confusion matrix is a tool used in [data science](https://www.blog.trainindata.com/data-science-prerequisites/) and [machine learning](https://www.blog.trainindata.com/machine-learning-fundamentals/) to evaluate the performance of classification models. It helps visualize how well a model’s predictions match the actual outcomes by comparing predicted versus true values—for each class in the dataset. In binary classification, this means evaluating performance across two classes; in multi-class classification, across multiple categories.

For example, suppose we have a binary classifier built to predict whether a patient has diabetes or not. The dataset contains medical records where each patient is labeled as either having diabetes (1 – the positive class) or not having diabetes (0 – the negative class). A confusion matrix allows us to assess how accurately the model identifies these classes.

> To master classification metrics in the context of imbalanced data, check out our book [Machine Learning with Imbalanced Data](https://www.trainindata.com/p/imbalanced-data-myths-mistakes-solutions-book).

### **Components of a Confusion Matrix**

A confusion matrix consists of the following values:

1. **True Positives (TP):** The number of instances that were correctly predicted as the positive class.
2. **True Negatives (TN):** The number of instances that were correctly predicted as the negative class.
3. **False Positives (FP):** The number of instances that were predicted as positives, but actually belong to the negative class.
4. **False Negatives (FN):** The number of instances that were predicted as negatives, but belong to the positive class.

The following diagram shows a confusion matrix and its components. On the y-axis we find the actual labels for the target. On the x-axis, we see the predicted labels. The quadrants marked in green show the instances correctly predicted by the model, that is, the True Positives and True Negatives. The quadrants marked in red show the instances incorrectly predicted by the model, that is, the False Positives and False Negatives.

![Confusion matrix showing actual and predicted values.]({{ site.baseurl }}/assets/images/posts/precision-recall-curves/Confusion-Matrix.png)

The first word of the terminology represents the truth: either it is true or false. The second word in the terminology represents what model predicted about the instance, that is, whether it belonged to positive class or negative class. So, for “True Positive “, “Positive” indicates that the classifier predicted that patient has diabetes and “True“ represents that yes, indeed this is the reality and classifier’s prediction is true.

### **Precision**

Precision is the ratio between true positive predictions and the total number of positive predictions. In other words, precision represents the proportion of positive predictions that were actually correct.

Precision answers the question, *“Of all the instances the model predicted as positive, how many were actually positive?”* Precision is also known as Positive Predictive Value.

A **High Precision value** means that when the model predicts a patient has diabetes, it is usually correct.

**Example**: Imagine our binary classification model predicts 10 positive instances, i.e, patients having diabetes. If 8 out of these 10 patients actually do have diabetes, the precision is high.

Precision is calculated as:

![Formula of precision]({{ site.baseurl }}/assets/images/posts/precision-recall-curves/Precision-formula-1.png)

So, if 8 patients actually have diabetes out of the 10 positive predictions, the precision score is:

Precision=8/10 = 0.80 or 80%

### **Recall**

Recall is the ratio of true positive predictions to the the total number of actual positives. In other words, recall represents how well a classification model identifies **all actual positive classes.**

Recall answers the question, *“Of all the actual positive instances, how many did the model correctly identify?”* Recall is also known as True Positive Rate (TPR).

A **High Recall value** means that the model successfully identifies most of the patients with diabetes.

**Example:** If there are 20 patients with diabetes in our dataset and the binary classification model correctly identifies 15 of them, the recall is high.

Recall is calculated as:

![Formula of recall]({{ site.baseurl }}/assets/images/posts/precision-recall-curves/Recall-formula.png)

So, if 15 out of 20 actual diabetics are correctly identified, the recall is:

Recall=15/20 = 0.75 or 75%

> Want to extend precision and recall to multiclass classification? Check out our book [Machine Learning with Imbalanced Data](https://www.trainindata.com/p/imbalanced-data-myths-mistakes-solutions-book).

## **Precision Recall Tradeoff – Striking the right balance**

Precision and recall often trade off against each other, meaning that improving one can come at the cost of reducing the other. This tradeoff is particularly important when dealing with imbalanced datasets, such as our diabetes medical diagnosis dataset.

To determine a good balance between precision and recall, we can use the f1-score.

### **F1 score**

The f-1 score is the geometric average between precision and recall. It is calculated as:

F1-score = 2 x precision X recall / (precision + recall)

So following our example on diabetes and using the former values of precision and recall, we obtain:

F1-score = 2 x 0.8 x 0.75 / (0.8 + 9.75) = 2.75

Precision, recall and f1-score are **threshold dependent** metrics. Let’s understand what a threshold value is and how it affects the different metrics.

### **Classification Threshold Value**

A threshold value is a probability point that determines the cut-off at which the model (or, actually we) decides whether to classify an instance as positive or negative.

For example, in binary classification, the model outputs a probability score between 0 and 1 for each instance. The higher the score, the more likely the instance belongs to class 1. Now, we need a threshold to decide if the instance will be classified as class 1. Lets say, for now, the threshold value is 0.8

If the probability score (say 0.9) is higher than the threshold (0.8), the model classifies the instance as **positive (diabetic).**

If the probability score (say 0.7) is lower than the threshold (0.8), the model classifies the instance as **negative (non-diabetic).**

By default, this threshold is usually set at **0.5** (meaning a patient is classified as diabetic if the probability value is greater than 0.5). A threshold of 0.5, however, is rarely useful, in particular when working with imbalanced data, so we can (and must) adjust it. That means that we would move the threshold up or down depending on whether we want to prioritize precision or recall.

> Discover why the classification threshold is crucial to correctly evaluate a model’s performance in our book [Machine Learning with Imbalanced Data](https://www.trainindata.com/p/imbalanced-data-myths-mistakes-solutions-book).

### **How the Threshold Affects Classification**

The threshold directly influences the balance between precision and recall:

- **Higher Threshold:** The model is more strict, requiring higher confidence before classifying an instance as a positive case. This typically improves precision (fewer false positives), but lowers recall (more false negatives).
- **Lower Threshold:** The model becomes less strict, classifying more instances as positive. This increases recall (fewer false negatives), but reduces precision (more false positives).

In most cases, increasing precision will reduce recall, and vice versa. This happens because of the way classification thresholds work. The tradeoff occurs because by adjusting the threshold, we change how strict the model is about classifying positives.

> Incorrectly setting up the classification threshold, has led to years of misuse of resampling techniques. Find out why on our free booklet “[7 Takes on Working with Imbalanced Data](https://www.blog.trainindata.com/should-you-use-imbalanced-learn-in-2025/)“.

[![7 takes on working with imbalanced data, free booklet.]({{ site.baseurl }}/assets/images/posts/should-you-use-imbalanced-learn-in-2025/MLID-booklet-presentation.png)](https://www.blog.trainindata.com/should-you-use-imbalanced-learn-in-2025/)

### **Optimizing the Right Metric**

When dealing with **diabetes prediction**, the priority between precision and recall depends on the context of the application, but in most cases, **recall** is more important. Here’s why:

#### **Precision vs. Recall in Diabetes Prediction**

**Recall** focuses on minimizing false negatives—instances where a diabetic patient is wrongly classified as non-diabetic. Missing a diagnosis of diabetes can be very harmful as it can lead to other complications. Therefore, we want the model to identify as many actual diabetic cases as possible. High recall ensures that most patients who have diabetes will be correctly identified, and therefore will receive the right treatment.

**Precision** (minimizing false positives) is important. False positives might lead to unnecessary tests or treatments, but they are less risky compared to the consequences of missed diagnoses.

Therefore, it’s crucial to find a threshold that maximizes recall while maintaining an acceptable level of precision.

## **Precision Recall Curve**

A precision recall curve is very useful for visualizing the change in precision and recall values at different classification thresholds. It helps us optimize the 2  metrics by finding the best threshold value.

Here’s why a Precision-Recall curve is useful:

1. **Imbalanced Datasets:** When one class (e.g., “negative”) is much more common than the other (e.g., “positive”), traditional metrics like accuracy are misleading. A Precision-Recall curve provides a clearer picture of the model’s ability to detect the minority (positive) class.
2. **Trade-off Analysis:** There is a trade-off between precision and recall—if we increase precision, recall might decrease, and vice versa. The PR curve visually represents this trade-off, allowing us to choose the most appropriate balance for our use case.
3. **Model Comparison:** The area under the Precision-Recall curve (AUC-PR) can be used to compare models. A higher AUC-PR indicates better performance.

## **Implementation of PR curve**

Let’s plot the precision-recall curve for a diabetes prediction dataset and logistic regression. For this example, I used the diabetes dataset which is available on [Kaggle](https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset).

### **Step 1: Loading necessary libraries**

First of all, we need to import the necessary Python libraries. We will use numpy, scikit-learn and matplotlib:

```
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    classification_report,
    confusion_matrix,
)
```

### **Step 2: Preparing the data**

In this step, we load the dataset and prepare it for model training by creating the **Feature Matrix** `X` and the **Target Vector** `y`:

- **X (Feature Matrix)** contains the independent variables or features, such as age, BMI, or glucose level, that will be used by the model to make predictions.
- **y (Target Vector)** contains the labels, which indicate the outcome we want to predict. In this case, it shows whether a patient is diabetic (1) or not (0).

Once we have defined X and y, we split the dataset into **training** and **testing** sets using an 80-20 split:

- **X_train** and **y_train** represent the data used to train the model.
- **X_test** and **y_test** represent the data used to evaluate the model’s performance after training.

Finally, we scale the features to standardize the data, which ensures that each feature contributes equally to the model. [Feature scaling](https://www.blog.trainindata.com/feature-scaling-in-machine-learning/) or standardization is a prerequisite for logistic regression and consists of transforming the data to have a mean of 0 and a standard deviation of 1.

```
# I downloaded the dataset from Kaggle and stored it in a csv file:
df = pd.read_csv("diabetes.csv")

# Create feature matrix and target
X = df.drop('Outcome', axis=1)  # Features
y = df['Outcome']               # Target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
)

# Scale the features to standardize the data
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### **Step 3: Training the Logistic Regression Model**

Once the data is prepared and scaled, we proceed with training a logistic regression model. Logistic regression is a classification algorithm used for binary classification problems, such as predicting whether a patient has diabetes or not.

The model is trained using the training data (X_train_scaled and y_train), so that it learns to associate input features with the target labels (diabetic or non-diabetic).

```
model = LogisticRegression(random_state=42)

model.fit(X_train_scaled, y_train)
```

### **Step 4: Predicting Probabilities and Generating a Precision Recall curve**

Instead of predicting the classes directly (diabetic or non-diabetic), we predict the **probability** of a patient being diabetic. This allows us to explore different thresholds for classification and evaluate the precision-recall tradeoff.

- **y_scores**: The predicted probabilities for the test set are stored in y_scores.
- Using these probabilities, we can compute the Precision-Recall (PR) curve, which plots precision (y-axis) against recall (x-axis) at various classification thresholds.

```
# Predict probabilities for the test set
y_scores = model.predict_proba(X_test_scaled)[:, 1]

# Generate precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
```

### **Step 5: Finding the Best Threshold Based on F1-Score**

The **F1-score** represents the harmonic mean of both precision and recall. F1 score is also a reasonable metric to optimize as it balances both precision and recall, making it useful when the dataset is imbalanced or when we want to ensure both false positives and false negatives are minimized.

We calculate the F1-score for each threshold and choose the one that gives the highest F1-score.

```
# Find the threshold that gives the best F1-score

f1_scores = 2 * (precision * recall) / (precision + recall)

best_threshold_index = np.argmax(f1_scores)

best_threshold = thresholds[best_threshold_index]

best_f1 = f1_scores[best_threshold_index]

# Print the best threshold and corresponding F1-score

print(f"Best Threshold: {best_threshold:.2f}")

print(f"Best F1-score: {best_f1:.2f}")
```

Below we see the best threshold and f1-score values:

```
Best Threshold: 0.54
Best F1-score: 0.69
```

### **Step 6: Plotting the Precision-Recall Curve**

Next, we plot the Precision-Recall curve and highlight the point where the F1-score is highest (i.e., the best threshold).

- The plot helps visualize the trade- off between precision and recall across various thresholds.
- The red point on the curve shows the threshold with the best F1-score

```
# Plot the precision-recall curve

plt.figure(figsize=(8, 6))

plt.plot(recall, precision, label='Precision-Recall curve')

plt.xlabel('Recall')

plt.ylabel('Precision')

plt.title('Precision-Recall Curve')

plt.legend(loc='best')

# Plot the threshold points

plt.scatter(recall[best_threshold_index], precision[best_threshold_index], color='red', label=f'Best Threshold (F1 = {best_f1:.2f})')

plt.legend(loc='best')

plt.show()
```

We examine the plot in the next section.

### **Findings**

At low recall values (near 0), the precision is high (close to 1), indicating the model is being very selective but accurate when it does make positive predictions. As recall increases, precision gradually decreases, showing that the model is capturing more positives but at the cost of producing more false positives. The best threshold point (marked with the red dot) is where the model achieves the most balanced performance based on precision and recall, leading to the highest F1-score (0.69).

![Precision Recall curve highlighting the best threshold value]({{ site.baseurl }}/assets/images/posts/precision-recall-curves/Precision-recall-curve-1.png)

A good model will have a curve that stays close to the top-right corner, which means both precision and recall are high across most threshold settings. Like this, the area under the PR curve can be an indicator of performance. The larger the area, the better the model performs in terms of precision and recall trade-offs.

### **Step 7: Evaluating the Model**

To evaluate our trained model, we use the **best threshold** found earlier to make predictions on the test set (y_pred_best). These predictions are then evaluated using common metrics like precision, recall, F1 score, and accuracy.

- We use the **classification report** to display these metrics.
- We also print the **confusion matrix**, which gives insight into the number of true positives, true negatives, false positives, and false negatives.

```
# Evaluate the model using the best threshold

y_pred_best = (y_scores >= best_threshold).astype(int)

print(classification_report(y_test, y_pred_best))

print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred_best))

```

![Evaluation of the model using precision, recall, f1-score, and a confusion matrix.]({{ site.baseurl }}/assets/images/posts/precision-recall-curves/Model-output-1.png)

## **Conclusion**

In conclusion, the Precision-Recall Curve is a valuable tool for assessing machine learning models, particularly in scenarios involving imbalanced datasets like medical diagnosis. By exploring the relationship between precision and recall, using the F1-score if necessary, we are well-equipped to understand how a classification model performs.

## **Additional resources**

To master the use of evaluation metrics for imbalanced datasets, check out our book [“Machine learning with imbalanced data”](https://www.trainindata.com/p/imbalanced-data-myths-mistakes-solutions-book).

[![Imbalanced Data: Myths, Mistakes and Modern Solutions - book by Soledad Galli]({{ site.baseurl }}/assets/images/imbalanced-data-book-cover.jpg)](https://www.trainindata.com/p/imbalanced-data-myths-mistakes-solutions-book)
