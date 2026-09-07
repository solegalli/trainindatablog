---
layout: post
title: "The Complete Guide to Platt Scaling"
author: sole
description: "Learn about calibration in machine learning using Platt scaling. Find out how it works and how to apply it in Python using Scikit-learn."
excerpt: "Learn about calibration in machine learning using Platt scaling. Find out how it works and how to apply it in Python using Scikit-learn."
categories: [Data Science, Imbalanced Data, Machine Learning]
image: assets/images/posts/complete-guide-to-platt-scaling/Platt-scaling-banner.jpg
---

Platt scaling is a calibration technique used to convert the raw outputs of classification machine learning models into true probabilities. True probabilities accurately reflect the chance of an event occurring.

Machine learning models are widely used for decision-making in various fields like banking, healthcare, insurance policy claims, and more. The outputs of many algorithms, like Support Vector Machine (SVM), decision trees, and neural networks, are not directly interpretable as probabilities.

Platt scaling is a technique that can convert the model outputs or scores into well-calibrated probabilities between 0 and 1. In this article, we’ll understand the need for calibration and how Platt scaling works, along with hands-on examples in Python.

> To master probability calibration, enroll in our course [Machine Learning with Imbalanced Data](https://www.trainindata.com/p/machine-learning-with-imbalanced-data).

## **Why Is Calibration Essential in Machine Learning?**

In machine learning, [probability calibration](https://www.blog.trainindata.com/probability-calibration-in-machine-learning/) is the process of adjusting a model’s predictions to align with the actual likelihood of events. “Actual likelihood of events” refers to how often the event occurs in real life. For example, when a model predicts a 30% chance of rain, it really does rain 30% of the time — not more, not less.

Let’s take a look at why calibration is crucial in data science projects:

### **1. Reliable Probability Estimates:**

Classification models like XGBoost and SVM output probability estimates, which represent how confident the model is in its prediction. However, these probability estimates can be unreliable as the model could be overconfident or underconfident based on training data, fine-tuning, etc. Directly using the model outputs without adjusting them can lead to poor decision-making.

For example, let’s consider a classification model trained to predict if a patient is positive for tuberculosis (TB) based on lung CT scans. The model is overconfident and predicts an 80% chance of a patient having TB, while the actual chance is only 50%. The doctor may recommend aggressive treatment options, leading to increased stress and medical costs. With this example, we can see how using uncalibrated model outputs can cause adverse impacts.

### **2. Handling** [**Imbalanced Datasets:**](https://www.blog.trainindata.com/class-imbalance-in-machine-learning/)

If a dataset is skewed towards one class or category, classifier models may often produce biased probabilities. For example, consider a fraud detection dataset with 85% genuine transactions and 15% fraud. Models trained on this data tend to favor the majority class and may predict ‘non-fraud’ with overconfidence.

We can use calibration techniques like Platt scaling or Isotonic Regression to adjust the model scores and correct for the skewness in classification tasks.

To master probability recalibration, check out our course on how to [work with imbalanced datasets](https://www.trainindata.com/p/machine-learning-with-imbalanced-data).

[![Online course Machine Learning with Imbalanced data.]({{ site.baseurl }}/assets/images/posts/should-you-use-imbalanced-learn-in-2025/imbalanced-data-course.png)](https://www.trainindata.com/p/machine-learning-with-imbalanced-data)

### **3. Enhanced Model Interpretability**[**:**](https://www.blog.trainindata.com/class-imbalance-in-machine-learning/)

When the model results represent the actual likelihood of events, it increases interpretability and trust among the stakeholders. This is crucial for highly regulated industries like finance and healthcare.

### **4. Easier to benchmark and compare performance**[**:**](https://www.blog.trainindata.com/class-imbalance-in-machine-learning/)

In data science projects, we often need to compare various models to choose the most efficient option. If one model is overconfident while the other is underconfident, we cannot compare them against the same metrics. Calibration helps us standardize model outputs and makes it easier to compare performance, select score thresholds, and more.

## **What is Platt scaling?**

**Platt scaling** is a **probability calibration technique** that trains a logistic regression model with the classifier’s scores as the input and the actual dependent variable as the output, to learn the relation between them.

Researchers originally developed Platt scaling to transform the outputs of SVMs (Support Vector Machines). An SVM is a large-margin classifier algorithm that separates data into different classes using a hyperplane. The algorithm bases its output scores on the distance from the margin — the farther a data point is from the margin, the more confident the prediction. However, these scores do not map well to probabilities, so Platt scaling was introduced to address this issue. Since then, it has been extended to various classifier models like XGBoost, Random Forest, and neural networks.

Though Platt scaling was originally designed for binary classifiers, it can also be used for multi-class classification problems using the OVR (One vs Rest) technique. The sklearn Python library provides a `CalibratedClassifierCV` function that automatically takes care of this when you are training a calibration model. We’ll cover how to use it in the later sections.

## **How does Platt scaling work?**

Let’s understand how Platt scaling works step-by-step:

1. First, we use the trained base classification model to score the validation dataset. The validation dataset should not be used during training to prevent data leakage. Let’s consider the scores of the model as f(x) and the actual class label as y.

2. Next, we train a logistic regression model using the model’s scores `f(x)` as the input and the actual labels `y` as the target:

![Platt Scaling Formula]({{ site.baseurl }}/assets/images/posts/complete-guide-to-platt-scaling/Platt_scaling_formula.png)

During training, the parameters A and B will be learnt through **maximum likelihood estimation** (usually by minimizing log loss).

3. Now, the logistic regression model has learned how to **map its raw scores to actual probabilities** based on the patterns. This mapping can be applied to all the new predictions of the classifier model.

## **Implement Platt scaling in Python**

Let’s see how to implement Platt scaling in Python using the scikit-learn library.

The first step is to import all the necessary libraries and modules as shown in the snippet below.

```
import numpy as np import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
```

**1. Create Training & Testing sets**

For this example, let’s create a synthetic dataset for binary classification and train a Random Forest Classifier. We’ll use the `make_classification`  utility to generate a random dataset by providing the number of samples and class balance.

```
def create_dataset(weights):

    # returns arrays
    X, y = make_classification(
        n_samples=100000,
        n_features=20,
        n_informative=2,
        n_redundant=10,
        n_clusters_per_class=1,
        weights=[weights], # to balance (or not) the classes
        class_sep= 0.8, # how separated the classes are
        random_state=42)

    # transform arrays into pandas df and series
    X = pd.DataFrame(X)
    y = pd.Series(y)

    return X, y
```

```
X, y = create_dataset(weights=0.5)

```

For this example, I have generated a well-balanced dataset by providing a weight ratio of 0.5.

Now, let’s split the dataset into training sets and testing sets through random shuffling using the `train_test_split` function. The `test_size` parameter controls the size of the testing dataset (30% in our example):

```
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

X_train.shape, X_test.shape

```

### **2. Train a Random Forest model and plot a calibration curve**

Next, let’s train a Random Forest classification model on the training data with basic parameters, like 100 trees and depth of 2:

```
# Train a Random Forests
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=0,
    max_depth=2
).fit(X_train, y_train)

```

Now, we’ll use the trained model to score the test dataset as shown below. The scores obtained here will be the raw model output.

```
# score the test set
probs = rf.predict_proba(X_test)[:, 1]

```

Scikit-learn provides the `calibration_curve` function, which helps us check whether a model’s predicted probabilities are well calibrated. If they aren’t, we can apply Platt scaling or other calibration techniques to adjust them.

The `calibration_curve` function compares the predicted probabilities with the actual outcomes and returns data we can use to plot a calibration curve — a graph that shows how closely the predicted probabilities match the true likelihood of an event.

Let’s go ahead and plot the curve for our model:

```
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

def plot_calibration_curve(y_true, probs, bins, strategy):

    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, probs, n_bins=bins, strategy=strategy)

    max_val = max(mean_predicted_value)
    plt.figure(figsize=(8,10))
    plt.subplot(2, 1, 1)
    plt.plot(mean_predicted_value, fraction_of_positives, label='Random Forests')
    plt.plot(
        np.linspace(0, max_val, bins),
        np.linspace(0, max_val, bins),
        linestyle='--',
        color='red',
        label='Perfect calibration'
    )
    plt.xlabel('Probability Predictions')
    plt.ylabel('Fraction of positive examples')
    plt.title('Calibration Curve')
    plt.legend(loc='upper left')
    plt.subplot(2, 1, 2)
    plt.hist(probs, range=(0, 1), bins=bins, density=True, stacked=True, alpha=0.3)
    plt.show()

plot_calibration_curve(y_test, probs, bins=8, strategy='uniform')

```

The previous code returns the following plot:

![Random Forest Model's Calibration Curve]({{ site.baseurl }}/assets/images/posts/complete-guide-to-platt-scaling/RandomForest_Calibration_Curve-1024x587.png)

In the previous plot, the blue curve represents the calibration curve of the Random Forest, where the probability predictions represent the model’s predictions, and the “Fraction of positive examples”  indicates the actual proportion of positive samples in each bin of predicted probabilities.

Look at the graph when the Probability prediction is 0.4, then, check out that the fraction of positive examples is approximately 0.8. This means that the model predicted a 40% chance of positive class for samples in that bin, but in reality, **80% were positive**.

The model is underperforming in this region. The red dotted line indicates a perfectly calibrated model’s curve, where the model’s predictions are the same as the fraction of positive examples.

The **current Random Forest model is uncalibrated**, as its curve is highly deviant from the ideal calibration line.

> Advance your data science and machine learning skills with our [comprehensive expert led courses](https://www.trainindata.com/courses).

[![Advance your data science and machine learning skills with our comprehensive and expert led courses.]({{ site.baseurl }}/assets/images/posts/complete-guide-to-platt-scaling/Rectangle-7.png)](https://www.trainindata.com/courses)

### **3. Apply Platt scaling**

Let’s apply Platt scaling using the `CalibratedClassifierCV` function of sklearn to calibrate the probabilities. This function requires 3 input parameters:

- `base_estimator:` The base classifier you already trained (the Random Forest Classifier in our example)
- `cv=5:`Cross-validation strategy for calibration. It’s recommended to use 5-fold cross-validation to avoid overfitting during calibration.
- `method:` This parameter denotes the calibration method to use. We can use `'sigmoid'` for Platt scaling) or `'isotonic'` for non-parametric methods (more on this later).

Let’s fit a sigmoid classifier to the Random Forest model we trained previously:

```
# Sigmoid calibration
clf_sigmoid = CalibratedClassifierCV(rf, cv=5, method='sigmoid')
calibrated_model = CalibratedClassifierCV(
    base_estimator, method='sigmoid', cv=5)
```

### **4. Plot the Calibration curve after Platt scaling**

Now, let’s use the calibrated model to score the test dataset again:

```
clf_sigmoid.fit(X_test, y_test)
prob_sigmoid = clf_sigmoid.predict_proba(X_test)[:, 1]
```

Now, let’s plot the new calibration curve after applying Platt scaling:

```
plot_calibration_curve(y_test, prob_sigmoid, bins=8, strategy='uniform')
```

![Calibration Curve after Platt scaling]({{ site.baseurl }}/assets/images/posts/complete-guide-to-platt-scaling/Platt_scaling_Calibration_Curve-1024x543.png)

In the previous plot, the calibration curve (blue)  is more aligned with the reference line than what we had seen previously. Hence, the probability distribution is significantly more aligned with the actual outputs using the calibrated classifier.

This concludes our example use case of Platt scaling on Random Forest. Similarly, we can apply a sigmoid classifier to the outputs of other decision tree models like XGBoost or use it with SVM models.

## **Other Calibration Methods**

In addition to Platt scaling, data scientists commonly use other calibration methods like Isotonic Regression, Beta Calibration, and Temperature Scaling. Isotonic Regression fits a non-parametric function, offering more flexibility, though it requires more effort to train. Temperature Scaling works best for neural network models and integrates smoothly with frameworks like PyTorch. The following table compares different calibration methods based on factors such as flexibility and risk of overfitting.

![Table comparing the advantages and limitations of various probability calibration methods.]({{ site.baseurl }}/assets/images/posts/complete-guide-to-platt-scaling/Table-Probability-calibration-methods-comparison-1.png)

### **Advantages and limitations of Platt Scaling**

Platt scaling has many advantages over other methods, such as:

- Best suited for smaller-sized datasets due to the low risk of overfitting
- It uses a sigmoid function and has low  training complexity
- Easy to interpret

It also faces certain limitations:

- Limited flexibility: As Platt scaling is a parametric model, it may not be able to handle more complex probability mappings like non-monotonic trends.
- It can be slower and less accurate with multi-class problems compared to beta or temperature scaling methods

To learn additional calibration methods, check out our course [Machine Learning with Imbalanced Data](https://www.trainindata.com/p/machine-learning-with-imbalanced-data).

## **Conclusion**

Platt scaling is a simple and effective calibration technique for binary classification tasks like fraud detection, disease prediction, or sports forecasting. It can take raw, uninterpretable scores as input, like those from SVMs, and provide probabilistic outputs.

Calibrated models help data scientists make more accurate risk management and decision-making.

***Enjoyed our blog? Why not joining our*** [***newsletter***](https://www.trainindata.com/p/data-bites)***?* Join thousands of data scientists who get a single, powerful tip delivered every Monday. Our “Bite-Sized”** [**newsletter**](https://www.trainindata.com/p/data-bites) **cuts through the noise, giving you one actionable insight into a critical tool, emerging trend, or under-the-radar resource.** [**Subscribe now**](https://www.trainindata.com/p/data-bites) **and consistently learn what matters, without the overwhelm.**
