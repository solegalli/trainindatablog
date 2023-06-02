---
layout: post
title: "Cost-Sensitive Learning: Beyond the Accuracy in Imbalanced Classification"
author: sole
description: Find out what cost-sensitive learning is and how to implement it with Python.
excerpt: Find out what cost-sensitive learning is and how to implement it with Python.
categories: [Imbalanced data, Python, Machine learning]
image: assets/images/posts/imbalanced/cover_csl.gif
---

In the realm of machine learning, the primary objective for most models is to optimize accuracy, 
or in other words, to minimize the overall error rate. The error rate is the percentage of observations 
that are misclassified, regardless of their class.

Classification algorithms, like logistic regression, decision trees or support vector machines (SVM),  
assume that all misclassification errors carry the same cost. They were designed to work with datasets 
where the class distribution is homogeneous, that is, all classes are equally represented.

However, in certain real-world scenarios like fraud detection or healthcare, where there is a significant 
class imbalance, the cost of misclassifying rare occurrences tends to be significantly higher.

In the case of fraud detection, misclassifying a fraudulent transaction as legitimate can 
result in financial losses and damage to a company's reputation. In healthcare, misclassifying 
a patient's condition as non-critical or benign when it is actually severe can have 
detrimental effects on the individual's health and well-being. Therefore, there are 
higher costs associated with these misclassifications, which emphasizes the importance 
of accurately identifying and classifying such instances.

This is where cost-sensitive learning comes into play, allowing us to address the class 
imbalance problem and enhance the performance of classifiers by considering the varying 
costs associated with different types of misclassifications.

*To learn more about cost-sensitive learning and other learning techniques to tackle 
imbalanced data, check out our course 
[Machine Learning with Imbalanced Data](https://www.trainindata.com/p/machine-learning-with-imbalanced-data).*

## Understanding Cost-Sensitive Learning

Cost-sensitive learning is a branch of machine learning that acknowledges the varying costs 
associated with misclassification errors in imbalanced datasets. It focuses on modifying 
learning algorithms optimization functions so that they minimize the overall cost of misclassification 
instead of the overall error rate.

By assigning specific costs to different types of misclassifications, cost-sensitive learning 
methods allow the machine learning models to prioritize the minority class and achieve better 
performance in critical classification problems.

Then the question is, How do we derive the cost of misclassification for a particular 
classification task? And how do the algorithms incorporate the misclassification costs into 
their optimization functions?

Let’s break this down.


## Different Types of Costs

In cost-sensitive learning, misclassification costs can be categorized into four types: 

- false positive (cost of misclassifying the positive class as negative), 

- false negative (cost of misclassifying the negative class as positive), 

- true positive (correctly classifying the positive class), and 

- true negative (correctly classifying the negative class). 


You are probably familiar with these already since we can obtain them through a confusion matrix 
derived from the model predictions.

To illustrate the different costs, let’s consider a fraud detection model: misclassifying a 
fraudulent transaction as legitimate (false negative) can result in significant financial 
losses, whereas flagging a legitimate transaction as fraudulent (false positive) can inconvenience 
customers, for example by delaying their application, but most likely, they will get what they 
need. As you can see, there are different costs associated with the different misclassifications.

A cost matrix is like a confusion matrix, but instead of having the percentage of observations 
classified correctly or wrongly, it contains the costs associated with each correct and incorrect 
classification. Hence, the cost matrix captures these different costs and guides the learning 
process accordingly.

But how do we obtain a cost matrix?

## Obtaining the Cost 

Acquiring accurate cost information is crucial for effective, cost-sensitive learning. Having 
said this, determining the cost is hard. While finding the cost of incurring financial losses may 
be straightforward, determining the cost associated with "customer inconvenience", or the lack of 
well-being of a patient is much harder.

To determine the accurate costs associated with misclassifications, we need to work with domain 
experts and different stakeholders, analyze historical data, and leverage data mining techniques 
to identify patterns within the dataset that highlight the true costs of different misclassification 
scenarios. And even when we do so, determining the cost can still be a hard thing to do.

With these costs, we can perform cost-sensitive classification by assigning class weights proportional 
to the misclassification costs during the training of the machine learning algorithm. This ensures 
that the classifier is biased towards minimizing the total cost of misclassifications and not the 
error rate.

I am not going to discuss how to obtain suitable costs any further because it really depends on the 
domain and what is at stake when the model classifies rare instances incorrectly. Instead, I am 
going to focus on how we can optimize the cost of utilizing machine learning, which is something 
that at least we, as data scientists, can do.

We can, at least in practice, optimize the misclassification costs by utilizing grid or random 
search and treating the costs as hyperparameters in the models. We’d introduce varying costs to 
the minimization function of the algorithm and then evaluate a certain performance metric (more 
on metrics at the end of the article). And we would select the costs that return the best value f
or the performance metric.

## Introducing costs in the minimization function.

So we talked about modifying the algorithms so that they minimize the cost instead of the error 
rate. But how do we do that?

Each algorithm has its own loss function to minimize. In logistic regression, the algorithm 
normally minimizes the following loss function:

```
-y log(h(x)) – (1-y) log (1- h(x))
```

where h is `1/(1+ e^βTx)`.

When we introduce the costs, the algorithm will now minimize the following function:

```
- w1 y log(h(x)) – w0 (1-y) log (1- h(x))
```

In the case of decision tree-based algorithms, the cost is introduced in the various minimization 
functions, such us the gini, the entropy or the misclassification rate, as shown in the image:

![loss function with costs in decision tree based models]({{ site.baseurl }}/assets/images/posts/imbalanced/tree-cost-function.png)
 

The formulas are taken from the [Scikit-learn documentation](https://scikit-learn.org/stable/modules/tree.html#classification-criteria).

Now that we understand how we modify the algorithms to introduce cost, let’s carry out cost-sensitive 
learning in Python.

## Implementing Cost-Sensitive Learning in Python 

We can easily implement cost sensitive learning in Python using Scikit-learn. There are 2 main ways 
to do that. One is introducing class weights. And the second one is introducing sample weights. 
Let’s see how we can do that.

We’ll begin by importing pandas and numpy, the LogisticRegression from Scikit-learn and a function 
to evaluate the ROC-AUC:


```
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
```

Next, we will load a dataset:

```
data = pd.read_csv('../kdd2004.csv').sample(10000)
```

We find the fraction of observations in each of the two classes in the target variable:

```
data.target.value_counts() / len(data) 
```

In the result, we see that there is a different class distribution, where -1 
dominates the dataset:

```
-1    0.9903
 1    0.0097
Name: target, dtype: float64
```   

The class labels in the former example are -1 and 1, where 1 is the rare occurrence and 
the one that we are interested in predicting accurately. In other words, the cost of 
misclassifying 1 is bigger than the cost of misclassifying -1.

Let's separate the data into training examples and a test set:

```
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(labels=['target'], axis=1),  # drop the target
    data['target'],  # just the target
    test_size=0.3,
    random_state=0,
)
````

### Using class_weight

We’ll begin by implementing cost-sensitive learning with the class_weight parameter offered by 
Scikit-learn models. In this example, we use logistic regression, but class_weight is also available 
in bagging and boosting algorithms like random forests and gradient boosting machines.

We simply initialize the cost, which is given by the weights, when we set up the transformer:

``` 
def run_Logit(X_train, X_test, y_train, y_test, class_weight):    
    logit = LogisticRegression(
        penalty='l2',
        solver='newton-cg',
        random_state=0,
        max_iter=10,
        n_jobs=4,
        class_weight=class_weight # weights / cost
    )
    
    logit.fit(X_train, y_train)

    print('Train set')
    pred = logit.predict_proba(X_train)
    print('roc-auc: {}'.format(roc_auc_score(y_train, pred[:, 1])))

    print('Test set')
    pred = logit.predict_proba(X_test)
    print('roc-auc: {}'.format(roc_auc_score(y_test, pred[:, 1])))
```

Now that we set up the function, we can train models with different costs, or with no costs associated 
to the misclassifications. We begin by training a model without cost-sensitive learning. This will 
give us the baseline performance.


 ``` 
run_Logit(X_train,
          X_test,
          y_train,
          y_test,
          class_weight=None)
```

Below, we see the performance of a Logistic regression trained on an imbalanced dataset:

``` 
Train set roc-auc: 0.9192043838780551 
Test setroc-auc: 0.8979334677419354
```

Next, we train and evaluate the performance of a model trained with cost-sensitive learning. A general 
heuristic for the class weighting is to utilize the inverse of the class distribution of the dataset, 
or in other words, the inverse of the class imbalance ratio.

Scikit-learn does that automatically, when we set the class_weight to “balanced”:

```
run_Logit(X_train,
          X_test,
          y_train,
          y_test,
          class_weight='balanced')
```   

We see that the performance of the model improved with the weights:

```
Train set roc-auc: 0.9925445596049605 
Test set roc-auc: 0.9620855734767024
``` 

We can also test a different cost. In fact, we can pass the cost associated with each class 
in a dictionary, like this: 

``` 
run_Logit(X_train,
          X_test,
          y_train,
          y_test,
          class_weight={-1:1, 1:10})
``` 

Again, we see that the performance of the model trained with costs is better than the baseline model, 
trained without cost-sensitive learning:

```
Train set roc-auc: 0.9617874072272288 
Test set roc-auc: 0.9445704525089607 
```

In both cases, we see that implementing cost-sensitive learning does improve the performance of the 
logistic regression.

In this example, we optimized the cost of a binary classification task. But we can do the same for
multi-class classification. If we set the class_weight to "balanced," we will be using the imbalance 
ratio of all classes as the cost. Alternatively, we can pass a dictionary with the cost associated 
with each class, as we did in the last code block.

### Using sample_weight

In the former demo, we used class weights. That is, we assigned a misclassification cost to each one 
of the class labels.

Instead, we can fine-tune the learning even further by assigning weights to each individual sample. 
This way, we could attribute higher costs, such as more expensive loans or car claims, in order to 
penalize them more if they are fraudulent.

We begin by setting up a function:

```
def run_Logit(X_train, X_test, y_train, y_test, sample_weight):
    
    logit = LogisticRegression(
        penalty='l2',
        solver='newton-cg',
        random_state=0,
        max_iter=10,
        n_jobs=4,
    )
    
    # costs are passed here
    logit.fit(X_train, y_train, sample_weight=sample_weight)

    print('Train set')
    pred = logit.predict_proba(X_train)
    print('roc-auc: {}'.format(roc_auc_score(y_train, pred[:, 1])))

    print('Test set')
    pred = logit.predict_proba(X_test)
    print('roc-auc: {}'.format(roc_auc_score(y_test, pred[:, 1])))
```

Next, we evaluate the performance of an algorithm trained on the imbalanced data without cost-sensitive 
learning:

```
run_Logit(X_train,
          X_test,
          y_train,
          y_test,
          sample_weight=None)
```

In the following output, we see the performance of a logistic regression without cost-sensitive 
learning. This gives us the baseline performance:

```
Train set roc-auc: 0.9192043838780551 
Test set roc-auc: 0.8979334677419354 
```

Now, I create a vector of weights for each of the observations. The vector should have the same 
length as the target. My vector is going to be very simple, but you could have a very complicated 
vector of individual weights for each one of your samples if you wanted.

```
run_Logit(X_train,
          X_test,
          y_train,
          y_test,
          sample_weight=np.where(y_train==1,99,1))
```

We see that cost-sensitive learning improved the performance of the model:

```
Train set roc-auc: 0.992609819428047 
Test set roc-auc: 0.9542450716845878
```

The aim of this demo is to show you how to implement cost-sensitive learning using Scikit-learn. 
I kept it very simple, and I compared only the performance metric given by the ROC-AUC. You’d 
probably want to carefully select the metric that works best for your use case, and make plots 
instead of obtaining single values, like plotting a ROC curve and precision and recall curves.


## More on Cost sensitive learning

In this article, we implemented cost-sensitive learning by modifying the loss function to introduce a cost to 
the misclassification of different classes. What if the loss function can’t be modified to introduce costs?

There is an alternative algorithm called Metacost that makes cost-insensitive algorithms cost-sensitive. 
We are not going to describe it any further because there isn't, unfortunately, an open-source implementation 
of this algorithm yet. But you can read more about it in the [original article](https://dl.acm.org/doi/10.1145/312129.312220) 
or in our course on [Machine Learning with Imbalanced Data](https://www.trainindata.com/p/machine-learning-with-imbalanced-data).


## Evaluation and Performance Metrics 

When dealing with imbalanced datasets, traditional performance metrics like accuracy alone may not provide an 
accurate representation of the classifier's effectiveness. Instead, evaluation metrics such as precision, recall, 
F1-score, and the area under the receiver operating characteristic curve (ROC curve) are more suitable. These 
metrics offer a comprehensive view of the classifier's performance, especially in scenarios where the minority 
class is of primary interest.


## Mitigating Overfitting and Generalization 

As with any machine learning problem, overfitting must be avoided during cost-sensitive learning. Overfitting 
occurs when the classifier becomes too specialized for the training set, leading to poor generalization on unseen 
data. To prevent overfitting, techniques such as cross-validation and regularization can be applied. Cross-validation 
helps estimate the model's performance on unseen data by partitioning the dataset into multiple subsets for 
training and testing. Regularization techniques like L1 or L2 regularization can control the complexity of the 
model, reducing the risk of overfitting.


## Alternatives to Cost-Sensitive Learning for imbalanced data

Cost-sensitive learning is one way to tackle class imbalances. But there are other machine-learning techniques. 
Resampling is a common data preprocessing step that can be implemented before training a cost-insensitive algorithm.

To mitigate class imbalance, techniques like oversampling (replicating minority class examples) or undersampling 
(removing examples from the majority class) can be employed. Resampling the training data helps create a balanced 
dataset and facilitates better learning of cost-insensitive machine learning models.


## Further Reading and Additional Resources

- Elkan, C. (2001). [The foundations of cost-sensitive learning](https://cseweb.ucsd.edu/~elkan/rescale.pdf). In Proceedings of the 17th international joint conference of artificial intelligence (pp. 973–978). Seattle: Morgan Kaufmann.

- Ling, C.X., Sheng, V.S. (2011). [Cost-Sensitive Learning](https://doi.org/10.1007/978-0-387-30164-8_181). In: Sammut, C., Webb, G.I. (eds) Encyclopedia of Machine Learning. Springer, Boston, MA.

- Domingos, P. 1999. [MetaCost: A general method for making classifiers cost-sensitive](https://dl.acm.org/doi/10.1145/312129.312220). In Proceedings of the Fifth International Conference on Knowledge Discovery and Data Mining, 155-164, ACM Press

- [Machine Learning with Imbalanced Data](https://www.trainindata.com/p/machine-learning-with-imbalanced-data) - online course