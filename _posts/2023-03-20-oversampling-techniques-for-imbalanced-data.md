---
layout: post
title: "Exploring Oversampling Techniques for Imbalanced Datasets"
author: sole
description: "Oversampling techniques for imbalanced datasets in Python."
excerpt: "Oversampling techniques for imbalanced datasets in Python."
categories: [Imbalanced Data, Machine Learning]
image: assets/images/posts/oversampling-techniques-for-imbalanced-data/cover_oversampling.png
---

Data drives the world of machine learning and neural networks, yet data quality can make or break a model’s performance.

Imbalanced datasets are those where one class has significantly fewer instances than the other(s), and they are a common occurrence in data science and data mining projects. Some examples include fraud detection and medical diagnosis datasets.

Traditional classification algorithms can perform badly when trained on imbalanced datasets. Oversampling can be a useful way of overcoming the class imbalance and hence improving the model’s performance.

This article will discuss various oversampling techniques, highlighting their advantages and limitations. We will also show how to implement oversampling in Python before training machine learning models to achieve improved performance.

So, let’s dive in.

> Discover the use of oversampling and undersampling through step-by-step tutorials, and implement them in Python, with our course [Machine Learning with Imbalanced Data](https://www.trainindata.com/p/machine-learning-with-imbalanced-data).

[![Online course Machine Learning with Imbalanced data.]({{ site.baseurl }}/assets/images/posts/class-imbalance-in-machine-learning/imbalanced-data-course-1024x576.png)](https://www.trainindata.com/p/machine-learning-with-imbalanced-data)

## What is Oversampling?

Oversampling is a data augmentation technique used when dealing with imbalanced datasets where the majority class dominates the minority class. It balances the distribution of classes by increasing the number of samples in the minority class.

![Diagram showing the process of random oversampling a dataset with 3 classes where one is minority]({{ site.baseurl }}/assets/images/posts/oversampling-techniques-for-imbalanced-data/oversampling-1024x576.png)

Oversampling increases the number of data points that belong to the minority class by duplicating existing ones or generating new ones by creating synthetic data.

### Why Do We Need Oversampling?

In an imbalanced dataset, the majority classes dominate and the minority classes are vastly outnumbered. This imbalance could lead to poor predictions by learning algorithms, particularly for the minority class.

When working with imbalanced datasets, we are usually interested in classifying the minority classes correctly. Hence, the cost of false negatives (i.e., failing to detect the minority class) is much higher than that of false positives (i.e., wrongly identifying a sample as belonging to the minority class).

Traditional machine learning models, like logistic regression and random forests, optimize for balanced metrics, and are naturally geared to working with balanced data. Oversampling can therefore help ensure that the classifier can identify the minority class accurately and minimize the cost of false negatives, by producing a balanced dataset.

### How Oversampling Differs from Undersampling

Oversampling and undersampling are resampling techniques for balancing imbalanced datasets, therefore resolving the imbalance problem. They are commonly used to generate suitable training data sets.

While oversampling adds new samples of the minority class, undersampling (or downsampling) reduces the number of samples in the majority class.

When deciding between these two approaches for balancing an imbalanced dataset, one should consider their advantages and limitations.

Undersampling can be effective when the majority class has many redundant or similar samples or when dealing with huge datasets. But it can also lead to a loss of information, resulting in biased models.

On the other hand, oversampling can be effective when the datasets are small and there are limited available samples of the minority class. But, it can also lead to overfitting due to data duplication or the creation of synthetic data that are not representative of the real data.

In the rest of the article, we’ll discuss various oversampling techniques.

> The use of oversampling methods like SMOTE has been recently re-evaluated and the results were not as promising as initially thought. Find the latest trends in our free booklet “[7 takes on imbalanced data](https://www.trainindata.com/p/7-takes-on-working-with-imbalanced-data)“.

## Random Oversampling

Random oversampling selects existing original samples from the minority class randomly and duplicates them to balance out the dataset. In other words, random oversampling duplicates data points from the minority class at random.

This method can be helpful when dealing with small datasets, as it helps increase the number of samples without the need to gather more data.

Let’s see how we can implement random oversampling. We will use the RandomOverSampler from the open-source package imbalanced-learn. So let’s make the necessary imports:

```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_blobs
from imblearn.over_sampling import RandomOverSampler
```

Now we create a toy dataset with 2 classes, where 80% of the observations belong to the majority class and 20% to the minority class. We use the `make_blobs` function from Scikit-learn:

```
X, y = make_blobs(
    n_samples=1600,
    centers=[(0, 0), (5, 5)],
    n_features=2,
    cluster_std=1.5)

X = pd.DataFrame(X, columns=['VarA', 'VarB'])
y = pd.Series(y)

X = pd.concat([
    X[y == 0],
    X[y == 1].sample(200, random_state=42)
], axis=0)

y = y.loc[X.index]
```

After that, we can go ahead and plot our toy dataset:

```
sns.scatterplot(
    data=X, x="VarA", y="VarB", hue=y, alpha=0.5
)
plt.title('Toy dataset')
plt.show()
```

The blue dots in the toy dataset represents the majority class, whereas the orange dots represent the minority class:

![Scatterplot showing data with 2 clusters of imbalanced classes.]({{ site.baseurl }}/assets/images/posts/overcoming-class-imbalance-with-smote/sample_raw_two_classes.png)

We’ll use the following code to select instances from the minority class in a random manner with replacement until the number of observations in the minority class equals the number of observations in the majority class:

```
ros = RandomOverSampler(
    sampling_strategy='auto',
    random_state=0,
)

X_res, y_res = ros.fit_resample(X, y)
```

When we apply `resample()`, the `RandomOverSampler()` duplicates at random data points from the minority class.

To better understand the result, let’s display the shape of the original data:

```
X.shape, y.shape
```

The below output tells us that 1000 samples and two features are in the original data, and 1000 data points belong to the target:

```
((1000, 2), (1000,))
```

Now, let’s evaluate the size of the over-sampled dataset:

```
X_res.shape, y_res.shape
```

The following output shows the size of the oversampled data. We see that it has 600 samples more than the original data:

```
((1600, 2), (1600,))
```

Let’s determine the number of minority class observations before and after oversampling:

```
y.value_counts(), y_res.value_counts()
```

The output shows the count of minority and majority class observations before and after oversampling. In the original data, there are 800 instances of the majority class and 200 of the minority. In the oversampled data, there are 800 data points of each class:

```
(0    800
 1    200
 dtype: int64,
 1    800
 0    800
 dtype: int64)
```

We can wrap up the demo by creating a a scatterplot of the oversampled dataset:

```
sns.scatterplot(
    data=X_res, x="VarA", y="VarB", hue=y_res, alpha=0.5
)

plt.title('Over-sampled dataset')
plt.show()
```

Below is our oversampled data set.

![Scatter plot showing an imbalanced data set with 2 classes after random oversampling the minority class.]({{ site.baseurl }}/assets/images/posts/oversampling-techniques-for-imbalanced-data/sample_ros.png)

Although there are more instances of the minority class after random oversampling, they may not appear more prominent when displayed because of the overlapping. However, you can see darker orange tones now for the minority class, which correspond to the duplicated instances.

We can now use the resampled dataset, `X_res` to train our machine learning classifiers. It is important to remember that while we train the models on the balanced dataset, we should test their performance on a test set with the original class distribution.

### Advantages of Random Sampling

The main advantage of random oversampling is its simplicity and ease of implementation. It does not require any complex algorithms or assumptions about the underlying distribution of the data. It can be applied to any dataset with a class imbalance and does not require prior knowledge about the dataset.

### Limitations of Random Sampling

Random oversampling may lead to overfitting, where the model becomes too specific to the training data and may not generalize well to new data.

The reason is that random oversampling does not add new information to the dataset. The new samples are generated by duplicating existing data. Hence, they do not provide further details about the minority class or classes, and instead, it might just increase their noise.

## Random Oversampling with Noise

Random oversampling with noise is a variation of random oversampling, where noise is added to the new samples generated during the oversampling process. The noise is introduced to prevent the exact replication of the minority class instances, which can lead to overfitting.

In this technique, instead of selecting exact replicas of the minority class instances, new data points are generated by introducing random noise into the existing minority class instances. A hyperparameter controls the amount of noise added and can be adjusted to obtain the desired noise level.

Adding noise can help diversify the minority class instances and reduce overfitting, preventing the model from relying too heavily on the exactly replicated minority class instances. You can quickly implement it using Python libraries such as imbalanced-learn.

For a Jupyter notebook with the full implementation, visit our [GitHub repository](https://github.com/solegalli/machine-learning-imbalanced-data/blob/master/Section-05-Oversampling/05-01-Random-Oversampling.ipynb).

## SMOTE (Synthetic Minority Oversampling Technique)

SMOTE, which stands for Synthetic Minority Oversampling Technique, is a popular oversampling method used to address class imbalance in machine learning.

The main idea behind the SMOTE algorithm is to generate synthetic data points of the minority class by interpolating between the minority class instances. In other words, SMOTE creates new data artificially.

To achieve this, SMOTE randomly selects a minority class instance and then finds its k nearest minority class neighbors. It then generates new synthetic instances by interpolating between the original minority instance and its k nearest neighbors.

Let’s see how we can implement SMOTE with imbalanced learn. First, we make the necessary imports:

```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_blobs
from imblearn.over_sampling import SMOTE
```

Now, let’s create a toy dataset with 2 classes, 80% of them belonging to a majority class:

```
X, y = make_blobs(
    n_samples=1600,
    centers=[(0, 0), (5, 5)],
    n_features=2,
    cluster_std=1.5)

X = pd.DataFrame(X, columns=['VarA', 'VarB'])
y = pd.Series(y)

X = pd.concat([
    X[y == 0],
    X[y == 1].sample(200, random_state=42)
], axis=0)

y = y.loc[X.index]
```

Now, lets implement SMOTE. We need to define specific parameters to over sample only the minority class, and define the number of k-nearest neighbors for the interpolation:

```
sm = SMOTE(
    sampling_strategy='auto',
    random_state=0,
    k_neighbors=5,
)

X_res, y_res = sm.fit_resample(X, y)
```

With `resample()` we’ve created the synthetic data.

Let’s now compare the original and new data sizes. Let’s display the size of the original data:

```
X.shape, y.shape
```

Here’s the output indicating 1000 observations and two features in the X dataset and 1000 corresponding labels in the y dataset.

```
((1000, 2), (1000,))
```

To find the size of oversampled data, we use the following code.

```
X_res.shape, y_res.shape
```

It leads us to the following output, where we see that the final training data contains 1600 observations:

```
((1600, 2), (1600,))
```

We want to identify the number of minority class observations before and after applying SMOTE.

```
y.value_counts(), y_res.value_counts()
```

The below output tells us that in the original dataset, there are 800 observations of class 0 and 200 observations of class 1. After applying SMOTE, the resampled dataset now has 800 observations of each class:

```
(0    800
 1    200
 dtype: int64,
 1    800
 0    800
 dtype: int64)
```

We can now use the following code to generate a scatter-plot for our oversampled dataset.

```
sns.scatterplot(
    data=X_res, x="VarA", y="VarB", hue=y_res, alpha=0.5
)

plt.title('Over-sampled dataset')
plt.show()
```

![Scatter plot showing a dataset with 2 imbalanced classes, where synthetic data points were created for the minority class using SMOTE.]({{ site.baseurl }}/assets/images/posts/overcoming-class-imbalance-with-smote/sample_smote.png)

So you can notice that there are now new observations in between the original data points of the minority class.

### Advantages of SMOTE

SMOTE can generate new samples based on existing ones, which helps to add more information to the dataset to improve model performance.

### Limitations of SMOTE

One of the main drawbacks of SMOTE is that it may introduce noise with the synthetic instances, especially when the number of nearest neighbors is set too high. Additionally, SMOTE may not work well on tightly clustered minority class instances or when there are few instances in the minority class.

> Before reading any further, get familiar with the latest thoughts on the use of SMOTE with our [7 takes on imbalanced data](https://www.trainindata.com/p/7-takes-on-working-with-imbalanced-data) FREE booklet.

## SMOTE Variants

Several SMOTE variants are available to address these limitations.

### Borderline-SMOTE

One of the key limitations of SMOTE is that it generates synthetic samples along the line connecting the minority class sample to its nearest neighbors. This approach can lead to the misclassification of synthetic samples as the majority class, especially when the minority class sample is close to the decision boundary.

Borderline-SMOTE addresses this issue by only generating synthetic samples for minority-class samples close to the decision boundary. Under the hood, Borderline SMOTE uses templates to create the synthetic data observations from the minority, for which the majority of the neighbors are from the opposite class. Then, it interpolates between these instances and their minority class neighbors.

### SVM-SMOTE

SVM-SMOTE is an alternative to Borderline SMOTE, where the data points that are used to create the synthetic data are the support vectors of a support vector machine separating the classes.

The synthetic data is created by interpolating between each minority class support vector and its nearest neighbors of the minority class. Hence, it creates more data points at the decision boundary.

### Other SMOTE variants

In addition to the SMOTE variants mentioned above, there are more SMOTE alternatives like K-means SMOTE and variations that are also suitable for discrete and categorical variables, like SMOTE-N and SMOTE-NC. If you want to know more about these methods, as well as, their advantages and limitations and how to implement them in Python, Check out our course [Machine Learning with Imbalanced Data](https://www.trainindata.com/p/machine-learning-with-imbalanced-data).

## Adaptive Synthetic Sampling (ADASYN)

ADASYN is an alternative oversampling technique that aims to address the issue of generating synthetic samples in regions of the feature space that are closer to the decision boundary. It works by developing more synthetic samples for minority class samples that are more difficult to learn, i.e., those closer to the decision boundary.

ADASYN uses as templates for the synthetic data samples from the minority class if some of their closest neighbours are from opposite classes. The more neighbours from the opposite class, the more likely it is to be used as template. After selecting the templates, it then generates the examples by interpolation between the template and the closest neighbors from its same class.

## Wrap-up

Addressing class imbalance can be critical for developing accurate and robust machine learning models. Oversampling techniques, such as random oversampling, SMOTE, and their variants, have proven effective in dealing with imbalanced datasets.

However, selecting the best oversampling process requires careful consideration of performance metrics, pros and cons, and data preparation.

With the help of Python libraries and frameworks, implementing oversampling techniques has become more accessible.

Need help figuring out where to start? The [Machine Learning with Imbalanced Data](https://www.trainindata.com/p/machine-learning-with-imbalanced-data) course comprehensively guides you through all the methods with step-by-step tutorials!
