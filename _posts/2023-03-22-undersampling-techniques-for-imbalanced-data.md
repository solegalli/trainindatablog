---
layout: post
title: "The Role of Undersampling in Tackling Imbalanced Datasets in Machine Learning"
author: sole
description: "Undersampling techniques for imbalanced datasets in Python."
excerpt: "Undersampling techniques for imbalanced datasets in Python."
categories: [Imbalanced Data, Machine Learning]
image: assets/images/posts/undersampling-techniques-for-imbalanced-data/cover_undersampling.png
---

Machine learning algorithms are becoming increasingly popular for data mining and predictive analytics. However, traditional machine learning models, like random forest and logistic regression, can suffer from poor performance when dealing with imbalanced datasets due to skewed class distributions, leading to classification problems.

Imbalanced datasets are a common challenge in many real-world scenarios, from fraud detection to medical diagnosis, and can significantly impact the accuracy and reliability of our classification models. To address this problem, data science professionals use various resampling methods for tackling imbalanced classification in machine learning, including undersampling, [oversampling](https://www.blog.trainindata.com/oversampling-techniques-for-imbalanced-data/), and [cost-sensitive learning](https://www.blog.trainindata.com/cost-sensitive-learning-for-imbalanced-data/).

Undersampling involves removing samples from the majority class to balance the dataset. Today, we’ll review some undersampling methods and tutorials and provide valuable resources.

For more information about undersampling and oversampling methods, and step-by-step tutorials to implement them in Python, check out our book [Machine Learning with Imbalanced Data](https://www.trainindata.com/p/imbalanced-data-myths-mistakes-solutions-book).

[![Imbalanced Data: Myths, Mistakes and Modern Solutions - book by Soledad Galli]({{ site.baseurl }}/assets/images/imbalanced-data-book-cover.jpg)](https://www.trainindata.com/p/imbalanced-data-myths-mistakes-solutions-book)

**A quick note before we start:** undersampling does not make a model better at discriminating between classes. What it does is shift the model's decision boundary so that, at the default classification threshold of 0.5, we make more cost-sensitive decisions — that is, we correctly flag a larger proportion of the minority class, which is usually the class we care about the most. You can achieve this exact same effect by training on the original, unmodified data and simply adjusting the classification threshold afterward, without undersampling anything at all. We'll come back to this point throughout the article.

## What is Undersampling?

Undersampling is a technique that can reduce the size of the majority class in a dataset. It involves removing samples from the majority class until it matches the size of the minority class or until specific criteria are met.

We can divide undersampling algorithms into two groups based on their logic: fixed undersampling and cleaning methods.

### Fixed Undersampling Methods

Fixed undersampling algorithms remove samples from the majority class until they match the size of the minority class (or until a preferred balance ratio is achieved). They result in a more balanced class distribution within the dataset. The primary method in this category is random undersampling, but there are others, like NearMiss and the instance hardness threshold.

### Cleaning Undersampling Methods

The remaining undersampling algorithms belong to this class. They aim to remove samples from the majority class based on some criteria, which change with the methodology. Hence, when using any of these algorithms, we won’t necessarily obtain a dataset with as many observations from the majority class as those from the minority class. Instead, we obtain datasets where problematic samples from the majority class are removed. The number of removed observations depends on the nature of the data and the method used for undersampling.

Relevant cleaning undersampling methods are Tomek links, edited nearest neighbors and their variants, and condensed nearest neighbors.

## How Does Undersampling Differ from Oversampling?

While undersampling involves removing examples from the majority class to reduce class imbalance, oversampling involves adding examples to the minority class to achieve the same goal.

In oversampling, we can create synthetic examples using Synthetic Minority Over-sampling Technique (SMOTE) or Adaptive Synthetic Sampling (ADASYN). With random oversampling, we can also resample the minority class with replacement.

### When to Use Oversampling?

The choice between oversampling and undersampling techniques depends on the data at hand. Oversampling can be helpful when we have limited data and can’t afford to discard observations.

However, oversampling can also result in overfitting, where the model learns the noise and variability of the minority class and performs poorly on new examples.

### When to Use Undersampling?

Undersampling, on the other hand, is useful when we have large training data sets and the majority class is well represented so that we do not lose information when we remove some of the examples. By reducing the size of the training set, we save storage and speed model training times.

On the downside, undersampling can also result in the loss of valuable information from the majority class, which may lead to a less accurate model.

## Undersampling Methods

A complete list of undersampling methods includes:

- Random Undersampling
- Condensed Nearest Neighbours
- Tomek Links
- One Sided Selection
- Edited Nearest Neighbours (ENN)
- Repeated ENN
- AllKNN
- Neighbourhood Cleaning Rule
- Near Miss
- Instance Hardness Threshold

We’ll now review various undersampling methods and show how to implement them in Python utilizing the open-source Python library imbalanced-learn.

## Random Undersampling

Random undersampling randomly removes instances from the majority class to balance out the dataset. It eliminates a subset of data points from the majority class to create a more balanced dataset.

Let’s implement random undersampling utilizing the `RandomUnderSampler` from imbalanced learn. We’ll begin by importing the necessary libraries, classes, and functions:

```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from imblearn.under_sampling import RandomUnderSampler
```

Let’s now create a toy dataset for the demonstration using the `make_classification` function from Scikit-learn. The code following snippet returns the dataframe X with the features and the Series y with the labels:

```
X, y = make_classification(n_samples=1000,
                       n_features=2,
                       n_redundant=0,
                       n_clusters_per_class=1,
                       weights=[0.99],
                       class_sep=2
                       random_state=1)

X = pd.DataFrame(X, columns =['varA', 'varB'])
y = pd.Series(y)
```

Let’s now make a scatter plot to visualize the class distribution:

```
sns.scatterplot(
    data=X, x="varA", y="varB", hue=y
)
plt.title('Separation: 2'))
plt.show()
```

Below is our original dataset, where blue dots indicate the majority class and orange dots indicate the minority class. Of 1,000 samples, 983 are blue dots, and 17 are orange dots.

![Scatterplot showing toy dataset with 2 clusters of imbalanced classes]({{ site.baseurl }}/assets/images/posts/undersampling-techniques-for-imbalanced-data/sample_raw_us_blog.png)

Now we’ll use the `RandomUnderSampler` to obtain a final class distribution of 50:50. We need to set the parameter `sampling_strategy` to auto for this purpose.

```
rus = RandomUnderSampler(
    sampling_strategy='auto',
    random_state=0,
)

X_resampled, y_resampled = rus.fit_resample(X, y)
```

From the previous output, `X_resampled` contains the resampled dataset, and `y_resampled` has the resampled target.

Let’s go ahead and compare the size of the original and resampled datasets. The following code will identify the original data size:

```
X.shape, y.shape
```

The output indicates that the original data has 1000 samples with two features.

```
((1000, 2), (1000,))
```

We want to know the size of undersampled data:

```
X_resampled.shape, y_resampled.shape
```

The below output indicates that after applying random undersampling to the original dataset, the undersampled dataset has 34 observations, ensuring a 50:50 balancing ratio as we expected.

```
((34, 2), (34,))
```

We originally had 17 observations from the minority class. We, therefore, obtained a dataset containing only 17 observations of the majority class.

Now, we can plot the undersampled data and compare it with our previous original data plots.

```
sns.scatterplot(
    data=X_resampled, x="varA", y="varB", hue=y_resampled
)
plt.title('Undersampled dataset')
plt.show()
```

Here you can see an equal number of blue and orange dots resulting from random undersampling.

![Scatter plot showing class distribution after random undersampling]({{ site.baseurl }}/assets/images/posts/undersampling-techniques-for-imbalanced-data/sample_rus.png)

In our [GitHub repository](https://github.com/solegalli/machine-learning-imbalanced-data/blob/master/Section-05-Undersampling/01-Random-Undersampling.ipynb) of random undersampling, you’ll find a few more advanced applications, such as changing the balancing ratio, loading data, handling imbalanced targets, and comparing machine learning performance.

## Tomek Links

Tomek Links focuses on cleaning up data at the decision boundary. So what is a Tomek Link? If two samples are nearest neighbors and from a different class, they are Tomek Links.

Tomek Links undersampling removes the Tomek Link from the majority class in its more conservative form. In its more aggressive variant, it removes the entire Tomek Link. The underlying assumption is that samples that are closest neighbors yet from a different class contribute noise to the training data.

Let’s implement Tomek Links in Python using imbalanced learn. Again, we’ll begin by importing the necessary libraries, functions, and classes:

```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from imblearn.under_sampling import TomekLinks
```

Next, we’ll create a toy training set:

```
X, y = make_classification(n_samples=1000,
                       n_features=2,
                       n_redundant=0,
                       n_clusters_per_class=1,
                       weights=[0.99],
                       class_sep=2,
                       random_state=1)


X = pd.DataFrame(X, columns =['varA', 'varB'])
y = pd.Series(y)
```

Next, we set up the `TomekLinks()` to remove samples identified as Tomek Links. By setting the parameter `sampling_strategy` to ‘auto,’ we will remove only the observations from the majority class in the Tomek Link:

```
tl = TomekLinks(sampling_strategy='auto')

X_resampled, y_resampled = tl.fit_resample(X, y)
```

Let’s now compare the training data size before and after the undersampling. We find the original data size:

```
X.shape, y.shape
```

The below output indicates that the original dataset has 1000 samples and two features, and the target variable has 1000 values.

```
((1000, 2), (1000,))
```

The following input will give us the undersampled data size:

```
X_resampled.shape, y_resampled.shape
```

Below is the size of the undersampled dataset. The algorithm has removed six samples from the majority class.

```
((994, 2), (994,))
```

Let’s now find out the class distribution in the original data:

```
y.value_counts()
```

The below output indicates that there are 983 samples in the majority class labeled as 0 and 17 in the minority class labeled as 1.

```
0    983
1     17
dtype: int64
```

Let’s repeat the procedure for the resampled data:

```
y.value_counts()
```

As expected, we have six fewer samples in the majority class.

```
0    977
1     17
dtype: int64
```

After applying Tomek Links, let’s plot both data sets. First, we plot the original training set:

```
sns.scatterplot(
    data=X, x="varA", y="varB", hue=y
)

plt.title('Original dataset')
plt.show()
```

![Scatter plot showing 2 imbalanced classes with high overlap between classes.]({{ site.baseurl }}/assets/images/posts/undersampling-techniques-for-imbalanced-data/sample_tl1.png)

And now, we create a scatterplot for the undersampled dataset:

```
sns.scatterplot(
    data=X_resampled, x="varA", y="varB", hue=y_resampled
)

plt.title('Undersampled dataset')
plt.show()
```

![Scatter plot showing classes after applying cleaning with Tomek Links]({{ site.baseurl }}/assets/images/posts/undersampling-techniques-for-imbalanced-data/sample_tl2.png)

The closer the samples from the majority class to the decision boundary, the more observations will be removed with Tomek Links.

## Edited Nearest Neighbors (ENN) Undersampling

Unlike random undersampling, Edited Nearest Neighbors (ENN) is a selective method that removes examples from the majority class whose neighbours, identified by the k-nearest neighbor (k-NN) algorithm, belong to a different class.

The ENN algorithm works as follows:

- First, the k-NN algorithm is used to identify examples in the majority class whose neighbours are from a different class.
- Next, it removes these examples.

In other words, ENN removes observations from the majority class if the majority of their neighbours are from a different class. These are data points likely to be missclassified by a classifier.

ENN can reduce noise in the data by eliminating overlapping data points often present at the decision boundary between the classes. The resulting dataset is then used for training the machine learning model.

There are a few things to consider when using the KNN algorithm though:

- Traditional distances like the Euclidean distance, between discrete or categorical variables are not ideal.
- KNNs do not scale. It may take long run times when using cross-validation or massive datasets.
- The Nearest neighbors algorithms are distance-based algorithms. Thus, the dataset requires scaling.

Let’s implement ENN in Python using imbalanced learn. We’ll begin by importing the necessary libraries, functions, and classes:

```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from imblearn.under_sampling import EditedNearestNeighbours
```

Next, we’ll create a toy training set:

```
X, y = make_classification(n_samples=1000,
                       n_features=2,
                       n_redundant=0,
                       n_clusters_per_class=1,
                       weights=[0.99],
                       class_sep=2,
                       random_state=1)


X = pd.DataFrame(X, columns =['varA', 'varB'])
y = pd.Series(y)
```

Next, we set up the `EditedNearestNeighbours()`. By setting the sampling strategy to ‘auto,’ we will remove only the observations from the majority class:

```
enn = EditedNearestNeighbours(
    sampling_strategy='auto',
    n_neighbors=3,
    kind_sel='all')

X_resampled, y_resampled = enn.fit_resample(X, y)
```

Let’s now compare the training data size before and after the undersampling. We find the original data size:

```
X.shape, y.shape
```

The below output indicates that the original dataset has 1000 samples and two features, and the target variable has 1000 values.

```
((1000, 2), (1000,))
```

The following input will give us the undersampled data size:

```
X_resampled.shape, y_resampled.shape
```

Below is the size of the undersampled dataset. We see that the algorithm has removed only five samples from the majority class:

```
((978, 2), (978,))
```

Let’s now find out the class distribution in the original data:

```
y.value_counts()
```

The below output indicates that there are 983 samples in the majority class labeled as 0 and 17 in the minority class labeled as 1.

```
0    983
1     17
dtype: int64
```

Let’s repeat the procedure for the resampled data:

```
y.value_counts()
```

As expected, we have five fewer samples in the majority class:

```
0    978
1     17
dtype: int64
```

After applying Edited Nearest Neighbours, let’s plot both data sets. First, we plot the original training set:

```
sns.scatterplot(
    data=X, x="varA", y="varB", hue=y
)

plt.title('Original dataset')
plt.show()
```

![Scatter plot showing 2 imbalanced classes with high class overlap.]({{ site.baseurl }}/assets/images/posts/undersampling-techniques-for-imbalanced-data/sample_enn1.png)

And now, we create a scatterplot for the undersampled dataset:

```
sns.scatterplot(
    data=X_resampled, x="varA", y="varB", hue=y_resampled
)

plt.title('Undersampled dataset')
plt.show()
```

![Scatter plot showing imbalanced classes after cleaning with edited nearest neighbors.]({{ site.baseurl }}/assets/images/posts/undersampling-techniques-for-imbalanced-data/sample_enn2.png)

We can now see more orange examples among the blue dots, corresponding to those that were previously covered by examples from the majority class.

ENN also has multiple variants: repeated ENN (RENN) and AllKNN.

As its name suggests, RENN repeats the process of Edited Nearest Neighbours multiple times. It will continue to do so until a pre-determined number of iterations has been reached or when no more observations can be removed — whichever comes first.

AllKNN is another ENN variation that repeats the ENN algorithm several times. The first iteration examines only the closest neighbor of each observation from the majority class. Then, it increases the number of neighbors examined at each iteration by 1. Afterward, the algorithm stops at the round corresponding to the user-determined number of neighbors or when one of the majority classes becomes the minority — whichever happens first.

Want to find out more on RENN and AllKNN? Visit our book [Machine Learning with Imbalanced Data](https://www.trainindata.com/p/imbalanced-data-myths-mistakes-solutions-book).

## NearMiss Undersampling

NearMiss is a family of undersampling methods that select examples from the majority class based on their (Euclidean) distance from examples in the minority class. The basic idea is to keep examples that are close to the minority class and discard examples that are far away from them.

There are three variants of the NearMiss algorithm:

**NearMiss-1:** The algorithm identifies the K closest neighbors of each majority class observation from the minority class. It determines the average distance between each majority observation and its K nearest minority class neighbors. The algorithm then selects the majority class observations with the smallest average distance to the minority class, retaining only those samples.

**NearMiss-2:** Like NearMiss-1, it also involves selecting examples based on their distance from the minority class. However, instead of selecting examples based on the mean distance to the closest minority examples, NearMiss-2 selects examples based on the mean distance to the furthest minority examples.

To achieve this, NearMiss-2 determines the mean distance of each observation from the majority class to its K furthest neighbors from the minority class. Then, it retains the majority class observations with the smallest average distance.

**NearMiss-3**: NearMiss-3 tackles class imbalance by selecting majority class examples close to minority class ones. It works by finding the three closest neighbors of each minority class example that belong to the majority class. Then, it removes all majority class examples not among the three closest neighbors.

Next, for the remaining majority class examples, it calculates the average distance to its K closest neighbors from the minority class. Finally, the algorithm retains only those majority class examples with the largest average distance.

For a Python implementation of NearMiss with imbalanced learn, visit our [GitHub repository](https://github.com/solegalli/machine-learning-imbalanced-data/blob/master/Section-05-Undersampling/09-NearMiss.ipynb) on NearMiss undersampling.

## Wrap-up

It is important to note that selecting the best undersampling method depends on the specific characteristics of the dataset. Hence, the most effective technique may vary from one dataset to another. Therefore, it is crucial to evaluate the performance of each method to determine the best approach for a specific task.

We’ve reviewed only a handful of undersampling methods from the above list in this article. Check out our [Machine Learning with Imbalanced Data](https://www.trainindata.com/p/imbalanced-data-myths-mistakes-solutions-book) book. You’ll find detailed discussions and tutorials on other methods and comprehensive applications of Python open-source libraries for handling imbalanced data.

The data science community is yet to reach a consensus on which technique is the most effective or suitable for any given dataset. So, take this opportunity to learn more about these techniques and determine what works best for you.

You can always practice using our [GitHub repository](https://github.com/solegalli/machine-learning-imbalanced-data/tree/master/Section-05-Undersampling) of undersampling examples or learn more from our [Machine Learning with Imbalanced Data](https://www.trainindata.com/p/imbalanced-data-myths-mistakes-solutions-book) book.
