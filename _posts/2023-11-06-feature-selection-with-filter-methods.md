---
layout: post
title: "Feature Selection with Filter Methods in Python"
author: sole
description: "Discover what filter methods for feature selection are, their advantages and limitations, and how to implement them in Python."
excerpt: "Discover what filter methods for feature selection are, their advantages and limitations, and how to implement them in Python."
categories: [Feature Selection]
image: assets/images/posts/feature-selection-with-filter-methods/embedded-methods-feature-selection.png
---

Picture yourself faced with a vast dataset and on a mission to train a machine learning algorithm. The real challenge here is deciphering which features among the multitude of variables should be included to craft a high-performing, yet intepretable machine learning model.

This is precisely where feature selection steps in, to help us navigate the data maze, find the best features and create more robust and interpretable machine learning models.

In the lifecycle of a machine learning project, feature selection is done after data cleaning and pre-processing. It is a crucial part of any data analysis project, as it decides the quality of the input data.

In essence, feature selection involves cherry-picking a set of features from the dataset to build the best-performing model. The primary goal of feature selection algorithms is to reduce the number of features, improve model performance, enhance interpretability and maximize the resilience of the models.

When it comes to feature selection methods, we encounter three distinct categories: filter methods, wrapper methods, and embedded methods. Among wrapper methods, we find multiple techniques like forward feature selection, backward feature elimination, and recursive feature elimination. With embedded methods, the selection process occurs during the training of the machine learning algorithms, for example, by obtaining tree-derived feature importance or by using Lasso regularization.

Our focus for today is on filter methods. By the end of this article, you’ll be familiar with the different filter-based selection methods, how they work, and when to use them.

For more details about these and other feature selection methods, check out our [Feature Selection for Machine Learning course](https://www.trainindata.com/p/feature-selection-for-machine-learning) and [Feature Selection in Machine Learning book](https://www.trainindata.com/p/feature-selection-in-machine-learning-book).

[![Feature Selection for Machine Learning, online course.]({{ site.baseurl }}/assets/images/posts/feature-selection-with-filter-methods/feature-selection-course-1024x576.png)](https://www.trainindata.com/p/feature-selection-for-machine-learning)

## What are filter-based feature selection methods?

As the name suggests, these methods filter out less important features and help you retain the relevant features that add value to your model. How do we decide which features are important?

In filter methods, the variables that have the most impact on the output or target variable are considered important. The variables are ranked based on their significance towards the output. The top-ranking features are selected and the irrelevant features are removed.

![Feature selection process based on filter methods]({{ site.baseurl }}/assets/images/posts/feature-selection-with-filter-methods/ch1-fig4.png)

### Ranking the variables

But, how do we rank the variables?

There are multiple statistical tests that we can use to find associations between features and the target variable. These tests rank features based on their “importance”. The most commonly used statistical tests in data science projects are:

- Chi-squared
- Anova
- Correlation

The statistical method we choose depends on multiple factors. For example, whether the target and the predictor variable are continuous or categorical, and the relationship between them.

In the coming sections, I’ll walk you through each of these tests, and show you how to implement them in Python.

## Chi-square

The chi-square test is used to determine the association between categorical variables. So, you can use this test when both your independent variable and target variable are categorical. The categorical variables can be binary or multiclass.

![Chi-square is suitable to select features when both predictors and target are categorical]({{ site.baseurl }}/assets/images/posts/feature-selection-with-filter-methods/ch4-chisquare.png)

How does it work?

- First, the algorithm assumes a **Null Hypothesis (H0):** There’s no association between the two categorical variables.
- **Alternative Hypothesis (H1):** There’s a significant association between the two variables.
- The test compares observed frequencies to expected frequencies of categorical data, to check for association. The expected frequency is that of no-association.
- Next, it computes the p-value to reject or agree with the Null hypothesis. If the null hypothesis is rejected, we can say that there is an association between the variables.

If the observed frequencies are very different from the expected frequencies, then the p-value is small, and we reject the null hypothesis. We use these p-value to rank the variables and select the top-ranking subset of features.

Now, let’s see an example of how to use this test. Let’s use chi-square to assess the association of categorical and discrete variables with survival in the Titanic data set.

### Chi-square feature selection with Python

The first step is to import the libraries, functions, and classes:

```
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
```

Let’s load a subset of the Titanic dataset with categorical and discrete variables:

```
variables = ['pclass', 'survived', 'sex', 'sibsp', 'parch', 'embarked']
data = pd.read_csv(
    'https://www.openml.org/data/get_csv/16826755/phpMYEkMl',
    usecols=variables,
    na_values='?',
    )
data.dropna(subset=['embarked'], inplace=True)
data.head()
```

In this data, the ‘survived’ column is our target, which indicates if a passenger survived the Titanic disaster.

We have two categorical features: ‘gender’ and ‘embarked’. The rest of the variables are discrete and contain only a few distinct values, so we treat them as categorical features too.

We aim to evaluate their association with ‘survival ‘using the chi-square test and choose the best features for the classifier.

Let’s split the data into a training data set and a test set:

```
X_train, X_test, y_train, y_test = train_test_split(
    data.drop("survived", axis=1),
    data['survived'],
    test_size=0.3,
    random_state=0,
)
```

The first step to calculating the chi-square statistic is to obtain the contingency tables between the input variables and the target:

```
c = pd.crosstab(y_train, X_train['sex'])
```

If we execute `c`, we will see the observed frequencies for the variables gender and survival:

![Contingency table between gender and survival in the titanic]({{ site.baseurl }}/assets/images/posts/feature-selection-with-filter-methods/ch4-fig3.png)

With the contingency table, we can now calculate the expected frequencies, degrees of freedom, chi-square, and the probability of the variables not being associated:

```
chi2_contingency(c)
```

The chi-square statistic, the probability of no association, the degrees of freedom, and the expected frequencies are all included in the output of `chi2_contingency`:

```
(249.44419858265127, # chi-square stat
 3.432495124524887e-56, # probability of no association
 1,
 array([[199.63676149, 372.36323851],
 [119.36323851, 222.63676149]]))
```

We’ve analyzed the association between gender and survival. Similarly, let’s obtain the probability of association between every variable and the target to select new features:

```
chi_ls = []

for feature in X_train.columns:
    c = pd.crosstab(y_train, X_train
)
    p_value = chi2_contingency(c)[1]
    chi_ls.append(p_value)
```

Let’s transform the list with the probabilities into a pandas series, add the variable names in the index, sort the probability values in ascending order, and plot a bar chart:

```
pd.Series(chi_ls, index=X_train.columns).sort_values(ascending=True).plot.bar(rot=45)
plt.ylabel("p value")
plt.title("Feature importance based on chi-square test")
```

In the following visualization, we see a bar plot with the probability of no association for every feature with the target. The smaller the probability, the higher the association of the variable with the target.

![Bar plot showing feature important determined as the p-value of a chi-square test]({{ site.baseurl }}/assets/images/posts/feature-selection-with-filter-methods/ch4-fig4.png)

We can use different criteria to select features based on probabilities. One criterion is to select all features with a p-value greater than 0.05. There’s a catch in this approach, however. With bigger datasets, tiny differences in the category frequencies will be considered significant. In other words, in large datasets, selecting features based on the p-value may increase the error type I.

An alternative selection criteria is to select the top k number of features or the top k percentile. For example, let’s capture the names of the top 3 ranking features:

```
selected = pd.Series(chi_ls, index=X_train.columns).sort_values(ascending=True)[0:3].index
```

If we execute `selected`, we see the names of the most important variables according to chi-square:

```
Index(['sex', 'pclass', 'embarked'], dtype='object')
```

We now use the top 3 features to train our predictive model. That’s how you select features from a pool of categorical variables utilizing Pearson’s chi-square test in a classification problem.

## Anova

ANOVA stands for Analysis of Variance and can be used as a feature selection technique when the **variables are continuous** and **the target is categorical.**

To reduce the feature space with one-way ANOVA, we first obtain a p-value for each feature. The p-value indicates the likelihood of the mean value of the feature being similar in the different target classes. The smaller the p-value, the more different the distributions are, hence, the more important the feature is towards the target output. We rank the features based on the p-values, and finally, we select the top-ranked features.

### Feature Selection with ANOVA with Python

We will use the breast cancer dataset from Scikit-learn for this demo. The data contains characteristics of tumors and the goal is to predict if the tumor is benign or malignant. The variables are continuous and the target is binary.

Let’s import the libraries, functions, and classes:

```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import (
    f_classif,
    SelectFpr,
    SelectKBest,
)
from sklearn.model_selection import train_test_split
```

Let’s load the breast cancer data set and separate it into training and test sets:

```
breast_cancer = load_breast_cancer()
X = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
y = breast_cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
```

Let’s perform one-way ANOVA for all the features:

```
univariate = f_classif(X_train, y_train)
```

If we execute `univariate` we will obtain 2 arrays, the first one with the F-ratio values and the second one with the p-values:

```
(array([4.56888468e+02, 8.07899168e+01, 4.90890258e+02,
         3.94647061e+02,     6.91090732e+01, 2.51091933e+02, 3.77169033e+02,
         5.64534892e+01, 2.32061889e-01, 1.89198831e+02, 4.60132652e-01,
         1.77794821e+02, 1.61603286e+02, 3.44368683e+00, 3.19209297e+01,
         .......
         9.36811856e+01, 4.94480861e+01]),
  array([2.55823842e-69, 8.37108067e-18, 8.19690935e-73, 1.48052645e-62,
         ............
         3.75876060e-20, 8.20428355e-12]))
```

Let’s capture the p-values in a pandas series, add the variable names in the index, sort the features based on their p-values, and make a bar plot:

```
univariate = pd.Series(univariate[1])
univariate.index = X_train.columns
univariate.sort_values(ascending=True).plot.bar(figsize=(20, 6), rot=45)
plt.ylabel("p-values")
plt.title("Anova")
```

In the following plot, we see that most features have p-values smaller than 0.05. For those features, we conclude that the mean value of the features between benign and malignant tumors is not the same.

There are a few features whose p-value is bigger than 0.05, which means that their mean value is similar for benign and malignant tumors.

![Bar plot showing feature importance determined with the p-values of ANOVA]({{ site.baseurl }}/assets/images/posts/feature-selection-with-filter-methods/ch4-fig7-1024x391.png)

Next, we will use one-way ANOVA and select features whose p-value is bigger than 0.05:

```
sel = SelectFpr(f_classif, alpha=0.05).fit(X_train, y_train)
```

If we execute `X_train.columns[sel.get_support()]` we’ll see the selected features:

```
Index(['mean radius', 'mean texture', 'mean perimeter', 'mean area',
        'mean smoothness', 'mean compactness', 'mean concavity',
        'mean concave points', 'mean symmetry', 'radius error',
        'perimeter error', 'area error', 'compactness error',
        'concavity   error','concave points error', 'worst radius',
        'worst texture','worst perimeter', 'worst area',
        'worst smoothness','worst compactness', 'worst concavity',
        'worst concave points','worst symmetry',
        'worst fractal dimension'],
dtype='object')
```

Let’s reduce the data to the selected variables:

```
X_train_t = sel.transform(X_train)
X_test_t = sel.transform(X_test)

X_train_t = pd.DataFrame(X_train_t, columns=sel.get_feature_names_out())
X_test_t = pd.DataFrame(X_test_t, columns=sel.get_feature_names_out())
```

You can see that the final data set contains 25 from the original 30 variables.

What if you are working with high dimensional data and want to select the top ‘K’ number of features? That’s also possible! You can do it with a one-line code as shown below:

```
sel = SelectKBest(f_classif, k=10).fit(X_train, y_train)
```

If we execute `X_train.columns[sel.get_support()]` we obtain the names of the features:

```
Index(['mean radius', 'mean perimeter', 'mean area', 'mean concavity',
'mean concave points', 'worst radius', 'worst perimeter', 'worst area',
'worst concavity', 'worst concave points'],
dtype='object')
```

You now have the top 10 features! ANOVA is a regression model, and as such, is commonly used to select features for linear models. You can try the selected features on other models, like decision-tree-based algorithm like a random forest, or SVM. However, the advantage of these models is that they capture non-linearity between features and target, so selecting features with ANOVA for these models may defeat the point of using non-linear models.

## Correlation

Correlation tests are used extensively in data science, particularly for linear regression models. By correlation, we mean the relationship between any two individual features. For feature selection, you can use correlation tests when **both the features and target are continuous.** Correlation helps with best feature subset selection and also aids in identifying and removing redundant features.

How does it work?

Two variables are related if changes in one variable are met with similar changes in the other variable. For example, if one of the variables deviates from its mean, we expect the other variable to also deviate from its mean value. Correlation tests capture this behavior of dependency between any two variables.

Pearson’s correlation coefficient (R) is one way of measuring correlation. It measures the degree of association between 2 variables and can vary between -1 and 1. When R is positive, there is a positive association between the variables; that is, the bigger the values of x, the bigger the values of y. When R is negative, there is a negative association between the variables. When R is 0, the variables are not associated.

We get to know both the strength and the direction of the relationship between two different features using this method. Hence, we rank the features based on the absolute value of the correlation coefficient and then select the top-ranking features.

### Pearson’s correlation with Python

Let’s select features using correlation in Python. We will select features based on Pearson’s correlation utilizing Scikit-learn’s `f_regression` function. This function calculates the Pearson correlation coefficient for each feature and then derives the *t*-statistic and the probability of obtaining said coefficient if the variables were not associated.

Let’s begin by importing the libraries, classes, and functions:

```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.feature_selection import (
    f_regression,
    SelectPercentile,
)
from sklearn.model_selection import train_test_split
```

Let’s load the California house price data set and separate it into a train and a test set:

```
X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
```

Let’s obtain the *t*-statistic and p-value for every feature:

```
univariate = f_regression(X_train, y_train)
```

If we execute `univariate` we will see the following 2 arrays, the first one with the *t*-statistics and the second one with the p-values:

```
(array([1.43228768e+04, 1.74602749e+02, 3.73936211e+02, 2.74469506e+01,
  1.01599261e+01, 1.59941176e+01, 3.30604925e+02, 3.17583459e+01]),
array([0.00000000e+00, 1.19851133e-39, 2.43044189e-82, 1.63580388e-07,
  1.43811689e-03, 6.38347779e-05, 4.08861089e-73, 1.77640721e-08]))
```

Let’s capture the p-values in a pandas series, add the features to the index, sort them in increasing order, and make a bar plot:

```
plt.rc("axes", titlesize=15) #fontsize of the title
univariate = pd.Series(univariate[1])
univariate.index = X_train.columns
univariate.sort_values(ascending=True).plot.bar(figsize=(10, 5), rot=45)
plt.ylabel("p-values")
plt.title("Correlation")
plt.show()
```

In the following plot, we see that all variables are significantly associated with the target; all probability values are smaller than 0.05:

![Bar plot showing feature importance derived from Pearson's correlation coefficient goodness of fit.]({{ site.baseurl }}/assets/images/posts/feature-selection-with-filter-methods/ch4-fig11.png)

We will now use Scikit-learn’s `f_regression` together with `SelectPercentile` to select the features ranked in the top 30th percentile:

```
sel = SelectPercentile(f_regression, percentile=30).fit(X_train, y_train)
sel.get_feature_names_out()
```

Below we see the features that rank in the top 30th percentile:

```
array(['MedInc', 'AveRooms', 'Latitude'], dtype=object)
```

Now, you can reduce the dataset to the selected features. Don’t forget to convert the Numpy arrays into the data frames as shown below.

```
X_train_t = sel.transform(X_train)
X_test_t = sel.transform(X_test)

X_train_t = pd.DataFrame(X_train_t, columns=sel.get_feature_names_out())
X_test_t = pd.DataFrame(X_test_t, columns=sel.get_feature_names_out())
```

And that’s it! We have now selected variables based on their correlation with the target.

Apart from the tests described in this article, the mutual information (Information gain) is also used as a filter-based feature selection technique. If you are interested, you can check out our blog on [mutual information](https://www.blog.trainindata.com/mutual-information-with-python/).

## Conclusion

Feature selection is an essential step in reducing the size of the data and ensuring optimization of the computational resources needed. I hope you are clear on which feature selection method to use based on the type of input and target variables. These filter methods compare variables in relation to the target. So they are not suitable for unsupervised learning. On the upside, they are extremely fast to compute, so they offer a good option to reduce the feature step during the early steps of data preprocessing.

Apart from feature selection, you can also explore techniques like PCA for dimensionality reduction. But be careful, PCA is NOT a feature selection procedure (add link to newsletter).

Last but not least, after carrying out feature engineering and feature selection. remember to check for underfitting or overfitting in the machine-learning model. It’s also recommended to use cross-validation to test the performance metrics of your model. In the ever-evolving landscape of artificial intelligence, you should always keep an eye out for new techniques.

To learn more about these and other feature selection methods, check out our [Feature Selection for Machine Learning course](https://www.trainindata.com/p/feature-selection-for-machine-learning) and [Feature Selection in Machine Learning book](https://www.trainindata.com/p/feature-selection-in-machine-learning-book).

[![Feature Selection in Machine Learning with Python, book cover]({{ site.baseurl }}/assets/images/posts/machine-learning-books-for-beginners/book-cover-1024x1024.png)](https://www.trainindata.com/p/feature-selection-in-machine-learning-book)

## References

1. A. Jović, K. Brkić and N. Bogunović, “A review of feature selection methods with applications,” 2015 38th International Convention on Information and Communication Technology, Electronics and Microelectronics (MIPRO), Opatija, Croatia, 2015, IEEE, pp. 1200-1205, doi: 10.1109/MIPRO.2015.7160458.

2. Sánchez-Maroño, N., Alonso-Betanzos, A., Tombilla-Sanromán, M. (2007). Filter Methods for Feature Selection – A Comparative Study. In: Yin, H., Tino, P., Corchado, E., Byrne, W., Yao, X. (eds) Intelligent Data Engineering and Automated Learning – IDEAL 2007. IDEAL 2007. Lecture Notes in Computer Science, vol 4881. Springer, Berlin, Heidelberg. <https://doi.org/10.1007/978-3-540-77226-2_19>
3. Hall, M. (2000). “Correlation-based Feature Selection of Discrete and Numeric Class Machine Learning,” in Proceedings of the Seventeenth International Conference on Machine Learning (ICML 2000), Stanford University, Stanford, CA, USA, June 29 – July 2, 2000.
