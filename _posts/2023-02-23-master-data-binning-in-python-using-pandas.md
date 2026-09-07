---
layout: post
title: "Master Data Binning in Python using Pandas"
author: sole
description: "Find out what data binning is, why we do it, and how to implement it in Python."
excerpt: "Find out what data binning is, why we do it, and how to implement it in Python."
categories: [Data Preprocessing, Feature Engineering, Machine Learning]
image: assets/images/posts/master-data-binning-in-python-using-pandas/cover-5.gif
---

Binning (also called discretization) is a widely used data preprocessing approach. It consists of sorting continuous numerical data into discrete intervals, or “bins.” These intervals or bins can be subsequently processed as if they were numerical or, more commonly, categorical data.

Binning can be helpful in data analysis and data mining since it can make complex data sets simpler and easier to handle. It can help in locating patterns and trends in data, making it easier to draw insightful conclusions and make data-driven decisions.

Binning data, sometimes also referred to as bucketing, is also useful in data science and machine learning projects, as it reduces the training time of decision tree-based algorithms by reducing the number of cut-points they examine during the induction (training process).

In this tutorial, we’ll look into binning data in Python using the `cut` and `qcut` functions from the open-source library pandas. We will discuss three basic types of binning: arbitrary binning, equal-frequency binning, and equal-width binning.

To to help you master data binning and many other data transformation techniques in Python check out the [Python Feature Engineering Cookbook](https://www.packtpub.com/en-us/product/python-feature-engineering-cookbook-9781835883587) and [Feature Engineering for Machine Learning course](https://www.trainindata.com/p/feature-engineering-for-machine-learning).

[![Feature Engineering for Machine Learning course]({{ site.baseurl }}/assets/images/posts/feature-scaling-in-machine-learning/feature-engineering-machine-learning-course-1024x576.jpg)](https://www.trainindata.com/p/feature-engineering-for-machine-learning)

Let’s dive in.

## Equal-Width Binning

Equal-width binning is the process of dividing continuous variables into a predetermined number of equal-width intervals. These are examples of contiguous equal-width intervals: 0-10, 10-20, 20-30, and so on.

The number of bins into which the variable will be sorted is determined by the user. It usually depends on the desired granularity of the final variable. The number of bins can become a parameter that needs to be optimized to minimize the information loss while maximizing the variable simplification, which in turn results in returning the maximum performance of a classifier or a regression model.

The steps involved in performing equal-width binning are as follows:

- Define the number of bins you want to create.
- Find the variable’s minimum and maximum and calculate the value range.
- Determine the width of each bin by dividing the value range by the number of bins.
- Define the bins, that is, each bin’s lower and upper limits.
- Assign the observations to the appropriate bin based on their value.

The width of each bin (step 3), is given by:

bin_width = (max_value – min_value) / number_of_bins

where ‘max_value’ is the maximum value of the variable being binned, ‘min_value’ is the minimum value, and the ‘number_of_bins’ is the desired number of bins.

Using the bin width, the bin edges or limits (step 4) are calculated as follows:

- bin_1 = [min_value, min_value + bin_width],
- bin_2 = [min_value + bin_width, min_value + 2 bin_width]
- …
- bin_n = [min_value + (n-1) bin_width, max_value]

To illustrate this with an example, if we have a variable with minimum and maximum values of 0 and 100, respectively, and we want to sort the values into 5 bins, the bin width is given by (100-0)/2, which is 20. Then, the bins are [0–20], [20–40], and [40–60]. [60-80] and [80-100].

Sorting variables into bins of equal width preserves the variable distribution. Hence, if the variable is skewed, it will still be skewed after the discretization.

### Equal-Width Binning with Pandas cut()

Let’s see how to perform equal-width binning using pandas `cut()`. We will use the California housing dataset from Scikit-learn.

Let’s import pandas and load the dataset into a pandas dataframe:

```
import pandas as pd
from sklearn.datasets import fetch_california_housing

data, y = fetch_california_housing(return_X_y=True, as_frame=True)

data.head()
```

In the following image, we see a snapshot of the variables in the pandas dataframe.

![First 5 rows of the california housiing dataframe.]({{ site.baseurl }}/assets/images/posts/master-data-binning-in-python-using-pandas/fig1-1024x291.png)

Note that we loaded the data directly from Scikit-learn. You’ll probably have to use pandas `read_csv` to load data from your computer.

Let’s now use pandas `cut()` to sort the variable Medinc into 10 bins of equal-width. We will capture the binned variable in a new column in the dataset.

```
data["income_binned"] = pd.cut(data["MedInc"], bins=10)

print(data["income_binned"])
```

Below we see the binned variable:

```
0          (7.75, 9.2]
1          (7.75, 9.2]
2          (6.3, 7.75]
3          (4.85, 6.3]
4          (3.4, 4.85]
             ...
20635    (0.485, 1.95]
20636      (1.95, 3.4]
20637    (0.485, 1.95]
20638    (0.485, 1.95]
20639      (1.95, 3.4]
Name: income_binned, Length: 20640, dtype: category
Categories (10, interval[float64, right]): [(0.485, 1.95] < (1.95, 3.4] < (3.4, 4.85] < (4.85, 6.3] ... (9.2, 10.65] < (10.65, 12.1] < (12.1, 13.55] < (13.55, 15.0]]
```

Here, `dtype` is the data type. Note that pandas `cut()` returns a categorical variable by default.

Note that the values of the variable are now the intervals. The method `cut()` has the parameter labels. Its default value is None, which returns the bin edges. If we set it to False instead, it returns integer indicators of the bins.

Now, let’s check if we got the 10 intervals and also count the number of observations within each bin:

```
data["income_binned"].value_counts()
```

We can see that there are ten intervals of equal width, and they show different numbers of observations:

```
(1.95, 3.4]      7436
(3.4, 4.85]      6098
(4.85, 6.3]      2990
(0.485, 1.95]    2247
(6.3, 7.75]      1061
(7.75, 9.2]       427
(9.2, 10.65]      178
(10.65, 12.1]      93
(13.55, 15.0]      63
(12.1, 13.55]      47
Name: income_binned, dtype: int64
```

Finally, let’s compare the original data distribution with the discretized data to see that the distribution doesn’t change even after binning.

For easy visualization, we plot a histogram with the continuous variable distribution next to a bar plot with the number of observations per bin of the binned variable.

```
from matplotlib import pyplot as plt
import seaborn as sns

fig, ax =plt.subplots(1,2, figsize=(20,8))

sns.histplot(data = data, x="MedInc", kde=True, ax=ax[0])
sns.countplot(data=data, x="income_binned", ax=ax[1])
ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45, horizontalalignment='right')

fig.show()
```

As you can see in the histogram, the median income is skewed, and so is the categorical variable created after binning.

![HIstogram showing theskewed distribution of median income followed by a bar plot showing equal width discretization of the same variable.]({{ site.baseurl }}/assets/images/posts/master-data-binning-in-python-using-pandas/fig2-1024x458.png)

### Advantages and Disadvantages of Equal-Width Binning

Equal-width binning, like any data preprocessing technique, has its advantages and disadvantages. Here are some of the pros and cons of using equal-width binning:

#### Advantages

- Equal-width binning is easy to understand and implement, making it a popular choice among data analysts and data scientists.
- It is a good option when we have no prior knowledge of the data and want to create bins that have an equal range of values.

#### Disadvantages

- Equal-width binning can result in bins that are not equally populated with data, which can lead to a loss of information by placing most observations in the same interval.
- The choice of the number of bins can be subjective and have a significant impact on the results, so it is important to choose an appropriate number of bin values that balances the trade-off between granularity and under-fitting.

Finally, before carrying out binning, make sure your variable does not have missing values. Otherwise, the data will remain missing after the discretization.

## Equal-Frequency Binning

In equal-frequency binning, we sort the data values of a continuous variable into bins that contain the same number of observations. The quantiles are used to determine the bin edges. The resulting intervals may not have equal width, and that’s OK.

For example, if we want to sort the variable income into 5 intervals of equal frequency, we would determine the 20th, 40th, 60th, 80th, and 100th quantiles to find out the limits of the bins.

The beauty of equal-frequency binning is that it improves the value spread of skewed variables.

### Equal-Frequency Binning with Pandas qcut()

Let’s carry out equal-frequency binning in Python using pandas `qcut()` using the California housing dataset. Let’s import the libraries and load the data:

```
import pandas as pd
from sklearn.datasets import fetch_california_housing

data, y = fetch_california_housing(return_X_y=True, as_frame=True)
```

We will use pandas `qcut()` to create ten intervals with equal-frequency. Just like in the previous example, we will capture the discretized variable in a new column:

```
data["income_binned"] = pd.qcut(data["MedInc"], q=10)

print(data["income_binned"])
```

By defining the number of quantiles using the q argument, we utilize `qcut()` to conduct equal-frequency binning of the variable. In this instance, we set the quantiles to 10, which separated the data into 10 equal-frequency groups.

In the following output, we see the resulting variable:

```
0         (6.159, 15.0]
1         (6.159, 15.0]
2         (6.159, 15.0]
3         (5.11, 6.159]
4        (3.535, 3.967]
              ...
20635    (0.499, 1.904]
20636     (2.352, 2.74]
20637    (0.499, 1.904]
20638    (0.499, 1.904]
20639     (2.352, 2.74]
Name: income_binned, Length: 20640, dtype: category
Categories (10, interval[float64, right]): [(0.499, 1.904] < (1.904, 2.352] < (2.352, 2.74] < (2.74, 3.141] ... (3.967, 4.438] < (4.438, 5.11] < (5.11, 6.159] < (6.159, 15.0]]
```

Let’s check if there are the same number of elements in each interval:

```
data["income_binned"].value_counts()
```

As expected, every bin has the same amount of observations because the `qcut()` method divides the data into bins of equal frequency. The output is shown below:

```
(2.74, 3.141]     2068
(0.499, 1.904]    2066
(3.967, 4.438]    2066
(1.904, 2.352]    2064
(3.535, 3.967]    2064
(5.11, 6.159]     2064
(6.159, 15.0]     2064
(2.352, 2.74]     2063
(4.438, 5.11]     2062
(3.141, 3.535]    2059
Name: income_binned, dtype: int64
```

Note however, that the value range of each interval varies.

Equal-frequency binning improves the value spread of the variable. Let’s compare a histogram with the original variable distribution with a bar plot of the binned variable:

```
from matplotlib import pyplot as plt
fig, ax =plt.subplots(1,2, figsize=(35,8))
plt.figure(figsize=(20,8))

sns.histplot(data = data1, x="population", kde=True, ax=ax[0])
sns.countplot(data=data1, x='bins_cut1', ax=ax[1])
fig.show()
```

As we see, the original variable is rightly skewed; however, the discretized variable shows a uniform distribution:

![HIstogram showing theskewed distribution of median income followed by a bar plot showing equal frequency discretization of the same variable.]({{ site.baseurl }}/assets/images/posts/master-data-binning-in-python-using-pandas/fig3-1024x459.png)

### Advantages and Disadvantages of Equal-Frequency Binning

Equal-frequency binning has some advantages and disadvantages, which I describe below:

#### Advantages

- It is a good option when the distribution of data is skewed and you want to ensure that each bin has a roughly equal representation of the data.
- It is suitable for data that is not normally distributed or contains outliers.

#### Disadvantages

- The number of data points in each bin may not be equal due to rounding errors or an uneven distribution of the data.
- If the variable is very skewed, we may need to chose a small number of quantiles, otherwise some bins may end up empty.
- It may result in a loss of information by putting in the same interval observations that are associated with a different (target) class.

As with equal-width discretization, in equal-frequency, the number of bins is determined arbitrarily by the user and its value might need to be optimize to extract the maximum value from a variable.

## Arbitrary Binning

In arbitrary binning, the bins limits are defined arbitrarily based on the domain knowledge or specific requirements of the problem. Unlike equal-width or equal-frequency binning, the bin boundary values are not determined by the data itself but rather by the data analyst or the problem domain.

Arbitrary binning is useful when the data has a specific meaning or context that cannot be captured by other binning methods. For example, on an e-commerce website, we might group customers into different categories based on their shopping habits or purchase history. This would involve creating custom bins that are relevant to the problem rather than relying on statistical criteria to determine the bin boundaries.

There is no specific mathematical formula for arbitrary binning, as the bin boundaries are determined by the specific requirements of the problem or domain knowledge. In practice, arbitrary binning involves selecting bin boundaries based on specific criteria, such as ranges of values, meaningful categories, or business rules.

To give an example, suppose we are analyzing the income levels of people in a specific region. We might define arbitrary bins based on the following income ranges: “low income” for incomes less than $30,000, “middle income” for incomes between $30,000 and $70,000, and “high income” for incomes above $70,000. In this case, the bins are determined by the specific problem domain, and the bin boundaries are not determined by any statistical criteria.

### Arbitrary Binning with Python

Here’s an example of how to use pandas `cut()` to perform arbitrary binning. First, we import the necessary libraries and load the California housing dataset as shown below:

```
import pandas as pd
import numpy as np

from sklearn.datasets import fetch_california_housing

data, y = fetch_california_housing(return_X_y=True, as_frame=True)
```

We define the bins edges “manually” and capture them in a custom_bins list. After that, we use pandas `cut()` to bin the data into the custom bins, assigning the bin labels ‘Very Low’, ‘Low’, ‘Medium’, ‘High’, and ‘Very High’ to each bin.

```
custom_bins = np.linspace(0, 10000, 6)
labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']

data['population_binned'] = pd.cut(data['Population'],
                                   bins=custom_bins,
                                   labels=labels,
                                  )

print(data['population_binned'].value_counts())
```

Let’s now count the number of observations in each interval:

```
print(data['population_binned'].value_counts())
```

Below we see the interval name followed by the number of observations in it:

```
Very Low     16960
Low           3094
Medium         394
High           132
Very High       37
Name: population_binned, dtype: int64
```

Let’s visualize the variable before and after the discretization:

```
from matplotlib import pyplot as plt
import seaborn as sns

fig, ax =plt.subplots(1,2, figsize=(20,8))

sns.histplot(data = data, x="Population", kde=True, ax=ax[0])
sns.countplot(data=data, x="population_binned", ax=ax[1])
ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45, horizontalalignment='right')

fig.show()
```

![HIstogram showing the skewed distribution of the variable Population followed by a bar plot showing arbitrary binning of the same variable.]({{ site.baseurl }}/assets/images/posts/master-data-binning-in-python-using-pandas/fig4-1024x446.png)

### Advantages and Disadvantages of Arbitrary Binning

Let’s discuss the advantages and limitations of binning arbitrarily:

#### Advantages

- Arbitrary binning provides flexibility in defining the bin boundaries based on the specific requirements of the problem, which can be more meaningful and relevant than automatically determined bins.

#### Disadvantages

- The bin boundaries are determined manually, which can introduce bias into the analysis.

## Conclusion

To wrap-up, binning is a fundamental data preparation technique that is essential to machine learning and data analysis.

When the continuous data is uniformly distributed, with equal-width binning we can separate the data into intervals of equal size, which is easy for us humans to understand and it preserves the variable distribution.

When the data is skewed, with equal-frequency binning, we can separate the data into intervals of equal frequency, therefore spreading the values more evenly across the range.

With arbitrary binning, we manually set the bin borders in accordance with our understanding of the issue domain or its needs.

The precise needs of the task determine which binning approach should be used.

Binning is a powerful data preprocessing technique that can aid in the extraction of significant features from continuous data. Being able to use the various binning techniques effectively is a crucial first step in developing as a data scientist. By employing binning efficiently, data scientists may increase the accuracy and interpretability of their models.

## Additional resources

For more details into data discretization, check our dedicated [article](https://www.blog.trainindata.com/data-discretization-in-machine-learning/).

The [Python Feature Engineering Cookbook](https://www.packtpub.com/en-us/product/python-feature-engineering-cookbook-9781835883587) and [Feature Engineering for Machine Learning course](https://www.trainindata.com/p/feature-engineering-for-machine-learning) are two excellent resources to help you master data binning and many other data transformation techniques in Python.

[![Python Feature Engineering Cookbook book cover]({{ site.baseurl }}/assets/images/posts/machine-learning-for-beginners/PFEC2ED-1024x1024.png)](https://www.packtpub.com/en-us/product/python-feature-engineering-cookbook-9781835883587)
