---
layout: post
title: "Population Stability Index and feature selection in Python"
author: sole
description: "Find out what the Population Stability Index is and how to use it to monitor or select features based on their distribution changes over time."
excerpt: "Find out what the Population Stability Index is and how to use it to monitor or select features based on their distribution changes over time."
categories: [Feature Selection, Machine Learning]
image: assets/images/posts/population-stability-index-and-feature-selection-python/psi.gif
---

If you worked in credit scoring, you probably heard about the **Population Stability Index**, or PSI.

The PSI is a metric that quantifies the changes in a variable distribution and it is commonly used to assess the risk of using a variable in a credit risk model.

Unstable features are variables whose distribution changes after certain events, for example due to policy changes or a recession. Unstable features may affect the model performance. Thus, we want to avoid utilizing unstable features in credit risk scorecards.

In this article, we will discuss what the Population Stability Index is, its uses in credit scoring and data science, and how we can select features based on their PSI values with the Python open-source package [Feature-engine](https://feature-engine.readthedocs.io/en/latest/index.html).

*For tutorials on feature selection, check out our course [Feature Selection for Machine Learning](https://www.trainindata.com/p/feature-selection-for-machine-learning) or our book [Feature Selection in Machine Learning with Python](https://www.trainindata.com/p/feature-selection-in-machine-learning-book).*

[![Feature Selection for Machine Learning, online course.]({{ site.baseurl }}/assets/images/posts/feature-selection-with-filter-methods/feature-selection-course-1024x576.png)](https://www.trainindata.com/p/feature-selection-for-machine-learning)

## Population Stability Index: What is it?

The Population Stability Index (PSI) is a statistical measure of how much a population distribution has changed over time or how different the variable distributions are between two samples.

The PSI was originally designed and used in credit risk scorecards to monitor the changes in the independent variable distributions in time, or between the training data (often referred to as a development sample) and the live data integration.

## Why would the variable distribution change?

Changes in population distribution can occur for various reasons. For example, due to

- changes in the economic landscape, e.g., after a recession.
- changes in the source of data, e.g., changes in the source of credit risk variables.
- Organizational decisions, e.g., the organization expands to serve other sectors of society.

Some of these changes are predictable and, therefore, we expect them, like policy or organizational changes or recession.

Some of these changes are unexpected, like the shut down of a credit data source or the credit data source changing the procedure in which they collect their data.

With the PSI, we can monitor the model variable distribution in real time and therefore detect these changes as soon as they occur.

![population distribution in time]({{ site.baseurl }}/assets/images/posts/population-stability-index-and-feature-selection-python/population-distribution-in-time.png)

## PSI: How is it calculated?

The PSI compares the frequency of observations across intervals of a numeric variable.

To determine the PSI, we first sort continuous features into discrete intervals. We often use the deciles to determine the interval limits. But we can use any other quantile as well.

Then, we determine the fraction of observations in each interval.

Finally, we compare the fraction of observations per interval between a reference data set and a test data set.

### Reference vs test data sets: examples

If we were monitoring the variables after a model was deployed, the reference data set could be the training data set, and the test data set could be a portion of the live data, containing the current observations.

If instead, we wanted to compare the distribution change caused in a certain time period, for example, during a recession, then the reference data set would contain observations before the recession, and the test data set would contain observations from the start of the recession onward.

#### In summary

In summary, the PSI is computed as follows:

- Define the interval limits into which the observations will be sorted.
- Sort the feature values into the intervals.
- Determine the fraction of observations within each interval.
- Calculate the PSI.

The PSI formula is:

![population stability index equation]({{ site.baseurl }}/assets/images/posts/population-stability-index-and-feature-selection-python/formula-1024x203.png)

Where “Actual” is the fraction of observations in the test data set and “Expected” is the fraction of observations in the reference data set.

If the variable distribution did not change, the “Expected” and “Actual” fraction of observations should be very similar, and thus the value of the PSI should be close to 0.

Greater PSI values indicate bigger differences between the expected and actual distributions.

## PSI: how big is the distribution shift?

The PSI is a metric to determine a distribution change. But what constitutes a significant change?

Different thresholds can be used to assess the magnitude of the distribution shift according to the PSI value. Although there are not many studies about the statistical properties of PSI and what constitutes a good threshold, the following are generally used as a “rule of thumb”:

The most commonly used thresholds are:

- Below 10%, the variable has not experienced a significant shift in its distribution.
- Above 25%, the variable has experienced a major shift in its distribution.
- Between those two values, there is a slight change to intermediate shift in the distribution.

In other words, if the PSI value is greater than 0.25, there’s been a significant change in the population distribution.

## PSI: What is it used for?

We’ve seen that the PSI values can tell us how much a variable distribution has shifted over time or between samples. So, how is this metric used in data science?

The most common uses of the PSI are for:

- finding unstable features
- model validation
- model monitoring

### Finding unstable features

During model development, we can use the PSI to determine if a feature is suitable for a statistical or machine learning model.

Unstable features, those whose distribution changes over time, may affect model performance. With the PSI, we can detect and avoid using unstable features as model variables.

This way of evaluating features is often referred to as “characteristic analysis”. This just means variable analysis, or feature analysis.

### Model validation

Once we created our model and evaluated it in a test and hold out sample, it is common practice to obtain more recent data and check again that the PSI values of the variables do not suggest a significant change in their distribution in recent times.

If the PSI values of the features changed in recent data, then the development sample, or in other words, the training data set, is no longer representative of the population distribution in the recent time period.

### Model monitoring

After the machine leaning model is put into production, the PSI values can be used to monitor and detect changes in the variable distribution that may compromise the model performance. If the PSI values of the variables is small, then we can, in principle, continue using the existing model safely.

![monitoring machine learning models]({{ site.baseurl }}/assets/images/posts/population-stability-index-and-feature-selection-python/monitoring-machine-learning-models.png)

### In summary

Continuously monitoring the variable distribution can help us understand population changes. This will help us assess the risk of using certain variables in predictive models and also understand potential drifts in the model performance once it is in production.

## PSI: considerations

There are a few things that we need to consider when using the PSI to determine the stability or suitability of a feature.

First, the PSI is not symmetric; switching the order of the reference and test data sets when calculating the PSI will return different values.

Second, the number of bins or intervals into which the continuous variables are sorted affects the PSI values.

It is common practice to sort variables into 10 intervals. But 10 intervals may not be suitable for skewed variables, leading to inaccurate PSI values.

It is useful to corroborate the PSI values with plots of the variable distributions. If the PSI value does not match the plots, it is worth exploring what might be the reason and trying a different number of intervals in the PSI calculation.

Third, the PSI is suitable for continuous and highly cardinal features. Categorical or discrete variable distributions can be evaluated by other statistical tests, like Chi-squared.

Finally, it is worth considering that an unstable feature, or in other words, a feature with high PSI, is not a synonym for a non-predictive feature. In fact, a feature can be very predictive of the target and yet show a high PSI. When excluding features based on the PSI, we do so to avoid incurring risks from potential changes in feature distribution, and not necessarily to return a better performing model.

## How to select features based on the PSI in Python?

We can select features based on the PSI by utilizing the open-source Python package [Feature-engine](https://feature-engine.readthedocs.io/en/latest/index.html).

Feature-engine has recently released a new transformer, [DropHighPSIFeatures](https://feature-engine.readthedocs.io/en/latest/api_doc/selection/DropHighPSIFeatures.html#feature_engine.selection.DropHighPSIFeatures), that takes in a dataframe, determines the PSI of its numerical features, and then selects those with low PSI values.

The transformer DropHighPSIFeatures takes a dataframe, typically the training set, and splits it into two: a reference and a test set. Then, it compares the PSI between these 2 data sets and identifies features with PSI above a certain threshold that can be determined by the user.

We can create an instance of the transformer as follows:

```
from feature_engine.selection import DropHighPSIFeatures

transformer = DropHighPSIFeatures(split_frac=0.6)
```

This will tell the transformer to split the training set so that 60% of the observations are in the reference data and 40% are in the test data.

Now, we can go ahead and calculate the PSI values and identify the variables with high PSI.

```
transformer.fit(X_train)
```

That will identify the features with high PSI.

We can now go ahead and remove those features from training and test sets.

```
X_train = transformer.transform(X_train)
X_test = transformer.transform(X_test)
```

The beauty of the DropHighFeatures resides in its versatility. We have the option to:

- Analyze all numerical variables or only a subset.
- Create equal-width or equal-frequency intervals.
- Select the number of intervals.
- Vary the threshold to select the features.
- Split the training set based on numerical, categorical, or date variables.
- Split the data based on the proportion of observations, specific categories, or dates.

In the rest of the article, I will show how we can use Feature-engine to evaluate the distributions of numerical variables across different portfolios and to evaluate changes in the distribution over time. For additional scenarios, visit the official [Jupyter Notebook](https://github.com/feature-engine/feature-engine-examples/blob/main/selection/Drop-High-PSI-Features.ipynb) demo and the [Feature-engine documentation](https://feature-engine.readthedocs.io/en/latest/user_guide/selection/DropHighPSIFeatures.html).

## Evaluate distribution changes across portfolios.

Sometimes, we have data on customers coming from different portfolios. We may want to determine if we can put the portfolios together based on the differences in the variable distribution. And, we can use the PSI to determine the variable drift.

We will work with the [Credit Approval data set](https://feature-engine.readthedocs.io/en/latest/user_guide/selection/DropHighPSIFeatures.html) from the UCI Machine Learning repository. Guidelines to download and prepare the dataset to follow along with this example can be found [here](https://github.com/feature-engine/feature-engine-examples/blob/main/selection/Drop-High-PSI-Features.ipynb).

One of the variables in the data set, A13, indicates the portfolio to which the customers belong. These could be for example different risk portfolios, which we call portfolios 1 to 3.

In the following plot, we can see the number of customers in each portfolio:

![Bar plot showing the percentage of customers in each portfolio.]({{ site.baseurl }}/assets/images/posts/population-stability-index-and-feature-selection-python/4-1024x813.png)

Say we know portfolios 2 and 3 are riskier, and then we want to determine if the distribution of the numerical variables is different in portfolio 1 with respect to portfolios 2 and 3.

We can do this as follows with Feature-engine:

```
transformer = DropHighPSIFeatures(
   cut_off=['portfolio_2', 'portfolio_3'],
   split_col='A13', # the categorical variable with the portfolios
   strategy = 'equal_width', # the intervals are equidistant
   bins=5, # the number of intervals to sort the numerical values
   threshold=0.1,
   missing_values='ignore',
)
```

When we set up the transformer as we did in the previous code block, when presented with a training data set, it will split it in two, so that observations in portfolios 2 and 3 will be assigned to the reference or “expected” dataframe. And the observations in portfolio 1 will be assigned to the test or “actual” dataframe.

With the method `fit()`, the transformer splits the data, determines the PSI, and identifies the features to drop:

```
transformer.fit(X_train)
```

After `fit()`, the transformer would have identified all the numerical variables, which are stored in the attribute `variables_`, and also calculated their PSI, which we can find in the following attribute:

```
transformer.psi_values_
```

Which results in:

```
{'A2': 0.1059045985405272,'A3': 0.4547839756788118,'A8': 0.2959469283080945,
'A11': 0.9635236290505808, 'A14': 0.2145456167378388,'A15': 0.10929706991455773}
```

If the PSI values are above the threshold that we entered when initializing the transformer, they will be dropped. We can find the variables to drop in another attribute:

```
transformer.features_to_drop_
```

Which results in:

```
['A2', 'A3', 'A8', 'A11', 'A14', 'A15']
```

In this case, the distribution of all the numerical variables between the lower and higher risk portfolios is very different.

We can go ahead and plot the cumulative distributions in what constituted the reference or expected data and the test or actual data. The `DropHighPSIFeatures` stores the value that was used to separate the data in its `cut_off_` attribute. We can use this attribute to split the training data manually, and then plot the cumulative distribution of a variable, say A3, in each data set:

```
tmp = X_train['A13'].isin(
    transformer.cut_off_)sns.ecdfplot(data=X_train, x='A3', hue=tmp,
    )
```

We obtain the following plot, where we appreciate that the distribution is quite different, in line with the high PSI value:

![Cumulative distribution of the variable in different portfolios, showing high PSI]({{ site.baseurl }}/assets/images/posts/population-stability-index-and-feature-selection-python/5-1024x684.png)

Now we can go ahead and remove the variables with high PSI from the training set and also the test set, utilizing the transform() method:

```
X_train = transformer.transform(X_train)
X_test = transformer.transform(X_test)
```

And that’s it. The variables have been removed.

## Evaluate distribution changes in time

In the previous example, we compared variable distributions across portfolios. We can also evaluate distributions in time. We can compare the feature distribution during and after a recession, for example.

If we have a datetime variable in our data, we can split the data based on time as follows:

```
transformer = DropHighPSIFeatures(
   cut_off = pd.to_datetime('2018–12–14'), # the cut_off date
   split_col='date', # the date variable
   strategy = 'equal_frequency',
   bins=8,
   threshold=0.1,
   missing_values='ignore',
)
```

When we instantiate the transformer like in the previous code block, the transformer will allocate all observations prior to September 14th, 2018 to the reference or “expected” data set, and all observations posterior to this date to the test or “actual” data set.

With `fit()` the transformer splits the data, determines the PSI and identifies the features to drop:

```
transformer.fit(X_train)
```

We find the PSI values for each variable in an attribute:

```
transformer.psi_values_
```

Which results in:

```
{'A2': 0.042897461606348365,'A3': 0.12454277821334697,
 'A8': 0.3103787266508977,'A11': 0.18886985933386097,
 'A14': 0.031337238449971605,'A15': 0.05807864506797143}
```

And the transformer also stores, separately, the variables that will be dropped based on their PSI value:

```
transformer.features_to_drop_
```

Which results in:

```
['A3', 'A8', 'A11']
```

To assess the value of the PSI for each feature against its cumulative distribution, we can use the cut_off_ value to manually split the data into the reference and test sets, and then plot the cumulative distribution of one variable:

```
tmp = X_train['A13'].isin(
    transformer.cut_off_)sns.ecdfplot(data=X_train, x='A3', hue=tmp)
```

![Cumulative distribution with moderate PSI]({{ site.baseurl }}/assets/images/posts/population-stability-index-and-feature-selection-python/6-1024x710.png)

We can see that there is a change in the cumulative distribution before and after the indicated date, which coincides with its higher PSI value.

For comparison, we can plot a variable with a low PSI:

```
sns.ecdfplot(data=X_train, x='A14', hue=tmp)
```

![Cumulative distributions showing low PSI]({{ site.baseurl }}/assets/images/posts/population-stability-index-and-feature-selection-python/7-1-1024x706.png)

As expected, the cumulative distribution before and after does not change for this particular variable with low PSI.

We can go ahead and remove those variables from the training and test sets as follows:

```
X_train = transformer.transform(X_train)
X_test = transformer.transform(X_test)
```

And that is it. Now, our datasets do not contain numerical features whose distribution changes in time.

Go ahead and check out our extensive demo for additional business scenarios in which the PSI can be useful.

## Wrapping up

If you made it this far, thank you for reading.

For tutorials on feature selection, check out our course [Feature Selection for Machine Learning](https://www.trainindata.com/p/feature-selection-for-machine-learning) or our book [Feature Selection in Machine Learning with Python](https://www.trainindata.com/p/feature-selection-in-machine-learning-book).

[![Feature Selection in Machine Learning with Python, book cover]({{ site.baseurl }}/assets/images/posts/machine-learning-books-for-beginners/book-cover-1024x1024.png)](https://www.trainindata.com/p/feature-selection-in-machine-learning-book)

## References

[Statistical properties of the Population Stability Index.](https://scholarworks.wmich.edu/cgi/viewcontent.cgi?article=4249&context=dissertations) Bilal Yurdakul. Dissertation
