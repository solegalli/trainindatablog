---
layout: post
title: "Your Guide to Missing Values Imputation"
author: shri
description: "Find out more about missing values, how they appear in the data, and how you can replace them with unbiased estimates for machine learning."
excerpt: "Find out more about missing values, how they appear in the data, and how you can replace them with unbiased estimates for machine learning."
categories: [Data Preprocessing, Feature Engineering]
image: assets/images/posts/your-guide-to-missing-values-imputation/missing-values.png
---

Machine learning models are used by businesses to understand their market and improve the customer experience across finance, e-commerce, marketing, and more. The performance of any machine learning model depends on the quality of the training data.

One of the biggest challenges in creating a quality dataset is dealing with missing values. It is crucial to understand why our data is missing, and how to handle it.

There might be numerous reasons why certain data points were not available during data collection. It might be due to manual errors during a survey, customers may prefer not to share some information, or there could be technical errors during storage. Understanding the reasons behind non-responses is vital for improving data collection methods and reducing the impact of missing data.

In this article, I’ll walk you through different types of missing data, and discuss different techniques for handling missing values.

## Why do we need to handle missing data?

In real-world data science projects,  there is usually a significant proportion of missing values. They can affect the performance of our machine-learning model in different ways:

- In statistical analysis, it can lead to biased learning if the observed values do not accurately reflect the population. For example, if high-income individuals often skip a particular survey question, our model might learn incorrect associations.
- If the chunk of missing data is large, our model will not be able to learn the true representation of the underlying data. This will decrease the model’s accuracy and reliability.
- Some machine learning algorithms cannot work with missing values in input data, they will throw an error.
- Missing values can break the relationships between features, leading to poor feature engineering and model accuracy.

To mitigate these issues, handling missing data is a crucial step in the feature engineering process of any data science project.

You can check out our comprehensive [feature engineering course](https://www.trainindata.com/p/feature-engineering-for-machine-learning) to learn about missing data imputations and other feature engineering steps like feature extractions, transformations and more.

[![Feature Engineering for Machine Learning course]({{ site.baseurl }}/assets/images/posts/winsorization-handling-outliers-in-machine-learning/feature-engineering-machine-learning-course.jpg)](https://www.trainindata.com/p/feature-engineering-for-machine-learning)

## Types of Missing Data

Before diving into how to address missing values, we need to understand the mechanisms by which missing values are introduced into the dataset. There are three main categories of missing data mechanisms: missing completely at random (MCAR), missing at random (MAR), and missing not at random (MNAR).

1. **Missing Completely at Random (MCAR):** When the data points are missing randomly across the dataset independent of other factors, the missing data is referred to as MCAR. When the data is missing completely at random, the absence of data is unrelated to any observed or unobserved values.For example, let’s consider the data collected from multiple sensors in a manufacturing plant. If some sensors fail to record values at random times throughout the day, the resulting missing data can be considered MCAR. Here, we assume that all sensors work similarly.
2. **Missing at Random (MAR)**: In this case, why certain values are missing is not entirely random. The likelihood of data being missing correlates with other observed variables.Let’s take the same example of sensor data. If some sensors are older than others, they will be more prone to randomly failing throughout the day. There will be more missing data for older sensors, and this is considered MAR.
3. **Missing Not at Random (MNAR):** When the probability of a data point being absent is related to the data, it is MNAR. In MNAR, the likelihood of missing data varies due to unknown reasons.For example, let’s say some sensors fail because they are located in hotter areas of the warehouse. Here, missing data are not introduced at random, but based on the unobserved variable (heat).

Handling MNAR is the most difficult among the three, as the reasons for values missing are related to the data that is missing or unobserved. We often need domain knowledge to address this effectively. Understanding the mechanisms in which missing data are introduced can help decide which imputation strategies to apply.

## Missing Data Imputation

In real-world data analysis, dropping columns and rows that have null values can reduce the data set size, resulting in model training on incomplete data. Hence, we use data imputation methods.

Data imputation is a process, where we replace the missing values in a column with a non-null value calculated based on some parameters.

There are two types of data imputation methods used widely: univariate and multivariate.

- **Univariate Data Imputation:** In these methods, we replace the missing values of a variable or column using the information of the same variables’s non-missing values.For example, let us say we have a dataset with 100 people’s ages, and 15 of them are missing. We impute the null values with aggregate functions like the mean, median, or mode of the 85 non-missing age values.
- **Multivariate Data Imputation:** In this method, we use information from multiple variables (columns) to estimate and fill in missing data values. It’s more complex but often more accurate, leading to more unbiased estimates of the missing data. In multivariate imputation missing values are estimated by training machine learning models to predict the values that are missing. We can use simple algorithms like k-nearest neighbors or linear regression to predict the missing values, and then use the predictions for the imputation.For example, let’s consider a data frame with 3 features: students’ grades, age, and IQ. To fill in the missing values in the “grade” column, we build a regression model with age, and IQ as features and the available grade values as output. Once trained, we can use this model to predict the missing grades.

When the data is missing completely at random (MCAR), univariate methods provide a good imputation. When the missing data is classified as MAR, multivariate imputation methods might help better estimate the real value of the missing data.

Having said this, the imputation method is often chosen based on the model we want to train, as certain models make assumptions about the input variables, or require certain distributions, which univariate methods may distort. On the flip side, multivariate methods add complexity and computational cost to the machine learning pipeline, as we train a model, to fill in the missing data, so that we can finally train the model we are interested in.

Learn more about univariate imputation in our [Python Feature Engineering Cookbook](https://www.packtpub.com/en-us/product/python-feature-engineering-cookbook-9781835883587).

[![Python Feature Engineering Cookbook book cover]({{ site.baseurl }}/assets/images/posts/detect-outliers-in-python/PFEC2ED.png)](https://www.packtpub.com/en-us/product/python-feature-engineering-cookbook-9781835883587)

In the next section, we’ll dive into the different commonly used methods for univariate and multivariate imputation.

## Univariate Imputation Techniques

Let’s go over the different imputation methods and how to implement them with the open-source Feature-engine. We discussed imputation of missing data with Scikit-learn’s [simple imputer](https://www.blog.trainindata.com/imputing-missing-data-with-scikit-learns-simple-imputer/) elsewhere.

I’ll use the house prices dataset available in the `sklearn.datasets` module. The `fetch_openml` function can be used to obtain the data as a pandas data frame. The dataset has variables denoting different house features like fireplace, garage, and area, and the sale price is the dependent variable. We’ll only work with a subset of the variables.

```
# Import basic libraries
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

data = fetch_openml(name='house_prices', as_frame=True)
data = data.frame

X = data[['LotFrontage','MasVnrArea',BsmtQual','FireplaceQu','GarageYrBlt']]
y = data['SalePrice']
```

To build a machine learning model with this data, we first split it into training and testing sets randomly. The, we check the number of missing values in each column using the `isnull().mean()` methods from pandas, as shown below.:

```
# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,  y, test_size=0.3,
    random_state=0,
)

X_train.isnull().mean()
```

In the following output, we see the fraction of missing data per variable:

```
LotFrontage    0.184932
MasVnrArea     0.004892
BsmtQual       0.023483
FireplaceQu    0.467710
GarageYrBlt    0.052838
dtype: float64
```

Now, we’ll use different univariate imputation methods from the [Feature-engine](https://feature-engine.trainindata.com/) open source Python library to impute the null values in training and testing sets.

### Mean or Median imputation

[Mean and median imputation](https://feature-engine.trainindata.com/en/latest/user_guide/imputation/MeanMedianImputer.html) are two of the simplest and most quick imputation methods. Here, we replace missing observations with either the mean or median of the observed values in the same variable.

We can implement mean or median imputation easily using the `fillna()` function of Pandas, or using the `MeanMedianImputer` class of Feature-engine.

If the variable has a normal distribution, we can use either the mean or median value. If the data distribution is skewed, the median is preferred, as the mean will be biased. Check out the documentation of the [MeanMedianImputer](https://feature-engine.trainindata.com/en/1.8.x/api_doc/imputation/MeanMedianImputer.html) to learn about the different parameters and methods that can be applied with this imputer.

In the following code block, we’ll apply median imputation to the entire dataset. With fit() the imputer learns the median values per variable. With transform() it replaces the missing values with the medians.

```

# Import from feature-engine
from feature_engine.imputation import MeanMedianImputer
imputer = MeanMedianImputer(imputation_method="median")

# we fit the imputer
imputer.fit(X_train)

# Impute the values
X_train_imputed = imputer.transform(X_train)
X_test_imputed = imputer.transform(X_test)
```

**Advantages:** mean imputation and median imputation are fast and easy to implement. They work well when the fraction of missing data is small and missing data is MCAR.

**Limitations:** It can distort the variance of the variable, and co-variance with other features in the dataset.

### **Arbitrary Number Imputation**

Arbitrary number imputation consists in replacing missing values in a numeric variable with an arbitrary number like 0, 999, -999, or -1. This method is used to flag missing values explicitly.

We can apply arbitrary number imputation with Feature-engine. The [ArbitraryNumberImputer()](https://feature-engine.trainindata.com/en/latest/api_doc/imputation/ArbitraryNumberImputer.html) class can be imported from this package, which is used for imputation as shown below.

```

# Arbitrary number imputation
imputer = ArbitraryNumberImputer(arbitrary_number=-999, variables=['Age'])
imputer.fit(X_train)

# Impute the values
X_train_imputed = imputer.transform(X_train)
X_test_imputed = imputer.transform(X_test)
```

**Advantages:** Arbitrary number imputation is easy to implement, and does not make missing values look like the majority of the observations.

**Limitations:** It can distort the data distribution, and it can mask outlier values. We need to be cautious that the arbitrary value is not close to either the mean or the median, otherwise, we’d be carrying out mean or median imputation instead.

[![14 Common Feature Engineering Questions]({{ site.baseurl }}/assets/images/posts/winsorization-handling-outliers-in-machine-learning/Want-a-quick-feature-engineering-reference-Download-our-free-booklet-14-Common-Feature-Engineering-Questions-covering-encoding-scaling-missing-data-outliers-and-more-Get-the-free-bookle.png)](https://www.trainindata.com/p/14-common-feature-engineering-questions)

### **End Tail imputation**

As the name suggests, we replace missing values with values estimated from the end or extreme values of the data distribution.

The [EndTailImputer()](https://feature-engine.trainindata.com/en/latest/api_doc/imputation/EndTailImputer.html) class of `feature_engine` can be used to implement end tail imputation. We can choose whether to estimate the imputation value from the right-end (high) or left-end (low) side of the variable distribution by using the `‘tail’` parameter. The `imputation_method` parameter is used to denote the method to calculate the values to use for the imputation. I chose Gaussian, which means that the imputation value will be the mean plus 3 times the standard deviation of the variable:

```

# Import the function
from feature_engine.imputation import EndTailImputer

# Perform imputation
imputer = EndTailImputer(imputation_method='gaussian', tail='right', fold=3, variables=['Age'])

imputer.fit(X_train)

# Impute the values
X_train_imputed = imputer.transform(X_train)
X_test_imputed = imputer.transform(X_test)

```

**Advantages:** end tail imputation is more convenient as it automates arbitrary value imputation.

**Limitations:** Imputing missing values with extreme values can skew the distribution of the data.

### **Random Sample Imputation**

With random sample imputation, we replace missing values with random values taken from the observed data of the variable. It useful to impute both numerical and categorical features.

The [RandomSampleImputer()](https://feature-engine.trainindata.com/en/latest/api_doc/imputation/RandomSampleImputer.html)class of `feature-engine` applies random sample imputation. We can use a seed value to ensure reproducibility, since random extraction is, indeed, a random event.

In the following code block, I replace missing data in Age by extracting ages at random from the available data:

```

# Import the function
from feature_engine.imputation import RandomSampleImputer

# Random sample imputation
imputer = RandomSampleImputer(random_state=0, variables=['Age'])

imputer.fit(X_train)

# Impute the values
X_train_imputed = imputer.transform(X_train)
X_test_imputed = imputer.transform(X_test)
```

**Advantages:** As we choose random values from the distribution, this method maintains the original distribution and variability of the data.

**Limitations:** It is computationally more expensive, as we extract a different value for each missing observation. As random imputation is, indeed, a random process, we need to ensure we set seeds correctly, to ensure reproducibility and fair predictions for observations that are alike, apart from the missing data.

### **Categorical Variable Imputation**

When data is missing in categorical variables, we either replace them with the most frequent category, or we treat the missing data as a category of itself, hence replacing the values with a certain string, like “missing”.

The [CategoricalImputer()](https://feature-engine.trainindata.com/en/latest/api_doc/imputation/CategoricalImputer.html) class of `feature-engine` can be used to impute categorical variables with any of these methods. When we set the parameter`imputation_method='frequent`, it will replace null values with the most frequently occurring value of the feature.

```
from feature_engine.imputation import CategoricalImputer

# Mode imputation
imputer = CategoricalImputer(imputation_method='frequent', variables=['Gender'])

imputer.fit(X_train)

# Impute the values
X_train_imputed = imputer.transform(X_train)
X_test_imputed = imputer.transform(X_test)
```

**Advantages:** Imputing categorical variables with any of these methods is fast and easy.

**Limitations:** If we use the most frequent category for the imputation, it can lead to an over-representation of this category. Therefore, it’s best to use this method if the missing values make up only 5% or less of the total observations.

Alternatively, we can impute missing data in categorical variables by replacing the missing values with the string “missing” as follows:

```
from feature_engine.imputation import CategoricalImputer
imputer = CategoricalImputer()

imputer.fit(X_train)
X_test_t = imputer.transform(X_test)
```

**Advantages:** There are no assumptions made about the data distribution. It’s the most intuitive way to treat missing values in categorical data.

**Limitations:** When the proportion of missing values is very low, then creating an additional category might introduce noise, and the same is true for highly cardinal variables.

### Adding Missing Indicators

Missing indicators are binary variables used to mark observations that contain missing values. The indicators show the value 1 or True, for observations where the data is missing, and 0 or False, otherwise.

We can use [AddMissingIndicator()](https://feature-engine.trainindata.com/en/latest/api_doc/imputation/AddMissingIndicator.html#feature_engine.imputation.AddMissingIndicator) class from Feature-engine to flag missing values in both numerical and categorical variables. However, missing indicators are never used alone. This process is usually combined with other imputation methods, like mean or median imputation for numerical data or frequent category imputation for categorical data. The imputed values look like the majority of the observations in the variable, but they are, at the same time, “marked” by the missing indicators.

Using Scikit-learn’s pipeline, we can add the different imputers in series. We use the `variables` parameter in each imputer to specify the variables to impute:

```
# from feature-engine
from feature_engine.imputation import (
    AddMissingIndicator,
    MeanMedianImputer,
    CategoricalImputer,
)

#create a pipeline with the necessary imputers
pipe = Pipeline(
    [
        # 1. missing indicator
        ("missing_ind", AddMissingIndicator()),

        # 2. mode imputation for categorical cols
        ( "imputer_mode", CategoricalImputer(
          imputation_method="frequent", variables=["FireplaceQu", "BsmtQual"] ),
        ),

        # 3. median imputation for numerical cols
        (
            "imputer_median",
            MeanMedianImputer(
                imputation_method="median",
                variables=["LotFrontage", "MasVnrArea", "GarageYrBlt"] ),
        ), ])

# fit the pipeline on the training set
pipe.fit(X_train)

# Impute values
X_train_imputed = pipe.transform(X_train)
X_test_imputed = pipe.transform(X_test)
```

The method can be used when the missing data is both MCAR and MNAR. The imputed datasets can then be used to train machine learning models.

**Advantages:** This method captures the importance of missing values. It can enhance model performance or the statistical power of analysis.

**Limitations:** It increases the number of columns or dimensionality of the dataset. Often, there will be high correlation between the missing indicator variables, potentially affecting the model performance or its interpretability.

### Drop Missing Data

Dropping missing data, also called listwise deletion, consists in removing the rows or observations that contain missing values from the data. We can use the [DropMissingData()](https://feature-engine.trainindata.com/en/latest/api_doc/imputation/DropMissingData.html#feature_engine.imputation.DropMissingData) class of `feature-engine` to implement it.

This method is best suited when we have MCAR data and the number of rows with missing values is less than 5% of the complete data. Otherwise, it would reduce the dataset size, which can cause overfitting. It might also create bias in our data, as we are excluding a subset of rows.

## **Multivariate Imputation Techniques**

In multivariate imputation techniques, data from multiple columns is used to impute the nan values of a variable.

### Multivariate Imputation of Chained Equations (MICE)

In MICE, missing values are estimated by training machine learning models to predict the values that are missing. We can use simple algorithms like k-nearest neighbors or linear regression to predict the missing values, and then use the predictions for the imputation.

MICE works by training an algorithm to estimate missing data, impute those values with this algorithm, so that the imputed variable can be now used to estimate null values in a different variable, and so on. We can use ScikitLearn’s `IterativeImputer` package for this method, as shown below.

```

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Create an instance of IterativeImputer
iterative_imputer = IterativeImputer()

# Fit and transform the dataset
imputed_data = iterative_imputer.fit_transform(X_train)

# Fit and transform the dataset
imputed_data = imputer.fit_transform(X_train)
```

**Advantages:** This method accounts for the uncertainty around the missing data by generating multiple imputation rounds to obtained more unbiased estimates of the missing data.

**Limitations:** When the data size is large, performing multiple imputations can be computationally expensive.

## **Imputation of time series**

Time series data are observations indexed in chronological order. Hence, to preserve the time patters of the data, we use specific methods to replace null values, like forward fill, or last observation carried forward, or interpolation using different functions between the 2 available observed values. You can learn how to impute time series for [forecasting](https://www.blog.trainindata.com/time-series-forecasting-python/) in our course [Feature engineering for time series forecasting](https://www.trainindata.com/p/feature-engineering-for-forecasting).

[![Feature engineering for time series forecasting course]({{ site.baseurl }}/assets/images/posts/multi-seasonal-time-series-decomposition-using-mstl-in-python/feature-engineering-forecasting-course.png)](https://www.trainindata.com/p/feature-engineering-for-forecasting)

## Conclusion

Dealing with missing data is one of the most common tasks for data scientists working with real-world data science projects. Univariate imputation methods offer simplicity and ease of implementation. However, they may fall short of maintaining the complex relationships between variables in a dataset. Multivariate imputation methods provide a more sophisticated approach by considering the relationships among multiple variables. While each method has its advantages and limitations, the choice ultimately depends on the nature of the missing data, our expense limit, and the size and complexity of the dataset. If you are looking to learn more on how to improve your machine learning model during preprocessing or feature extraction, check out our end-to-end [feature engineering course](https://www.trainindata.com/p/feature-engineering-for-machine-learning) with hands-on code examples.

## Resources

- [Feature engineering for machine learning course](https://www.trainindata.com/p/feature-engineering-for-machine-learning)
- [Python Feature Engineering Cookbook](https://www.packtpub.com/en-us/product/python-feature-engineering-cookbook-9781835883587)
- [Feature engineering for time series forecasting course](https://www.trainindata.com/p/feature-engineering-for-forecasting)
