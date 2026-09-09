---
layout: post
title: "Imputing missing data with Scikit-learn’s simple imputer"
author: sole
description: "Implement the most common missing value imputation methods, like mean, median, and most frequent imputation with sklearn's simple imputer."
excerpt: "Implement the most common missing value imputation methods, like mean, median, and most frequent imputation with sklearn's simple imputer."
categories: [Data Preprocessing, Feature Engineering]
image: assets/images/posts/imputing-missing-data-with-scikit-learns-simple-imputer/simple-imputer.png
---

Missing data is an unavoidable problem in most data sources. Most machine learning model implementations by scikit-learn can’t handle missing data out-of-the-box. Hence, we must transform observations with missing data into permitted values if we want to use these models.

Fortunately, scikit-learn’s simple imputer can implement the most commonly used imputation methods. With the simple imputer we can do:

- Mean and median imputation
- Imputation with the most frequent value
- Imputation with an arbitrary value

And in this article, we will discuss how these methods work and then show how to code them with the simple imputer from Scikit-learn. Let’s dive in.

## Missing data imputation

Missing data imputation consists of replacing missing data with an estimate of the missing value and is a common [data preprocessing](https://www.blog.trainindata.com/mastering-data-preprocessing-techniques/) step in most, if not all, data science projects.

Machine learning algorithms make computations by using numbers as inputs. They, in general, cannot operate with nan values the same way they cannot process the strings found in categorical data. This is why missing value imputation together with categorical encoding processes, like [one-hot encoding](https://www.blog.trainindata.com/one-hot-encoding-categorical-variables/), are some of the first steps in any data analysis and machine learning project.

### Univariate imputation

There are many methods to handle missing values that we can use to replace nan with permitted numbers. Mean, median, mode, and arbitrary imputation are the most common ones. They are simple, and they also have their limitations, like masking nan as common values and distorting the variable’s distribution. Nevertheless, they are widely adopted for their simplicity, and if we add a placeholder to mark the fact that the value was missing, we can overcome some of their limitations.

![Diagram showing the missing data imputation methods and highlighting their limitations]({{ site.baseurl }}/assets/images/posts/imputing-missing-data-with-scikit-learns-simple-imputer/04_MissingData.png)

Additional univariate imputation methods not supported by scikit-learn include [random sample imputation](https://feature-engine.trainindata.com/en/latest/user_guide/imputation/RandomSampleImputer.html) and end of distribution imputation, both available on the alternative feature engineering open source Python library [Feature-engine](https://feature-engine.trainindata.com).

### Multivariate imputation

We can also replace missing values using multivariate imputation. With these methods, we consider more than one variable in the dataset to find the best estimates for the nan values. We can use regression to find these estimates, by using Scikit-learn’s iterative imputer in combination with any regression algorithm. And we can also use nearest neighbours to find the best nan replacements, with sklearn’s knn imputer.

For more details regarding how these and other data preprocessing tools from sklearn work, check out our course [Feature Engineering for Machine Learning](https://www.trainindata.com/p/feature-engineering-for-machine-learning) or our [Python Feature Engineering Cookbook](https://www.packtpub.com/en-us/product/python-feature-engineering-cookbook-9781835883587).

[![Feature Engineering for Machine Learning course]({{ site.baseurl }}/assets/images/posts/winsorization-handling-outliers-in-machine-learning/feature-engineering-machine-learning-course.jpg)](https://www.trainindata.com/p/feature-engineering-for-machine-learning)

In the rest of this tutorial, we’ll focus on those imputation techniques supported by sklearn’s SimpleImputer.

## Mean or median imputation

Mean or median imputation consists of replacing missing data with the variable’s mean or median value.

It is important to determine the mean or median using the train set, and then use these values to impute the train and test sets, and all future data. This is to avoid data leakage.

Let’s start with the imports:

```
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer
```

Let’s load the [credit approval dataset](https://archive.ics.uci.edu/dataset/27/credit+approval) directly from the UCI Machine Learning Repository, and separate it into a training and a testing set:

```
from ucimlrepo import fetch_ucirepo
import random
import numpy as np

credit_approval = fetch_ucirepo(id=27)
data = credit_approval.data.features
data["target"] = credit_approval.data.targets["A16"].map({"+": 1, "-": 0})

# reproduce the missing values injected in the original prepared dataset
random.seed(9001)
for i, var in enumerate(["A3", "A8", "A9", "A10"]):
    idx = list(set(random.randint(i, len(data)) for _ in range(100)))
    data.loc[idx, var] = np.nan

X_train, X_test, y_train, y_test = train_test_split(

    data.drop("target", axis=1),

    data["target"],

    test_size=0.3,

    random_state=0,

)

```

Let’s make a list with the name of the variables with numerical data type:

```
numeric_vars = X_train.select_dtypes(
    exclude="O").columns.to_list()
```

These are the names of the numerical variables:

```
['A2', 'A3', 'A8', 'A11', 'A14', 'A15'].
```

Let’s set up the simple imputer to replace missing data with the median:

```
imputer = SimpleImputer(strategy="median")
```

If you want to perform mean imputation, instead of median imputation, using simpleimputer, you need to pass “mean” to the strategy instead.

To restrict the imputation to the numerical variables, we need the ColumnTransformer():

```
ct = ColumnTransformer(

    [("imputer",imputer, numeric_vars)],

    remainder="passthrough"

    ) .set_output(transform="pandas")
```

Scikit-learn can return numpy arrays, pandas dataframes or polar frames, depending on how we set out the transform output. By default, it returns numpy arrays. But here, we set it up to return pandas dataframes.

Fit the simple imputer to the train set so that it learns the median values:

```
ct.fit(X_train)
```

Let’s check out the learned median values:

```
ct.named_transformers_.imputer.statistics_
```

The previous command returns the median values per variable:

```
array([ 28.835,   2.75 ,   1.   ,   0.   , 160.   ,   6.   ])
```

Let’s replace missing values with the median:

```
X_train_t = ct.transform(X_train)

X_test_t = ct.transform(X_test)
```

Let’s display the resulting training set:

```
print(X_train_t.head())
```

We see the resulting pandas dataframe in the following image:

```
     imputer__A2  imputer__A3  imputer__A8  imputer__A11  imputer__A14  \
596        46.08        3.000        2.375           8.0         396.0
303        15.92        2.875        0.085           0.0         120.0
204        36.33        2.125        0.085           1.0          50.0
351        22.17        0.585        0.000           0.0         100.0
118        57.83        7.040       14.000           6.0         360.0

     imputer__A15 remainder__A1 remainder__A4 remainder__A5 remainder__A6  \
596        4159.0             a             u             g             c
303           0.0             a             u             g             q
204        1187.0             b             y             p             w
351           0.0             b             y             p            ff
118        1332.0             b             u             g             m

    remainder__A7 remainder__A9 remainder__A10 remainder__A12 remainder__A13
596             v           NaN              t              t              g
303             v           NaN              f              f              g
204             v             t              t              f              g
351            ff             f              f              f              g
118             v             t              t              t              g
```

And that’s it, we’ve not replaced missing data in numerical variables with their median values.

**Tip**: You can also perform median imputation using Feature-engine’s [MeanMedianImputer](https://feature-engine.trainindata.com/en/latest/api_doc/imputation/MeanMedianImputer.html). Unlike SimpleImputer, Feature-engine’s MeanMedianImputation allows you to restrict the imputation to selected variables from within the transformer, so you don’t need supporting classes like the ColumnTransformer.

[![14 Common Feature Engineering Questions]({{ site.baseurl }}/assets/images/posts/winsorization-handling-outliers-in-machine-learning/Want-a-quick-feature-engineering-reference-Download-our-free-booklet-14-Common-Feature-Engineering-Questions-covering-encoding-scaling-missing-data-outliers-and-more-Get-the-free-bookle.png)](https://www.trainindata.com/p/14-common-feature-engineering-questions)

## Most frequent category imputation

We used the mean and the median to impute numerical variables. What would be the equivalent for categorical data? The mode: that is, the most frequent value or category.

Here again, to avoid data leakage, we find the frequent categories from the train set. Then, we use these values to impute the train, test, and future datasets.

Let’s import **pandas** and the required functions and classes from scikit-learn:

```
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer
```

Let’s load the credit risk dataset and split it into train and test:

```
from ucimlrepo import fetch_ucirepo
import random
import numpy as np

credit_approval = fetch_ucirepo(id=27)
data = credit_approval.data.features
data["target"] = credit_approval.data.targets["A16"].map({"+": 1, "-": 0})

# reproduce the missing values injected in the original prepared dataset
random.seed(9001)
for i, var in enumerate(["A3", "A8", "A9", "A10"]):
    idx = list(set(random.randint(i, len(data)) for _ in range(100)))
    data.loc[idx, var] = np.nan

X_train, X_test, y_train, y_test = train_test_split(

    data.drop("target", axis=1),

    data["target"],

    test_size=0.3,

    random_state=0,

)
```

Let’s capture the categorical variable names in a list:

```
categorical_vars = X_train.select_dtypes(

    include="O").columns.to_list()
```

Let’s set up the simple imputer to find the most frequent category:

```
imputer = SimpleImputer(strategy='most_frequent')
```

Let’s restrict the imputation to the categorical variables:

```
ct = ColumnTransformer(

    [("imputer",imputer, categorical_vars)],

    remainder="passthrough"

    ).set_output(transform="pandas")
```

Now, we fit the imputer to the train set so that it learns the most frequent values:

```
ct.fit(X_train)
```

Let’s take a look at the most frequent values learned by the simple imputer:

```
ct.named_transformers_.imputer.statistics_
```

The previous command returns the most frequent values per variable:

```
array(['b', 'u', 'g', 'c', 'v', 't', 'f', 'f', 'g'], dtype=object)
```

Finally, let’s replace missing values with the frequent categories:

```
X_train_t = ct.transform(X_train)

X_test_t = ct.transform(X_test)
```

Note, that the ColumnTransformer() changes the names of the variables. The imputed variables show the prefix “imputer” and the remaining the prefix “remainder”.

**Tip**: you can also impute categorical data with the most frequent category by using Feature-engine’s [CategoricalImputer](https://feature-engine.trainindata.com/en/latest/api_doc/imputation/CategoricalImputer.html). The advantage is that it allows you to restrict the imputation, right from within the imputer, without the need of an additional class like the ColumnTransformer and it does not change the names of the variables after the transformation.

## Replacing missing values with an arbitrary value

We can replace missing data with an arbitrary or constant value. If the variable data types are numerical, we normally use values like **999**, **9999**. For categorical variables, we use a specific string, like “missing”, or “other”.

When replacing missing values with arbitrary numbers, we need to be careful not to select a value close to the mean, the median, or any other category name that already exists. Otherwise, we mask the fact that these observations contain nan.

Let’s import **pandas** and the required functions and classes:

```
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer
```

Let’s load the dataset and split it into a training and a testing set:

```
from ucimlrepo import fetch_ucirepo
import random
import numpy as np

credit_approval = fetch_ucirepo(id=27)
data = credit_approval.data.features
data["target"] = credit_approval.data.targets["A16"].map({"+": 1, "-": 0})

# reproduce the missing values injected in the original prepared dataset
random.seed(9001)
for i, var in enumerate(["A3", "A8", "A9", "A10"]):
    idx = list(set(random.randint(i, len(data)) for _ in range(100)))
    data.loc[idx, var] = np.nan

X_train, X_test, y_train, y_test = train_test_split(

    data.drop("target", axis=1),

    data["target"],

    test_size=0.3,

    random_state=0,

)
```

We’ll use **99** for the imputation because it is bigger than the maximum values of the numerical variables in this data set*.* We set up the simple imputer to replace missing values with **99**:

```
imputer = SimpleImputer(

    strategy='constant', fill_value=99

    )
```

Let’s fit the imputer to a slice of the train set containing the variables to impute:

```
imputer.fit(X_train[["A2", "A3", "A8", "A11"]])
```

Now, we replace the missing values with **99** in the desired variables:

```
X_train_t[["A2", "A3", "A8", "A11"]] = imputer.transform(

    X_train[["A2", "A3", "A8", "A11"]]

)

X_test_t[["A2", "A3", "A8", "A11"]] = imputer.transform(

    X_test[["A2", "A3", "A8", "A11"]]
)
```

Go ahead and check the lack of missing values by executing X_test_t[[“A2”, “A3”, “A8”, “A11”]].isnull().sum().

**Tip:** We can also impute missing values using Feature-engine’s [ArbitraryNumberImputer](https://feature-engine.trainindata.com/en/latest/api_doc/imputation/ArbitraryNumberImputer.html). The advantage is that you don’t need to slice the dataframe as we did here for the simple imputer.

## Marking imputed values

So far, we focused on replacing missing data with estimates of their values. In addition, we can add missing indicators to mark observations where values were missing.

A missing indicator is a binary variable that takes the value 1 or True to indicate whether a value was missing, or 0 or False otherwise. It is common practice to replace missing observations with the mean, median, or most frequent category while simultaneously marking those missing observations with missing indicators.

Let’s begin by making some imports and loading the data:

```
import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline


from ucimlrepo import fetch_ucirepo
import random
import numpy as np

credit_approval = fetch_ucirepo(id=27)
data = credit_approval.data.features
data["target"] = credit_approval.data.targets["A16"].map({"+": 1, "-": 0})

# reproduce the missing values injected in the original prepared dataset
random.seed(9001)
for i, var in enumerate(["A3", "A8", "A9", "A10"]):
    idx = list(set(random.randint(i, len(data)) for _ in range(100)))
    data.loc[idx, var] = np.nan

X_train, X_test, y_train, y_test = train_test_split(

    data.drop("target", axis=1),

    data["target"],

    test_size=0.3,

    random_state=0,

)
```

Let’s add missing indicators and simultaneously impute numerical and categorical variables with the mean and most frequent categories respectively, utilizing scikit-learn.

We first make a list with the names of the numerical and categorical variables:

```
numvars = X_train.select_dtypes(

    exclude="O").columns.to_list()

catvars = X_train.select_dtypes(

    include="O").columns.to_list()
```

We set up a pipeline to perform mean and frequent category imputation while marking the missing data:

```
pipe = ColumnTransformer([

   ("num_imputer", SimpleImputer(

         strategy="mean", add_indicator=True), numvars),

   ("cat_imputer", SimpleImputer(

         strategy="most_frequent", add_indicator=True), catvars),

]).set_output(transform="pandas")
```

Now, let’s carry out the imputation:

```
X_train_t = pipe.fit_transform(X_train)

X_test_t = pipe.transform(X_test)
```

Make sure to explore X_train_t.head() to get familiar with the pipeline’s output.

Here, we added missing indicators by using SimpleImputer. Scikit-learn has the MissingIndicator() transformer that just adds missing indicators. So if you want to do multivariate imputation while adding placeholders for the nan values, use this class instead. Alternatively, check out Feature-engine’s [AddMissingIndicator](https://feature-engine.trainindata.com/en/latest/api_doc/imputation/AddMissingIndicator.html) class, which allows you to add indicators only for the variables you pre-select, out of the box.

## Conclusion

In this article, we’ve seen how to carry out the most commonly used missing value imputation methods by using the Scikit-learn API. The advantage of using Scikit-learn is that it’s fast, as it runs on Numpy, and it can operate over pandas dataframes, polars or numpy arrays.

The limitations of scikit-learn’s API, is that their transformers transform the entire dataset, so to restrict the transformation to a group of variables, we need to use the ColumnTransformer. This class in turn, changes the names of the variables, so it takes some practice to learn how it works and the variable names that we will obtain.

As an alternative, explore Feature-engine. Feature-engine supports the same univariate imputation methods but its transformers can restrict the operations to variable subgroups without the need of additional classes, or slicing the dataframe. It also works over pandas dataframes or numpy arrays, but it returns pandas dataframes by default.

[![Python Feature Engineering Cookbook book cover]({{ site.baseurl }}/assets/images/posts/detect-outliers-in-python/PFEC2ED.png)](https://www.packtpub.com/en-us/product/python-feature-engineering-cookbook-9781835883587)

For more details regarding how these and other data preprocessing tools from sklearn work, check out our course [Feature Engineering for Machine Learning](https://www.trainindata.com/p/feature-engineering-for-machine-learning) or our [Python Feature Engineering Cookbook](https://www.packtpub.com/en-us/product/python-feature-engineering-cookbook-9781835883587).
