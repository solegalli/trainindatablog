---
layout: post
title: "One-hot encoding categorical variables"
author: sole
excerpt: Find out how to encode categorical variables using one-hot.
categories: [ Feature engineering, Python, Machine learning, categorical encoding ]
image: assets/images/posts/ohe/cover.gif
---

Categorical variables are those whose values are selected from a group of categories. For example, 
the variable "marital status," with the values "never married," "married, divorced, or "widowed," 
is categorical.

Categorical data is a common occurrence in many data science projects. To train machine 
learning algorithms like linear regression, decision trees, or random forests, in particular 
if we are utilizing Scikit-learn, all predictor variables must be numerical. Even for 
deep learning models we need numerical features. Hence, we need to encode categorical 
values into numerical values. The act of replacing categories with numbers is called 
categorical encoding.

There are many categorical encoding strategies, like ordinal encoding (sometimes called 
label encoding), target or mean encoding, encoding with counts or frequencies, and one hot encoding.

In this article, we will discuss one hot encoding and its variations. So let’s get started!


*For tutorials and step by step code implementations on other categorical encoding methods, visit our course 
[Feature engineering for machine learning](https://www.trainindata.com/p/feature-engineering-for-machine-learning) 
or check out our book [Python Feature Engineering Cookbook](https://www.amazon.com/Python-Feature-Engineering-Cookbook-transforming-dp-1804611301/dp/1804611301).*


## One-hot encoding

In one-hot encoding, we represent a categorical variable as a group of binary variables, 
where each binary variable represents one category. The binary variable takes the integer 
value 1 if the category is present, or 0 otherwise.

The following table shows the one hot encoded representation of the categorical variable 
**Color** with the values of **red**, **blue**, and **green**:

![one hot encoding]({{ site.baseurl }}/assets/images/posts/ohe/color-ohe.png)


## K vs k-1 binary variables

A categorical variable with k unique categories can be encoded in k-1 binary variables. To 
capture all of the information for the Color variable, which has three categories (k = 3; red, 
blue, and green), we need to create two (k - 1 = 2) binary variables.

- If the observation is red, it will be captured by the variable red (red = 1, blue = 0).
- If the observation is blue, it will be captured by the variable blue (red = 0, blue = 1).
- If the observation is green, it will be captured by the combination of red and blue (red = 0, blue = 0).

![one hot encoding into k minus 1 variables]({{ site.baseurl }}/assets/images/posts/ohe/ohe-keminus1.png)

Encoding into k-1 binary variables is well suited for linear models. 

There are, however, a few occasions in which we may prefer to encode the categorical variables 
with k binary variables:

- When training decision trees, since they do not evaluate the entire feature space at the same time.
- When selecting features recursively.
- When determining the importance of each category within a variable.

 
## Special case: binary variables

Binary variables, such as "is underage," which takes the values yes or no, are always encoded 
into one binary variable, because if the minor is underage, the binary takes the value 1, 
otherwise it takes the value 0.

![one hot encoding of binary variables]({{ site.baseurl }}/assets/images/posts/ohe/ohe-binary.png)


## Python implementation of one hot encoding

Let’s now compare the one-hot encoding implementations of pandas, scikit-learn, Feature-engine 
and Category Encoders.

We first make imports, load the data set and separate it into a training and a testing set:

```
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('https://www.openml.org/data/get_csv/16826755/phpMYEkMl')
df = df.replace('?', np.nan)

def get_first_cabin(row):
    try:
        return row.split()[0]
    except:
        return np.nan

df['cabin'] = df['cabin'].apply(get_first_cabin)
df['cabin'] = df['cabin'].str[0]
df.fillna("Missing", inplace=True)

usecols=['sex', 'embarked', 'cabin', 'pclass', 'sibsp', 'parch', 'survived']

df[usefols].head()
```

Below we see a view of the original dataset:

![View of the titanic dataset]({{ site.baseurl }}/assets/images/posts/ohe/titanic-full.png)

Let's now split it into a training and a testing set:

```
X_train, X_test, y_train, y_test = train_test_split(
    df[usecols],
    df['survived'],
    test_size=0.3,
    random_state=0,
)
```

### Pandas

The Titanic dataset has both categorical and numerical data. Let's one hot encode all the 
categorical columns into k-1 binaries, capturing the result in a new dataframe:

```
X_train_enc = pd.get_dummies(X_train,  drop_first=True)
X_test_enc = pd.get_dummies(X_test , drop_first=True)

X_train_enc.head()
```

![One hot encoding with pandas]({{ site.baseurl }}/assets/images/posts/ohe/pd-get-dummies.png)

The method `get_dummies` from pandas will automatically encode all variables of data type 
categorical or object in the dataframe. Note how it replaces the categorical variables by
the dummy variables in the previous image.

We can now use our categorical variables as predictors to train algorithms of artificial 
intelligence.


### Scikit-learn

Let’s now carry out one hot encoding utilizing Scikit-learn. Let’s import the one hot encoder 
and a class to transform variable subsets:

```
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
``` 

Let's set up the transformer to encode into k-1 binary variables and return a dataframe:

```
encoder = OneHotEncoder(drop=”first”, sparse=False)
```

We need to set `sparse` to `False`, so that the encoder returns and array instead of a sparse matrix. 
This is also necessary if we want to return dataframes as outputs.

```
ohe = ColumnTransformer(
    transformers=[("ohe", encoder, ["sex", "embarked", "cabin"]),],
    remainder="passthrough",
)

ohe.set_output(transform="pandas")
```

See the recently released [set_output API](https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_set_output.html)
for more information about how to make Scikit-learn transformers return pandas dataframes 
instead of arrays.

Let’s now fit the encoder to the training data and then transform the variables:

```
ohe.fit(X_train)

X_train_t = ohe.transform(X_train)
X_test_t = ohe.transform(X_test)

X_train_t.head()
```

![One hot encoding with scikit-learn]({{ site.baseurl }}/assets/images/posts/ohe/sklearn-ohe.png)

The binary vectors corresponding to the one hot encoded variable are shown to the left of 
the dataframe. After that, the column transformer appended the original numerical variables, 
that were not transformed.

### Feature-engine

Let’s now see the advantage of using the `OneHotEncoder` from Feature-engine. Let's 
import the encoder:

```
from feature_engine.encoding import OneHotEncoder
```

Next, let's set up the encoder to return k-1 binary variables:

```
ohe_enc = OneHotEncoder(drop_last=True)
```

Now, let’s transform the variables. With Feature-engine, the input data can be the entire 
dataframe, yet the transformer will encode only the categorical variables:

```
X_train_enc = ohe_enc.fit_transform(X_train)
X_test_enc = ohe.transform(X_test)

X_train_enc.head()
```

We see that Feature-engine automatically appends the dummy variables as new columns to the 
input dataframe:


![One hot encoding with Feature-engine]({{ site.baseurl }}/assets/images/posts/ohe/pd-get-dummies.png)

 
### Category encoders


Finally, let’s carry out one hot encoding with Category encoders. Let’s import the encoder:

```
from category_encoders.one_hot import OneHotEncoder
```

Let's set up the encoder to return k-1 binary variables adding the categories as column names:

```
ohe_enc = OneHotEncoder(use_cat_names=True)
```

And now, let’s fit the transformer and encode the variables:

```
ohe_enc.fit(X_train)

X_train_enc = ohe_enc.transform(X_train)
X_test_enc = ohe_enc.transform(X_test)
```

Category Encoders' `OneHotEncoder()` returns a copy of the original dataset plus the binary 
variables and without the original categorical variables. Thus, this data is ready to train 
machine learning models.

## Comparing the Python implementations

We compared the one-hot encoding implementation of pandas, scikit-learn, Feature-engine and 
Category Encoders. Each implementation has advantages and shortcomings.

Let’s summarize the main characteristics of each library:

Pandas, Feature-engine and Category Encoders can automatically identify and encode categorical 
variables, that is, those of type object or categorical. Scikit-learn’s `OneHotEncoder()`, on 
the other hand, will encode all variables in the dataset.

With pandas, Feature-engine and Category Encoders, we can encode only a subset of the variables, 
indicating their names in a list when calling the methods or transformers. With Scikit-learn 
we need to use an additional class, the ColumnTransformer(), to slice the data before the 
transformation.

![Comparing the one hot encoding implementations]({{ site.baseurl }}/assets/images/posts/ohe/table.png)

With Feature-engine and Category Encoders, the dummy variables are added to the original 
dataset and the categorical variables are removed after the encoding. With Scikit-learn, we 
need to use the ColumnTransformer, and with pandas, we need to carry out these procedures 
manually.

Finally, using the `OneHotEncoder()` from Scikit-learn, Feature-engine and Category Encoders, we 
can perform the encoding step within a Scikit-learn Pipeline, which is more convenient if we 
have various feature engineering steps or want to put the Pipelines into production. Pandas 
`get_dummies()` is otherwise well suited for data analysis and visualization.


## Encoding a subset of categories using one-hot

Oftentimes, we are only interested in a subset of the values of a categorical variable. For 
example, for the variable **City**, we may be interested only in the big capitals. Therefore, we 
can create binary variables to represent only those categories.

Let’s see how to encode only a subset of the categories of categorical variables utilizing 
numpy and scikit-learn.

Let's first import the Python libraries and functions:

```
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
```

Let's load the credit approval dataset (for guidelines to prepare the data check this 
[github repo](https://github.com/solegalli/Python-Feature-Engineering-Cookbook-Second-Edition/blob/main/ch02-categorical-encoding/donwload-prepare-store-credit-approval-dataset.ipynb):

```
data = pd.read_csv(“credit_approval_uci.csv”)


X_train, X_test, y_train, y_test = train_test_split(
    data.drop(labels=[“target”], axis=1),
    data[“target”], 
    test_size=0.3, 
    random_state=0,
)
```

Let's inspect the unique categories of the A6 variable:

```
X_train[“A6”].unique()
```
 
The unique values of A6 are displayed in the following output:

```
array(['c', 'q', 'w', 'ff', 'm', 'i', 'e', 'cc', 'x', 'd', 'k', 'j', 'Missing, 'aa', 'r'], dtype=object)
```

Let’s make a list with the categories from A6 that we want to encode using one-hot:

```
categories = ["aa", "cc", "ff"]
```

Let’s add the binary variables to the train and test sets for the former categories:

```
for category in categories:
    X_train[f"A6_{category}"] = np.where(X_train["A6"] == category, 1, 0)
    X_test[f"A6_{category}"] = np.where(X_test["A6"] == category, 1, 0)
```

Let’s visualize the data:

```
X_test.head()
```

We can see the binary variables at the right of the dataframe in the output of the precedent command:

![One hot encoding of specific categories]({{ site.baseurl }}/assets/images/posts/ohe/subsets-pandas.png)

Now, let’s automate the procedure utilizing scikit-learn. Let’s import the `OneHotEncoder()`
and the `ColumnTransformer()`:

```
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
```

Let’s set up the encoder to create binary variables for some categories in 2 of the variables:

```
encoder = OneHotEncoder(
    categories=[["aa", "cc", "ff"], ['ff', 'dd', 'bb']],
    handle_unknown="ignore",
    sparse=False,
)
```

Let’s set up the ColumnTransformer() to encode the variables A6 and A7 and to return all 
variables in the final output:

```
ct = ColumnTransformer(
    [("encoder", encoder, ["A6", "A7"])],
    remainder="passthrough",
)
```

Let’s fit the encoder to the train set:

```
ct.fit(X_train)
```

Let’s replace the categorical variables by the one-hot encoded in train and test sets:

```
X_train_enc = ct.transform(X_train)
X_test_enc = ct.transform(X_test)
```

Let’s obtain the name of the final features:

```
ct.get_feature_names_out()
```

We can see the name of the variables in the final output in the return of the precedent command:

```
array(['encoder__A6_aa', 'encoder__A6_cc', 'encoder__A6_ff',
       'encoder__A7_ff', 'encoder__A7_dd', 'encoder__A7_bb',
       'remainder__A1', 'remainder__A2', 'remainder__A3', 'remainder__A4',
       'remainder__A5', 'remainder__A8', 'remainder__A9',
       'remainder__A10', 'remainder__A11', 'remainder__A12',
       'remainder__A13', 'remainder__A14', 'remainder__A15'], dtype=object)
```

Finally, let’s transform the Numpy array into a pandas dataframe, add the variable names 
and display the first five rows:

```
X_test_enc = pd.DataFrame(X_test_enc)
X_test_enc.columns = ct.get_feature_names_out()

X_test_enc.head()
```

We can see the resulting dataframe in the return of the precedent command:

```
  encoder__A6_aa encoder__A6_cc encoder__A6_ff encoder__A7_ff encoder__A7_dd  \
0            0.0            0.0            0.0            0.0            0.0  
1            0.0            0.0            0.0            0.0            0.0  
2            0.0            1.0            0.0            0.0            0.0  
3            0.0            0.0            0.0            0.0            0.0  
4            0.0            0.0            0.0            0.0            0.0  

  encoder__A7_bb remainder__A1 remainder__A2 remainder__A3 remainder__A4  \
0            0.0             a         45.83          10.5             u  
1            0.0             b         64.08          20.0             u  
2            0.0             a         31.25          3.75             u  
3            0.0             b         39.25           9.5             u  
4            0.0             a         26.17           2.0             u  
 
  remainder__A5 remainder__A8 remainder__A9 remainder__A10 remainder__A11  \
0             g           5.0             t              t              7  
1             g          17.5             t              t              9  
2             g         0.625             t              t              9  
3             g           6.5             t              t             14  
4             g           0.0             f              f              0  
 
  remainder__A12 remainder__A13 remainder__A14 remainder__A15 
0              t              g            0.0              0 
1              t              g            0.0           1000 
2              t              g          181.0              0 
3              f              g          240.0           4607 
4              t              g          276.0              1 
```

Remember that now, instead of this workaround to transform the numpy array into a dataframe
we can just set the output to pandas with the `set_ouput` method.

## One-hot encoding of frequent categories

One-hot encoding represents each category of a categorical variable with a binary variable. 
Hence, one-hot encoding of variables with high cardinality or datasets with multiple categorical 
features can expand the feature space dramatically.

To reduce the number of binary variables, we can perform one-hot encoding of the most frequent 
categories only. One-hot encoding of top categories is equivalent to treating the less frequent 
categories as a single, unique category.

Let’s implement one-hot encoding of the most popular  categories using pandas and Feature-engine.

Let's first import the necessary Python libraries and get the dataset ready:

``` 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from feature_engine.encoding import OneHotEncoder
```

Let's load the dataset and divide into train and test sets:

```
data = pd.read_csv(“credit_approval_uci.csv”)

X_train, X_test, y_train, y_test = train_test_split(
    data.drop(labels=[“target”], axis=1),
    data[“target”],
    test_size=0.3,
    random_state=0,
)
```

Let's inspect the unique categories of the A6 variable:

```
X_train[“A6”].unique()
```

The unique values of A6 are displayed in the following output:

```
array(['c', 'q', 'w', 'ff', 'm', 'i', 'e', 'cc', 'x', 'd', 'k', 'j', 'Missing, 'aa', 'r'], dtype=object)
```
 
Let's count the number of observations per category of A6, sort them in decreasing order, 
and then display the five most frequent categories:

```
X_train[“A6”].value_counts().sort_values(ascending=False).head(5)
```

We can see the five most frequent categories and the number of observations per category in 
the output of the previous step:

```
c 93
q 56
w 48
i 41
ff 38
 ```

Let's capture the most frequent categories of A6 in a list:

```
top_5 = [x for x in X_train['A6'].value_counts().sort_values( ascending=False).head(5).index]
```

Now, let's add a binary variable per top category in the train and test sets:

```
for label in top_5:
   X_train[f"A6_{label}"] = np.where(X_train["A6"] == label, 1, 0)
   X_test[f"A6_{label}"] = np.where(X_test["A6"] == label, 1, 0)
```

Let's output the top 10 rows of the original and encoded variable, A6, in the train set:

```
print(X_train[['A6'] + [f"A6_{label}" for label in top_5]].head(10))
```

We can see in the output of the previous step, the original A6 variable, followed by the new 
binary variables and some of their values:

```
     A6  A6_c  A6_q  A6_w  A6_i  A6_ff
596   c     1     0     0     0      0
303   q     0     1     0     0      0
204   w     0     0     1     0      0
351  ff     0     0     0     0      1
118   m     0     0     0     0      0
247   q     0     1     0     0      0
652   i     0     0     0     1      0
513   e     0     0     0     0      0
230  cc     0     0     0     0      0
250   e     0     0     0     0      0
```

We can simplify one-hot encoding of frequent categories, with Feature-engine. Let's set up 
the one-hot encoder to encode the five most frequent categories of the variables A6 and A7:

```
ohe_enc = OneHotEncoder(top_categories=5, variables=['A6', 'A7'])
```

Let's fit the encoder to the train set so that it learns and stores the most frequent categories 
of A6 and A7: 

```
ohe_enc.fit(X_train)
```
 
Finally, let's encode A6 and A7 in the train and test sets:

```
X_train_enc = ohe_enc.transform(X_train)
X_test_enc = ohe_enc.transform(X_test)

X_train_enc.head()
```

![One hot encoding of top categories]({{ site.baseurl }}/assets/images/posts/ohe/top-cat-fe.png)

And that's it! We've now covered one hot encoding with a lot of detail!

## More categorical encoding in Python

If you want to learn more about categorical encoding and how to carry it out in Python, check out our course and book:

- course: [Feature Engineering for Machine Learning](https://www.trainindata.com/p/feature-engineering-for-machine-learning)
- book: [Python Feature Engineering Cookbook](https://www.amazon.com/Python-Feature-Engineering-Cookbook-transforming-dp-1804611301/dp/1804611301)

Both the course and the book contain a great amount of information regarding:

- the logic of the methods
- their advantages and limitations
- code examples using real world datasets

Course and book are suitable for beginner and intermediate data scientists alike.
