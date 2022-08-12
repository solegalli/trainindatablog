---
layout: post
title:  "Mutual information with Python"
author: sole
categories: [ Feature selection, Python, Machine learning ]
image: assets/images/posts/mutual_info_cover.png
---

Mutual information (MI) is a non-negative value that measures the mutual dependence between two random variables. The 
mutual information measures the amount of information we can know from one variable by observing the values of the 
second variable.


The mutual information is a good alternative to Pearson’s correlation coefficient, because it is able to measure any 
type of relationship between variables, not just linear associations. And also, it is suitable for both continuous and 
discrete variables, unlike Pearson’s correlation coefficient.


MI is closely related to the concept of entropy. Thus, I will first introduce the entropy, then show how we compute the 
entropy of a discrete variable. Next, I  will show how to compute the MI between discrete variables. I will extend the 
definition of MI for continuous variables. And finally, I will finish with a Python implementation of feature selection 
based on MI.


In summary, in the following paragraphs we will discuss:

- Entropy.
- Related entropy.
- Mutual information of discrete variables.
- Mutual information of continuous variables.
- Feature selection based on MI with Python.

*For tutorials on feature selection using the mutual information and other methods, check out our course 
[Feature Selection for Machine Learning](https://www.trainindata.com/p/feature-selection-for-machine-learning) or our 
book [Feature Selection in Machine Learning with Python](https://leanpub.com/feature-selection-in-machine-learning/).*

## Entropy

The entropy of a variable is a measure of the information, or alternatively, the "uncertainty," of the variable's possible values.


Entropy is defined as:

![]({{ site.baseurl }}/assets/images/posts/mi_eq.png)


where H(X) is the Shannon entropy of X and p(x) is the probability of the values of X. If the logarithm base is 2, then 
the unit of the entropy is a bit. If the logarithm base is e, then the unit is the nat. If the logarithm base is 10, the 
unit is the hartley.


To illustrate with an example, the entropy of a fair coin toss is 1 bit:

![]({{ site.baseurl }}/assets/images/posts/mi_eq2.png)

Note that the log in base 2 of 0.5 is -1.

To calculate the entropy with Python we can use the open source library Scipy:

```
import numpy as np
from scipy.stats import entropy
coin_toss = [0.5, 0.5]
entropy(coin_toss, base=2)
```

which returns 1.

## Relative entropy


The relative entropy measures the distance between two distributions and it is also called Kullback-Leibler distance. 
It is given by:


![]({{ site.baseurl }}/assets/images/posts/mi_eq3.png)

where p(x) and q(x) are two probability distributions.

## Mutual information

Utilizing the relative entropy, we can now define the MI. We define the MI as the relative entropy between the joint 
distribution of the two variables and the product of their marginal distributions.

Thus, the MI is given by:

![]({{ site.baseurl }}/assets/images/posts/mi_eq4.png)

where I(X,Y) is the MI between variables x and y, the joint probability of the two variables is p(x,y), and their marginal 
probabilities are p(x) and p(y).


Note that the MI can be equal or greater than 0. When p(x,y) = p(x) p(y), the MI is 0. The joint probability is equal to 
the product of the marginals when there is no association between the variables. When the MI is 0, then knowing the 
values of x does not tells us anything about y, and vice versa, that is knowing y, does not tell us anything about x. 

### MI between discrete variables

To illustrate the calculation of the MI with an example, let's say we have the following contingency table of survival 
on the Titanic based on gender:


| | Female | Male | Total |
| -| ------- | ----- | -----|
| Not survived | 89 | 483 | 571 |
| Survived | 230| 112 | 342 |
| Total | 319 | 595 | 914 |

With the table frequencies, we can create probability estimates by dividing the counts in each cell by the total number 
of passengers, which is 914:


| | Female | Male | Total |
| -| ------- | ----- | -----|
| Not survived | 0.09 | 0.52 | 0.62 |
| Survived | 0.25| 0.12 | 0.37 |
| Total | 0.34 | 0.65 | 1 |


The MI for the the variables survival and gender is:

![walking]({{ site.baseurl }}/assets/images/posts/ch4-eq4.png)

Thus, I(X,Y) = 0.2015.

The MI of 0.2015, which is bigger than 0, indicates that by knowing the gender of the passenger, we know more about 
their probability of survival.


To calculate the MI between discrete variables in Python, we can use the mutual_info_score from Scikit-learn. We can 
provide the vectors with the observations like this:

```
from sklearn.metrics import mutual_info_score
a = ['A', 'B', 'A', 'A', 'B', 'B', 'A', 'A', 'B', 'B']
x = ['X', 'X', 'X', 'Y', 'Z', 'Z', 'Y', 'Y', 'Z', 'Z']
mi = mutual_info_score(a,x)
```

which will return mi = 0.5021929300715018.

For the `mutual_info_score`, a and x should be array-like vectors, i.e., lists, numpy arrays or pandas series, of n_samples 
each, where n_samples is the number of observations. Alternatively, we can pass a contingency table as follows:

```
from scipy.stats.contingency import crosstab
from sklearn.metrics import mutual_info_score
a = ['A', 'B', 'A', 'A', 'B', 'B', 'A', 'A', 'B', 'B']
x = ['X', 'X', 'X', 'Y', 'Z', 'Z', 'Y', 'Y', 'Z', 'Z']
c = crosstab(a, x)
mutual_info_score(labels_true=None, labels_pred=None, contingency = c[1])
```

which will return the same value of mi.


### MI estimation for continuous variables

We can extend the definition of the MI to continuous variables by changing the sum over the values of x and y by the 
integrals:

![]({{ site.baseurl }}/assets/images/posts/mi_eq5.png)

With continuous variables, the problem is how to estimate the probability densities for each one of the variable values.

When the variable was discrete, we created a contingency table, estimated the marginal and joint probabilities, and then 
used those to compute the MI. With continuous variables, this is not possible for 2 reasons: first, the variables can take infinite values, and second, in any dataset, we will only have a few of those probable values. Thus, how can we calculate the MI?

The most obvious approach is to discretize the continuous variables, often into intervals of equal frequency, and then 
proceed as if they were discrete variables. But how do we find the optimal number of intervals? It's been shown that an 
incorrect number of intervals results in poor estimates of the MI.


Alternatively, a nearest-neighbour method was introduced to estimate the MI between 2 continuous variables, or between 
a continuous and a discrete variable. These methods have been shown to provide far better estimates of the MI for 
continuous data.

### Nearest-neighbours method to estimate the MI

We have a series of data points in our data sets that contain values for the continuous variables x and y, with a joint 
probability p(x,y) that we do not know but must estimate from the observed data. The nearest neighbour methods estimate 
the joint probability of these 2 continuous variables, and, as well, the joint probability of a continuous and discrete 
variable.


The following figure (Figure 1A) illustrates the joint distribution of the discrete variable x, which takes 3 values: 
red, green, or blue; and the continuous variable y. In this example, we see that the different values of x are associated 
with different values of y; for example, y is generally lower when x is green or red than when x is blue. Therefore, 
there is a relation between x and y, implying that MI is some positive number.

![walking]({{ site.baseurl }}/assets/images/posts/ch-mi-knn.png)

**Nearest-neighbor approach to estimate the MI. Taken from Ross, 2014, PLoS ONE 9(2): e87357.**

From the joint distribution (Figure 1A), we sample some observations, which represent the available data (Figure 1B). 
The challenge is to estimate the MI between x and y given those few observations.

The nearest-neighbour approach works as follows:

1- We take 1 observation and find the k closest neighbours that show to the same value for x (`N_xi`). So if we take an observation that is red, like the example in figure 1C, we find its 3 closest red neighbours.

2- We calculate the distance between the observation and its furthest neighbour.

3- We count the total number of observations (`m_i`), red and otherwise, within d of the observation in question.


Based on `N_xi`, `m_i`, k (the number of neighbours) and N (the total number of observations), we calculate the MI for that 
particular observation as:


![]({{ site.baseurl }}/assets/images/posts/mi_eq6.png)

where phi is the digamma function. To estimate the MI from the data set, we average I_i over all data points:

![]({{ site.baseurl }}/assets/images/posts/mi_eq7.png)

To evaluate the association between 2 continuous variables the MI is calculated as:

![]({{ site.baseurl }}/assets/images/posts/mi_eq8.png)

where N_x and N_y are the number of neighbours of the same value and different values found within the sphere 
generated by the distance determined in step 3.

The demonstration of how these equations were derived and how this method compares with the binning approach is beyond 
the scope of this article. You can find all the details in the references at the end of this article.


### Python implementation of the mutual information


Let’s calculate the mutual information between discrete, continuous and discrete and continuous variables. We’ll use the 
titanic dataset as an example.

Let’s make some imports:

```
import pandas as pd
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from feature_engine.encoding import RareLabelEncoder, OrdinalEncoder
```

Let’s load the titanic:

```
variables = [
    'pclass', 'survived', 'sex', 'age', 'sibsp',
    'parch', 'fare', 'cabin', 'embarked',
    ]

data = pd.read_csv('https://www.openml.org/data/get_csv/16826755/phpMYEkMl',
                   usecols=variables,
                   na_values='?',
                   dtype={'fare': float, 'age': float},
                   )

data.dropna(subset=['embarked', 'fare'], inplace=True)
data['age'] = data['age'].fillna(data['age'].mean())

def get_first_cabin(row):
    try:
        return row.split()[0]
    except:
        return 'N'

data['cabin'] = data['cabin'].apply(get_first_cabin).str[0]

encoder = RareLabelEncoder(variables='cabin', n_categories=2)
data = encoder.fit_transform(data)

# convert categorical variables to numbers
encoder = OrdinalEncoder(
                encoding_method='arbitrary',
                variables=["sex", "cabin", "embarked"],
                )
data = encoder.fit_transform(data)

data.head()
```

Below we see the first 5 rows of the resulting dataframe:

![View of the Titanic dataset.]({{ site.baseurl }}/assets/images/posts/ch4-fig20.png)

Let’s begin by computing the mutual information between 2 discrete variables. We can use the mutual_info_score as we 
did previously:

```
mutual_info_score(data["sex"], data["pclass"])
```

Or we can use the mutual_info_classif indicating that the random variable is discrete as follows:

```
mutual_info_classif(data["sex"].to_frame(), data["pclass"], discrete_features=[True])
```

To determine the mutual information between a continuous and a discrete variable, we use again the mutual_info_classif, 
but this time, we indicate that the random variable is continuous:

```
mutual_info_classif(data["fare"].to_frame(), data["pclass"], discrete_features=[False])
```

And finally, to estimate the mutual information between 2 continuous variables we use the mutual_info_regression as follows:

```
mutual_info_regression(data["fare"].to_frame(), data["age"], discrete_features=[False])
```

That’s it for the computation of the MI. 

### Feature selection in machine learning using MI

Selecting features with the MI is straightforward. First, we determine the MI between each feature and the target. 
Next, we rank the features based on the MI: higher values of MI mean stronger association between the variables. 
Finally, we select the top ranking features.

#### Python implementation of feature selection

The scikit-learn algorithm for MI treats discrete features differently from continuous features. Consequently, as we did 
previously, we need to flag discrete features. In other words, we need to inform the functions mutual_info_classif or 
mutual_info_regression if the variables are continuous or discrete. 

We will work with the Titanic dataset, which has continuous and discrete variables. Let's begin by making the necessary imports:

```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.feature_selection import mutual_info_classif
from feature_engine.encoding import RareLabelEncoder, OrdinalEncoder
```

Let's load and prepare the Titanic dataset:


```
variables = [
    'pclass', 'survived', 'sex', 'age', 'sibsp',
    'parch', 'fare', 'cabin', 'embarked']

data = pd.read_csv('https://www.openml.org/data/get_csv/16826755/phpMYEkMl',
                   usecols=variables,
                   na_values='?',
                   dtype={'fare': float, 'age': float},
                   )

data.dropna(subset=['embarked', 'fare'], inplace=True)
data['age'] = data['age'].fillna(data['age'].mean())

def get_first_cabin(row):
    try:
        return row.split()[0]
    except:
        return 'N'

data['cabin'] = data['cabin'].apply(get_first_cabin).str[0]

encoder = RareLabelEncoder(variables='cabin', n_categories=2)
data = encoder.fit_transform(data)

# convert categorical variables to numbers
encoder = OrdinalEncoder(
                encoding_method='arbitrary',
                variables=["sex", "cabin", "embarked"],
                )
data = encoder.fit_transform(data)
```

Let's separate the data into train and test sets:

```
X_train, X_test, y_train, y_test = train_test_split( 
    data.drop('survived', axis=1),
    data['survived'],
    test_size=0.3,
    random_state=0,
    )
```

Let's print out the variable names:

```
X_train.columns
```

We obtain the following array:

```
Index(['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'cabin', 'embarked'], dtype='object')
```

Let's create a mask flagging discrete variables:

```
discrete_vars = [True, True, False, True, True, False, True, True]
```

Now, let's calculate the mutual information of these discrete or continuous variables against the target, which is discrete:

```
mi = mutual_info_classif(X_train, y_train, discrete_features=discrete_vars)
```

If we execute mi we obtain the MI of the features and the target:

```
array([0.05762333, 0.13060906, 0.02407659, 0.01505698, 0.01739128, 0.15711667, 0.04897479, 0.01454174])
```

Now, let's capture the array in a pandas series, add the variable names in the index, sort the features based on the MI 
and make a bar plot:

```
mi = pd.Series(mi)
mi.index = X_train.columns
mi.sort_values(ascending=False).plot.bar(figsize=(10, 5))
plt.ylabel('Mutual Information')
plt.title("Mutual information between predictors and target")
```

We obtain the following plot with the MI of each feature and the target:

![MI between the features and the target]({{ site.baseurl }}/assets/images/posts/ch4-fig21.png)

In this case, all features show MI greater than 0, so we could select them all.

If we wanted to select features, we can use for example SelectKBest as follows:

```
sel = SelectKBest(mutual_info_classif, k=5).fit(X_train, y_train)
X_train = sel.transform(X_train)
X_test = sel.transform(X_test)
```

If you made it this far, thank you for reading. 

*Don't forget to check out our course [Feature Selection for Machine Learning](https://www.trainindata.com/p/feature-selection-for-machine-learning) and our 
book [Feature Selection in Machine Learning with Python](https://leanpub.com/feature-selection-in-machine-learning/).*

### Refereces

- Cover, Thomas, Elements of information theory, John Wiley & Sons, Ltd. Chapter 2, 2005 
- Ross, Mutual Information between Discrete and Continuous Data Sets, PLoS ONE 9(2): e87357, 2014.
- Kraskov, Stoegbauer, Grassberger, Estimating mutual information. Physical Review E 69: 066138, 2004.
