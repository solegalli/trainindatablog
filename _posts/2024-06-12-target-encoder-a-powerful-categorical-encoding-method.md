---
layout: post
title: "Target Encoder: A powerful categorical encoding method"
author: cmcouto
description: "Target encoder is Python implementation of the target encoding method for highly cardinal categorical variables."
excerpt: "Target encoder is Python implementation of the target encoding method for highly cardinal categorical variables."
categories: [Categorical Encoding, Data Preprocessing, Feature Engineering]
image: assets/images/posts/target-encoder-a-powerful-categorical-encoding-method/target-encoder.png
---

Most machine learning models don’t natively handle categorical inputs. That’s why we have multiple ways to encode categorical variables into numeric inputs so the models can process them. Depending on the model and the nature of the data, distinct encoding approaches can result in better model performance and/or reduce computational cost.

Among the various encoding techniques, the most commonly used are:

- [**OneHotEncoder**](https://www.blog.trainindata.com/one-hot-encoding-categorical-variables/): This method creates binary (dummy) variables for each category, which can lead to sparse matrices and increased computational complexity, especially with high-cardinality features (i.e., high number of categories).
- **OrdinalEncoder**: Also known as integer encoder, this method maps categories to integers. While this can reduce the computational cost compared to one-hot encoding, it introduces an ordinal relationship that may not exist, potentially misleading linear-based models.
- **TargetEncoder**: This method transforms categorical variables into numerical values based on the target variable. It offers an interesting solution for high-cardinality categorical data by leveraging the target variable‘s statistics to retain predictive power.

If you’ve ever built a machine learning model with categorical features, you’ve probably already tried one of these methods. When the computational complexity of one-hot encoding becomes an issue, and the arbitrary nature of ordinal encoding introduces inaccuracies, TargetEncoding usually becomes the go-to solution.

In this article, we’ll discuss the details of TargetEncoder, from its first publication to its implementation in the main Python libraries.

> Discover categorical encoding techniques beyond one hot encoding with our comprehensive [Feature Engineering for Machine Learning](https://www.trainindata.com/p/feature-engineering-for-machine-learning) course.

## A Brief History of Target Encoding

Target encoding, also known as “mean encoding” or “impact encoding,” is a technique for encoding high-cardinality categorical variables. This method captures the relationship between the categorical features and the target variable, potentially improving the model performance.

Daniele Micci-Barreca introduced this method in 2001 in his paper “A Preprocessing Scheme for High-Cardinality Categorical Attributes in Classification and Prediction Problems.” Back then, Micci-Barreca was using machine learning models to detect fraudulent e-commerce transactions, dealing with sparse categorical variables like ZIP codes, IP addresses, and SKUs. To address these challenges, he proposed transforming categorical variables into numerical values based on the mean of the target variable for each category. His approach addresses the challenge of high-cardinality categorical data by blending category-level target statistics with the overall target distribution, smoothing estimates, and reducing overfitting.

Although Micci-Barreca did not name his method “target encoding,” many subsequent studies and applications have adopted this term to describe similar techniques. Modern implementations of TargetEncoder can be found in open-source Python libraries such as [scikit-learn](https://scikit-learn.org/stable/), [feature-engine](https://feature-engine.trainindata.com/en/latest/), and [category_encoders](https://contrib.scikit-learn.org/category_encoders/), which we’ll analyze further soon. Variations of this method have also been implemented in autoML packages like [H2O](https://h2o.ai/) and state-of-the-art models for tabular data, like [CatBoost](https://catboost.ai/), which is well-known for its success in Kaggle competitions.

A recent paper, “Regularized Target Encoding Outperforms Traditional Methods in Supervised Machine Learning with High Cardinality Features,” extends Micci-Barreca’s work by emphasizing the importance of regularization. This study demonstrates that regularized target encoding can significantly improve the model’s robustness and reduce overfitting. It benchmarks target encoding against traditional methods, consistently showing superior performance across various datasets and machine learning models.

The paper cites several studies comparing target encoding with other techniques. These benchmarks highlight target encoding‘s effectiveness, particularly in handling high-cardinality features and improving predictive performance. The regularization techniques discussed in the paper, such as smoothing and cross-validation, are crucial for preventing overfitting and ensuring stable model performance. We’ll discuss the role of smoothing and cross-validation soon.

In summary, target encoding has evolved significantly since its introduction, with regularization techniques playing a vital role in its modern applications. The combination of blending category-specific statistics with global averages and applying regularization has made target encoding a powerful and reliable method in the [feature engineering](https://www.blog.trainindata.com/feature-engineering-for-machine-learning/) toolkit.

> Still using one hot encoding to train random forests? Make that mistake no further. Learn best encoding practices for different machine learning models with our course [Feature Engineering for Machine Learning](https://www.trainindata.com/p/feature-engineering-for-machine-learning).

## How does Target Encoding work?

In simple terms, target encoding transforms each category into the mean of the target variable for that category. It works similarly for regression and binary classification tasks and can be extended for a multiclass target.

Based on this description, one might think we could simply compute the average of the target class for each category and store those values in order to replace the future categories with new values.

However, simply encoding the categories with the average would probably lead to overfitting, mainly due to low-frequent categories. In fact, if you search over the Internet for target encoding, you’ll likely encounter terms like **overfitting** and **target leakage**.

### A note about target leakage

Although we don’t have data leakage regarding the training and testing sets, we have, at least to some extent, target leakage, because we’re using the target to transform some attributes. Of course, we do it intelligently, so we do not need to rely on the target variable to transform new data because the algorithm learns encoding attributes from the training data. Nonetheless, overfitting may still occur if we do not properly regularize the encoding during the learning phase.

As an extended note, one might think that the model performance may decrease significantly when the target distributions change. Yeah, that’s right! But it also applies to drifting features regardless of using target encoding, right? That’s why we need to monitor a machine learning model in production and re-train it every now and then.

To mitigate overfitting, target encoding blends the global target mean (the prior, in Bayes terms) with the category-specific target mean (the posterior) using a regularization mechanism. This is achieved by using a smoothing parameter that ensures rare categories are not overly represented, thus reducing the risk of overfitting and improving the stability of the encoded values.

Depending on the library, the user should specify the smoothing parameter value. Higher values lead to stronger smoothing (higher weight to the prior). Fortunately, `cikit-learn` and `feature-engine`can automatically detect the optimal smoothing parameter using empirical Bayes variance estimates.

To further reduce overfitting, `scikit-learn`‘s TargetEncoder employs a technique called cross-fitting.

This involves splitting the data into multiple folds, typically using K-Fold cross-validation. Here’s how it works:

1. **Data Splitting:** The dataset is divided into K folds.
2. **Encoding within Folds**: For each fold, the target encoding is computed using the target values from the other K-1 folds, excluding the fold being encoded. This ensures that the target mean for each category is calculated without including data from the current fold.
3. **Blending Means**: The category-specific mean (posterior) is blended with the global target mean (prior) using a smoothing parameter to prevent overfitting.
4. **Combining Results**: The encoded values from each fold are combined to form the final encoded feature.

For more detailed information, refer to the [scikit-learn documentation on target encoding](https://scikit-learn.org/stable/modules/preprocessing.html#target-encoder) and the [scikit-learn example on cross-fitting](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_target_encoder_cross_val.html#target-encoder-s-internal-cross-fitting).

### What about missing values and unseen categories?

We need to handle missing values and unseen categories somehow as part of our machine learning pipelines. I’ll list some possible options and then discuss how `scikit-learn`, `feature-engine` and `category_encoders` handle them.

Options to handle missing values:

- **Impute missing values beforehand:** We can create a pipeline to impute them with any strategy we want (e.g., replacing them by the mode) before applying the target encoding. It would be a prior step.
- **Treat missing values as a new category:** It’s similar to imputing the missing value with a constant like “Missing”.
- **Replace the target mean:** We use the prior value to encode unknown categories.
- **Raise an error:** If we should not expect a missing value, we can raise an error if we face one.
- **Ignore missing values:** We can ignore the missing values. In this scenario, missing values will remain missing after the transformation.

- **Drop missing values:** we could drop them before encoding the data, although it’s not recommended without a thorough analysis. It should not be implemented in any TargetEncoder, as it would be a prior step.

Likewise, for unseen categories, we can choose from:

- **Replace with the target mean:** We use the prior value to encode unknown categories.
- **Raise an error:** If an unknown category is not expected or acceptable, we can raise an error.
- **Ignore:** We can ignore the unknown categories. In this scenario, unknown categories will become missing values.

Imputing or dropping missing values should be an action taken by the user beforehand. Therefore, I’ll not discuss them here. However, we can create a pipeline to impute as the previous step to encoding. In fact, we’ll do that in the upcoming code section.

[Scikit-learn’s TargetEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html) implementation was chosen to consider missing values as another category. The unseen categories are encoded with the target mean. There is no flexibility to choose another option, like raising an error or ignoring missing values and unknown categories.

With [Feature-engine’s MeanEncoder](https://feature-engine.trainindata.com/en/latest/api_doc/encoding/MeanEncoder.html), we can raise an error or ignore missing values, but we cannot encode missing values directly by either considering them another category or using the prior. We need to handle the missing values beforehand (e.g., imputation). The unseen categories can be encoded with the prior, ignored, or set to raise an error message.

The [category_encoders’s TargetEncoder](https://contrib.scikit-learn.org/category_encoders/targetencoder.html) implementation seems to be the most flexible regarding missing values and unseen categories. It allows the user to choose between raising an error, ignoring, and encoding both.

Finally, it’s worth noticing that the formula for target encoding differs according to the target type: binary, multiclass, or continuous (regression). I thought about elaborating on it in this article, but the scikit-learn user guide had already done great work summarizing and explaining the formula succinctly. Therefore, I would rather refer you to [this page](https://scikit-learn.org/stable/modules/preprocessing.html#target-encoder) and ask you to expand their dropdowns for each target type. If you want to go deeper into the formula, please refer to the [original paper](https://dl.acm.org/doi/10.1145/507533.507538).

> Most courses will only show you how to apply one hot encoding and ordinal encoding. But there is a lot more that you can do to get value out of your categorical variables. Find out more in our course [Feature Engineering for Machine Learning](https://www.trainindata.com/p/feature-engineering-for-machine-learning) and stand out of the crowd.

### Notes

One should keep in mind that while target encoding was originally designed for high-cardinality categorical features, both one-hot encoding and ordinal encoding in scikit-learn can handle infrequent categories by grouping them (check out the `min_frequency` and `max_categories` parameters). Similarly, [`RareLabelEncoder`](https://feature-engine.trainindata.com/en/latest/api_doc/encoding/RareLabelEncoder.html) from Feature-engine can group these categories before you apply these traditional encoders.

So, what’s the best option? Well, as you’ve probably heard countless times in your data science journey (and if not, you will), it depends! It depends on your data and the machine learning algorithms you’re using. Give them a try and see what works best for your situation!

[![14 Common Feature Engineering Questions]({{ site.baseurl }}/assets/images/posts/winsorization-handling-outliers-in-machine-learning/Want-a-quick-feature-engineering-reference-Download-our-free-booklet-14-Common-Feature-Engineering-Questions-covering-encoding-scaling-missing-data-outliers-and-more-Get-the-free-bookle.png)](https://www.trainindata.com/p/14-common-feature-engineering-questions)

## Target Encoding in Python

In this section, I’ll demonstrate how to apply – and understand – TargetEncoder for the three types of targets: continuous (regression), binary, and multiclass classification. For each kind of target, we’ll apply the TargetEncoder implementations from `scikit-learn`, `feature-engine` and `category_encoders`.

First, let’s load all the required libraries:

```
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import TargetEncoder
from feature_engine.encoding import MeanEncoder
from category_encoders import TargetEncoder as CE_TargetEncoder

# Ensure to return a dataframe
from sklearn import set_config
set_config(transform_output='pandas')
```

Let’s also create a function to list the learned (fitted) attributes:

```
def show_fit_attributes(transformer):
    """List fit attributes as per scikit-learn convention"""
    return [attr for attr in dir(transformer) if not attr.startswith('_') and attr.endswith('_')]
```

**Note:** In scikit-learn, attributes ending with an underscore (e.g., `feature_names_in_`) indicate that they are learned during the fit method and are not user-specified parameters. This convention helps distinguish between model parameters and learned attributes. Since scikit-learn inspired many other libraries, Feature engine and category encoders also adopted this convention.

In the following examples, I intentionally selected a few categorical columns as features (`X`) for didactic purposes. At the end of this section, a complete data pipeline with TargetEncoder is provided, and it has mixed features (numeric + categorical).

### Continuous Target

Let’s use the [employee_salaries dataset](https://www.openml.org/search?type=data&sort=runs&id=42125) from OpenML to showcase the TargetEncoder application on a continuous target:

```
# Load dataset
X,y = datasets.fetch_openml(name='employee_salaries', return_X_y=True, as_frame=True)

# Filter target categorical columns
target_categorical_features = ['gender', 'department', 'division', 'assignment_category']
X = X[target_categorical_features].astype(str)

# Split data set into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display dataset
X.join(y)
```

![dataframe showing the employees salary dataset]({{ site.baseurl }}/assets/images/posts/target-encoder-a-powerful-categorical-encoding-method/figure1.png)

#### Scikit-learn

With scikit-learn, we can set ‘auto’ for most parameters to allow it to automatically identify the categorical features, the target type, and the smoothing value. Additionally, it provides a cross-fitting approach to learning the target statistic for the different k-folders, which helps prevent overfitting during the training. Note that this cross-fitting approach only works when directly applying the `.fit_transform` method, or when it’s used inside a Pipeline (see [this reference](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_target_encoder_cross_val.html#target-encoder-s-internal-cross-fitting)).

Let’s see the first rows of the training set:

```
X_train.head()
```

![training set - employees salaries dataset]({{ site.baseurl }}/assets/images/posts/target-encoder-a-powerful-categorical-encoding-method/figure2.png)

Let’s now instantiate and fit the encoder while directly transforming the data with `.fit_transform`:

```
# Instantiate encoder
encoder = TargetEncoder(categories='auto', target_type='continuous', smooth='auto', cv=5, random_state=42)

# Fit train & transform the training set
encoder.fit_transform(X_train, y_train).head()
```

![dataset after target encoding]({{ site.baseurl }}/assets/images/posts/target-encoder-a-powerful-categorical-encoding-method/figure3.png)

Have you seen distinct target values for the same “assignment_category“ in the first rows of the training data? That’s due to the cross-fitting when using `.fit_transform`. The model would learn with slightly different values, mitigating potential overfitting.

What if we apply `.transform` instead of `.fit_transform`?

```
# Transform the training data
encoder.transform(X_train).head()
```

![]({{ site.baseurl }}/assets/images/posts/target-encoder-a-powerful-categorical-encoding-method/figure4.png)

We can see the same value for the ” Fulltime-Regular” category from the “assignment_category” feature. Let’s inspect the learned attributes:

```
# Show learned attributes
print(show_fit_attributes(encoder))
```

We see the attributes in the following output:

```
['categories_', 'classes_', 'encodings_', 'feature_names_in_', 'infrequent_categories_', 'n_features_in_', 'target_mean_', 'target_type_']
```

We can inspect the learned attributes, but their names are pretty self-explanatory. The `classes_` will be empty for a continuous target. The `target_mean_` saves the target average for the entire training set, while the `encodings_` saves the average for each category of each categorical feature.

For example, let’s create a dictionary for the “assignment_category“ column:

```
{category: encoding for category,encoding in zip(encoder.categories_[3], encoder.encodings_[3])}
```

We can see the averages that will be used to replace the categories when transforming the data:

```
{'Fulltime-Regular': 77139.98657661081, 'Parttime-Regular': 35094.57315951283}
```

#### Feature-engine

Now, let’s use the feature-engine implementation of TargetEncoder, which is the [MeanEncoder](https://feature-engine.trainindata.com/en/1.8.x/api_doc/encoding/MeanEncoder.html).

Like `scikit-learn`, `feature-engine` provides a way to find the smoothing parameter (weight) automatically. Feature-engine also provides a way to select target variables and decide how to proceed with missing and unseen categories.

Let’s now instantiate and fit this encoder, then transform the test data:

```
# Instantiate encoder
encoder = MeanEncoder(smoothing='auto')

# Fit train data
encoder.fit(X_train, y_train)

# Transform test data
encoder.transform(X_test)
```

![]({{ site.baseurl }}/assets/images/posts/target-encoder-a-powerful-categorical-encoding-method/figure5.png)

What are the fitted attributes?

```
# Show learned attributes
print(show_fit_attributes(encoder))
```

Output:

```
['encoder_dict_', 'feature_names_in_', 'n_features_in_', 'variables_']
```

Here, the main attribute is the `encoder_dict_`which stores as a dictionary the target statistic for each category for each categorical variable.

#### Category_encoders

Finally, let’s explore how to implement TargetEncoder with the Category Encoders library. It also accepts a group of variables to encode (parameter `cols`), and some approaches for handling missing and unseen (aka unknown) categories. With `category_encoders`, the smoothing parameter cannot be automatically detected. On the other hand, it’s the only package among these three where we can specify `hierarchy`, a feature that Micci-Barreca publicly said is missing in current open-source implementations ([see this link](https://towardsdatascience.com/extending-target-encoding-443aa9414cae)).

Let’s instantiate and fit the training set and then transform the test set.

```
# Instantiate encoder
encoder = CE_TargetEncoder(smoothing=10)

# Fit train data
encoder.fit(X_train, y_train)

# Transform test data
encoder.transform(X_test)
```

![]({{ site.baseurl }}/assets/images/posts/target-encoder-a-powerful-categorical-encoding-method/figure6.png)

And now, let’s see the learned attributes:

```
# Show learned attributes
show_fit_attributes(encoder)
```

Output:

```
['feature_names_in_', 'feature_names_out_', 'n_features_in_']
```

Oops! It seems `category_encoders` doesn’t fully follow the  `scikit-learn` convention since the encoding and further learned attributes don’t end with an underscore. Nonetheless, we can see the encodings by accessing the attribute `mapping`, although it’s not so straightforward to analyze it because `category_encoders` applies OrdinalEncoder as part of the TargetEncoder process.

### Binary Classification

Let’s use the [churn dataset](https://www.openml.org/search?type=data&sort=runs&id=40701) from OpenML to showcase the TargetEncoder applications on binary targets. The code below follows the same structure as the code above.

```
# Load dataset
X,y = datasets.fetch_openml(name='churn', version=1, return_X_y=True, as_frame=True)

# Filter target categorical columns
target_categorical_features = ['state', 'area_code', 'phone_number', 'international_plan', 'voice_mail_plan']
X = X[target_categorical_features].astype(str)
y = y.astype(int)

# Split data set into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display dataset
X.join(y)
```

![]({{ site.baseurl }}/assets/images/posts/target-encoder-a-powerful-categorical-encoding-method/figure7.png)

#### Scikit-learn

```
# Instantiate encoder
encoder = TargetEncoder(categories='auto', target_type='binary', smooth='auto', cv=5, random_state=42)

# Fit train data
encoder.fit(X_train, y_train)

# Transform test data
encoder.transform(X_test)
```

![]({{ site.baseurl }}/assets/images/posts/target-encoder-a-powerful-categorical-encoding-method/figure8.png)

#### Feature-engine

```
# Instantiate encoder
encoder = MeanEncoder(smoothing='auto')

# Fit train data
encoder.fit(X_train, y_train)

# Transform test data
encoder.transform(X_test)
```

![]({{ site.baseurl }}/assets/images/posts/target-encoder-a-powerful-categorical-encoding-method/figure9.png)

#### Category_encoders

```
# Instantiate encoder
encoder = CE_TargetEncoder(smoothing=10)

# Fit train data
encoder.fit(X_train, y_train)

# Transform test data
encoder.transform(X_test)

![]({{ site.baseurl }}/assets/images/posts/target-encoder-a-powerful-categorical-encoding-method/figure10.png)
```

### Multiclass Target

Currently implemented only on scikit-learn, the TargetEncoder for multiclass works very similar to the binary target, but it binarizes each class using the “one-vs-all” schema via LabelBinarizer.

To demonstrate the sklearn’s LabelEncoder for a multiclass classification task, let’s load the [eucalyptus dataset](https://www.openml.org/search?type=data&sort=runs&id=188) from OpenML:

```
# Load dataset
X,y = datasets.fetch_openml(name='eucalyptus', version=1, return_X_y=True, as_frame=True)

# # Filter target categorical columns
target_categorical_features = ['Abbrev', 'Locality', 'Map_Ref']
X = X[target_categorical_features].astype(str)

# Split data set into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display dataset
X.join(y)
```

![]({{ site.baseurl }}/assets/images/posts/target-encoder-a-powerful-categorical-encoding-method/figure11.png)

#### Scikit-learn

```
# Instantiate encoder
encoder = TargetEncoder(categories='auto', target_type='multiclass', smooth='auto', cv=5, random_state=42)

# Fit train data
encoder.fit(X_train, y_train)

# Transform test data
encoder.transform(X_test)
```

![]({{ site.baseurl }}/assets/images/posts/target-encoder-a-powerful-categorical-encoding-method/figure12.png)

As we can see, a column per target class was created. A new feature with the approach “one-vs-all” is applied for each target category. Therefore, we’ll have a relationship of MxN transformed features, where M is the number of target categories, and N is the number of categorical features to be encoded.

### Showcasing a pipeline with mixed data using TargetEncoder

In this section, let’s explore using TargetEncoder efficiently within a pipeline. We’ll compare the distinct TargetEncoders alongside OneHotEncoder and OrdinalEncoder.

For this example, we’ll need to import the following libraries:

```
import time
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, TargetEncoder
from feature_engine.encoding import MeanEncoder
from category_encoders import TargetEncoder as CE_TargetEncoder
```

We’ll use this custom function to compute six traditional classification metrics at once:

```
def clf_metrics(y_true, y_pred, y_proba=None):
    """
    Calculate various performance metrics for a classification model.

    Args:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.
    y_proba (array-like, optional): Predicted probabilities for the positive class.

    Returns:
    dict: A dictionary containing calculated metrics such as Accuracy, Balanced Accuracy, Recall, Precision, F1, and optionally ROC_AUC.
    """
    dict_metrics = {
        'Accuracy': metrics.accuracy_score(y_true, y_pred),
        'Balanced Accuracy': metrics.balanced_accuracy_score(y_true, y_pred),
        'Recall': metrics.recall_score(y_true, y_pred),
        'Precison': metrics.precision_score(y_true, y_pred),
        'F1': metrics.f1_score(y_true, y_pred),
    }

    if y_proba is not None:
        dict_metrics['ROC_AUC'] = metrics.roc_auc_score(y_true, y_proba)

    return dict_metrics
```

And we’ll use the [adult census dataset](https://www.openml.org/search?type=data&sort=runs&id=45068) (data dictionary available [here](https://openml.org/search?type=data&status=active&id=42125&sort=runs)):

```
# Load the adult census data (v4)
X,y = datasets.fetch_openml(name='adult', version=4, return_X_y=True, as_frame=True)

# List numeric & categorical features (removing education-num because it's redundant)
NUMERIC_FEATURES = ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']
CATEGORICAL_FEATURES = X.select_dtypes(exclude='number').columns.tolist()

# Change dtype from category to str because we need to fix missing values
X[CATEGORICAL_FEATURES] = X[CATEGORICAL_FEATURES].astype(str)

# Fix missing values
X[CATEGORICAL_FEATURES] = X[CATEGORICAL_FEATURES].replace('nan', np.nan)

# Encode target variable using a rule-based approach (resulting into a binary variable)
y = y.eq('>50K').astype(int)

# Split data into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Show training data
X_train.join(y_train).head()
```

![]({{ site.baseurl }}/assets/images/posts/target-encoder-a-powerful-categorical-encoding-method/figure13.png)

With our imports, custom functions, and data, we can start exploring and modeling our data.

Let’s quickly check for missing values:

```
# Computing missing values
(
    X.isna().sum().to_frame('missing_count')
    .assign(missing_pct = lambda x: x.missing_count / X.shape[0])
    .sort_values('missing_pct', ascending=False)
    .query('missing_count>0')
    .style.format('{:.2%}', subset=['missing_pct'])
)
```

![]({{ site.baseurl }}/assets/images/posts/target-encoder-a-powerful-categorical-encoding-method/figure14.png)

As we can see, there are three categorical features with missing values. They’re going to impact encoders like `OrdinalEncoder` and `MeanEncoder` from `feature-engine`. We’ll need to impute the missing values before applying these specific encoders.

Note: I intentionally skipped the exploratory data analysis (EDA) step since it is not the goal of this article.

Now, we’ll implement a simple machine learning pipeline with scikit-learn:

```
# Set the numeric processor
numeric_processor = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Set the categorical processor for categorical values
categorical_processor = TargetEncoder(categories='auto', target_type='binary', smooth='auto', cv=5, random_state=42)

# Set an overall preprocessor
preprocessor = ColumnTransformer([
    ('numeric', numeric_processor, NUMERIC_FEATURES),
    ('categorical', categorical_processor, CATEGORICAL_FEATURES)
])

# Create the final model pipeline
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(class_weight='balanced', max_iter=500, random_state=42))
])

# Fit the pipeline
model_pipeline.fit(X_train, y_train)
```

![]({{ site.baseurl }}/assets/images/posts/target-encoder-a-powerful-categorical-encoding-method/figure15.png)

Once the model pipeline was trained (`.fit`), we can run our custom function to compute the metrics:

```
# Compute metrics
y_pred = model_pipeline.predict(X_test)
y_proba = model_pipeline.predict_proba(X_test)[:, 1]
clf_metrics(y_test, y_pred, y_proba)
```

which provides the following output:

```
{'Accuracy': 0.7975227761285699,
 'Balanced Accuracy': 0.8114646914388979,
 'Recall': 0.838655462184874,
 'Precison': 0.5559888579387187,
 'F1': 0.668676716917923,
 'ROC_AUC': 0.8961481181997769}
```

Not bad, huh!? Let’s now compare the performance of distinct categorical encoders:

```
# List all encoders we're going to evaluate
categorical_processors = dict(
    onehot=OneHotEncoder(handle_unknown='ignore', sparse_output=False),
    ordinal=make_pipeline(
        SimpleImputer(strategy='constant', fill_value='Missing'),
        OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
    ),
    target_sklearn=TargetEncoder(categories='auto', target_type='binary', smooth='auto', cv=5, random_state=42),
    target_fe=make_pipeline(
        SimpleImputer(strategy='constant', fill_value='Missing'),
        MeanEncoder(smoothing='auto'),
    ),
    target_ce=CE_TargetEncoder(smoothing=10)
)

# Evaluating the performance of each encoder
results = []

for encoder_name, encoder in categorical_processors.items():
    # Update categorical preprocessor
    model_pipeline.set_params(preprocessor__categorical=encoder)

    # Measure the fitting time
    start_time = time.time()
    model_pipeline.fit(X_train, y_train)
    fit_time = time.time() - start_time

    # Compute metrics
    y_pred = model_pipeline.predict(X_test)
    y_proba = model_pipeline.predict_proba(X_test)[:, 1]
    model_metrics = clf_metrics(y_test, y_pred, y_proba)
    model_metrics['encoder'] = encoder_name     # identify the encoder
    model_metrics['fit_time'] = fit_time        # add fit time to metrics

    # Append metrics to list
    results.append(model_metrics)

# Convert list of results into a dataframe
df_metrics = pd.DataFrame(results).set_index('encoder').round(4)

# Display the results
display(df_metrics)
```

![]({{ site.baseurl }}/assets/images/posts/target-encoder-a-powerful-categorical-encoding-method/figure16.png)

As we can see, the three target encoding implementations worked very similarly for this dataset. They worked much better than the ordinal encoder (since we’re using a linear model that assumes linearity) and were slightly worse than the one-hot encoding. We have also noticed that the training time is shorter when using TargetEncoder (mainly with scikit-learn and feature-engine), with one-hot encoding taking the longest training time.

So far, we’ve tested a unique dataset with a unique model, differing only in the encoder option. Check [this example](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_target_encoder.html#sphx-glr-auto-examples-preprocessing-plot-target-encoder-py) from scikit-learn to analyze the comparison among encoders on a distinct dataset for a regression task. If you’re interested in how the encoding technique might impact the performance depending on the model, check [this article](https://medium.com/@cmcouto.silva/the-categorical-encoder-you-choose-might-impact-the-model-performance-3dcc2727abb9), where I compare distinct encoders with distinct models across two datasets using cross-validation.

Finally, check out [this paper](https://link.springer.com/article/10.1007/s00180-022-01207-6) to dive deep into an analysis showing how multiple encoders perform among multiple datasets (n=24) with multiple models (n=5). Spoiler alert: they found that regularized versions of target encoding consistently provided the best results!

That’s all for this section! I hope you’ve enjoyed and learned something useful 🤓

## Final considerations

TargetEncoder is a special technique that efficiently encodes high-cardinal categorical features. Simple implementations should be carefully used to avoid overfitting and target leakage. Fortunately, the latest implementations from scikit-learn and feature-engine provide a reliable way to use it given the automatic search of the smoothing parameter, equipping us with a strong approach to include in our data science toolkit!

👉 The code used in this article is available on this [GitHub repository](https://github.com/cmcouto-silva/posts/blob/main/notebooks/TargetEncoder_TrainindataBlog.ipynb).

## References

- [Paper]: [A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems](https://dl.acm.org/doi/10.1145/507533.507538)
- [Paper]: [Regularized target encoding outperforms traditional methods in supervised machine learning with high cardinality features](https://link.springer.com/article/10.1007/s00180-022-01207-6#Sec29)
- [Scikit-learn]: [Target Encoder‘s Internal Cross fitting](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_target_encoder_cross_val.html#target-encoder-s-internal-cross-fitting)
- [Scikit-learn]: [Comparing Target Encoder with Other Encoders](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_target_encoder.html#comparing-target-encoder-with-other-encoders)
- [Feature-engine]: [Target encoder user guide](https://feature-engine.trainindata.com/en/1.8.x/user_guide/encoding/MeanEncoder.html)

## Further resources

Check out the [Feature Engineering for Machine Learning](https://www.trainindata.com/p/feature-engineering-for-machine-learning) course if you want to learn more about categorical encoders, including:

- Distinct encoding methods
- How they work
- When to use each of them
- How to use the latest Python implementations
- And much more!

[![Feature Engineering for Machine Learning course]({{ site.baseurl }}/assets/images/posts/winsorization-handling-outliers-in-machine-learning/feature-engineering-machine-learning-course.jpg)](https://www.trainindata.com/p/feature-engineering-for-machine-learning)
