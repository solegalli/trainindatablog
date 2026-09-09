---
layout: post
title: "Variance stabilizing transformations in machine learning"
author: sole
description: "The logarithm, power, and square root are variance stabilizing transformations. How and why are they used in machine learning?"
excerpt: "The logarithm, power, and square root are variance stabilizing transformations. How and why are they used in machine learning?"
categories: [Feature Engineering, Machine Learning]
image: assets/images/posts/variance-stabilizing-transformations-in-machine-learning/cover-1.png
---

You’ve probably heard that before training machine learning models, data scientists transform random variables to change their distribution into something closer to the normal distribution.

But, why do we do this? Which variables should we transform? Which transformations should we use? And, do we need to transform variables to train any machine learning algorithm?

These are the questions that we will address throughout this article. Let’s get started.

## Why do we transform random variables?

Many statistical methods used in data analysis and supervised machine learning models make assumptions about the data. For example, to draw conclusions from a linear regression model, many assumptions must be true. Some of the assumptions are the following:

- The values of the dependent variable (that is, the target) are independent.
- There is a linear relationship between the target and the independent (predictor) variables.
- The residuals, that is, the difference between the predictions and the values of the target, are normally distributed and centered at zero.
- The residuals have constant variance.

Many people, myself included, confuse the last assumption with the idea that all the predictor variables have to be normally distributed. But this is not the case. What needs to be normally distributed and centered at zero are the residuals, which means that any difference between the predictions and the target is just random.

![]({{ site.baseurl }}/assets/images/posts/variance-stabilizing-transformations-in-machine-learning/figure_1.png)

### **What happens if the assumptions are not met?**

When the assumptions are not met, the conclusions drawn from the data analysis or from the linear regression models, might not be reliable.

Fortunately, we can correct the failure in the assumptions by transforming the variables prior to the analysis. This would improve the performance and reliability of the models.

We can transform the target variable itself when its distribution is skewed. Transforming the predictor variables, very often helps meet the model assumptions when the raw data does not. The transformations that data scientists apply to predictor variables and the target are called variable stabilizing transformations, and I will explain what that means in the coming sections.

### When to apply variance stabilizing transformations?

You probably guessed that variable transformations are usually applied when we analyze data through linear statistical tests like ANOVA and when training linear regression models.

In other words, there is no need to transform variables when training non-linear models like decision tree based algorithms, nearest neighbors, or neuronal networks. What are variance stabilizing transformations?

### Variable transformations

Variable transformation consists of replacing the original variable values with a function of that variable. Transforming variables with mathematical functions helps reduce variable skewness, therefore improving the value spread, and sometimes unmasks linear and additive relationships between predictors and target.

Commonly used mathematical transformations include the logarithm, reciprocal, power, and square root transformations, as well as the Box-Cox and Yeo-Johnson transformations. These transformations are commonly referred to as “variance stabilizing transformations”. Variance stabilizing transformations intend to bring the distribution of the variable to a more symmetric, or in other words, Gaussian shape.

In other words, a variance-stabilizing transformation is a function f that turns all possible values of y into other values y’=f(y) in such a way that the variance of y’ remains constant.

For some distributions like Poisson or the binomial distribution, exact variance stabilization is not possible, so we say that the transformations are “approximate variance stabilizing” transformations.

![Density plot showing the normal distribution]({{ site.baseurl }}/assets/images/posts/variance-stabilizing-transformations-in-machine-learning/figure_2.png)

Many variance stabilizing transformations were discussed and analyzed in the context of Poisson distributions, where the variance of the variable is equal to the mean. Hence, the larger the mean, the larger the variance.

The transformation of the variables aims to obtain values, such that their variance is independent of their mean, that is, variables with a constant variance. Thus, variance stabilizing transformations.

[![Density plots with various Poisson distributions]({{ site.baseurl }}/assets/images/posts/variance-stabilizing-transformations-in-machine-learning/figure_3-1024x791.png)](https://commons.wikimedia.org/wiki/File:Poisson_pmf.svg)

In the following paragraphs, we will discuss the following variance stabilizing transformations:

- Logarithm
- Reciprocal
- Square root
- Arcsin
- Power
- Box-Cox
- Yeo-Johnson

## Logarithm transformation

The logarithm function is a powerful transformation for dealing with positive data with a right-skewed distribution (observations accumulate at lower values of the variable).

If y is the variable, then the logarithmic transformation is log(y).

Candidate variables for the log transformation are those like income, or salary, which are continuous variables and tend to show a heavy accumulation of observations towards smaller values. In other words, most people earn little, just a few earn a lot.

In particular, if we take the variable Median Income from the California housing data set from Scikit-learn, we see that it is continuous and right-skewed:

![Histogram showing the distribution of the variable Median Income from the California Housing data set.]({{ site.baseurl }}/assets/images/posts/variance-stabilizing-transformations-in-machine-learning/figure_4.png)

Yet, after the logarithm transformation, we observe more widely spread and evenly distributed values:

![Histogram showing the distribution of Median Income after applying the logarithm transformation]({{ site.baseurl }}/assets/images/posts/variance-stabilizing-transformations-in-machine-learning/figure_5.png)

## Reciprocal transformation

The reciprocal function is defined as 1/y, where y is the random variable. It is a transformation with a drastic effect on the variable distribution.

The reciprocal transformation is useful when we have ratios, that is, values resulting from the division of two variables. Classical examples are variables like population density, that is, people per area, or house occupancy, that is, the number of occupants per house.

When we calculate the inverse of these variables, we pass from a representation of people per area to area per person, or occupants per house to houses per occupant. The transformed data still make sense (to humans), and it tends to show a better spread of values.

If you don’t believe me, take a look at the histogram of house occupancy from the California housing data set from Scikit-learn, a highly skewed variable:

![Histogram showing the distribution of the variable House Occupancy from the California housing dataset]({{ site.baseurl }}/assets/images/posts/variance-stabilizing-transformations-in-machine-learning/figure_6.png)

And have a look at the distribution of the same variable after the reciprocal data transformation:

![Histogram showing the distribution of the variable House Occupancy after applying the reciprocal transformation.]({{ site.baseurl }}/assets/images/posts/variance-stabilizing-transformations-in-machine-learning/figure_7.png)

You can see how the reciprocal transformation dramatically improved the spread of values and even transformed a discrete variable into a continuous one.

One caveat of the reciprocal or inverse transformation is that it is not defined for the value 0. So if our variables contain zeroes… well, we should try something else.

## Square root transformation

We mentioned previously that variance stabilizing transformations are discussed quite often in the context of Poisson distributions.

The square root transformation (√x) is a variance stabilizing transformation that transforms variables with a Poisson distribution (counts) into variables with an approximately standard Gaussian distribution. The Anscombe transformation (√(x+3/8)) and the Freeman-Tukey transformation (√x + √(x+1)) are variations of the square root transformation that also achieve variance stabilization.

The square root transformation is a form of power transformation where the exponent is 1/2 and is only defined for positive values. We will discuss general power transformations in the coming paragraphs.

There are many variables with Poisson distributions. For example the number of credit cards or bank accounts per person, the number of children per family, or the number of pets. Those variables are naturally counts. So, in these cases, a square root transformation could be suitable to stabilize the variance.

Now, if we transform these variables using a Poisson distribution, we won’t see the same clear changes that we would see with continuous variables. However, we will see that the observations are more evenly distributed along the diagonal in Q-Q plots.

Example Poisson distribution before the transformation:

![Image showing a histogram and a q-q plot with a theoretical Poisson distribution.]({{ site.baseurl }}/assets/images/posts/variance-stabilizing-transformations-in-machine-learning/figure_8-1024x433.png)

The same distribution after the square root transformation:

![Image showing a histogram and Q-Q plot of a variable with a Poisson distribution after applying the square root transformation]({{ site.baseurl }}/assets/images/posts/variance-stabilizing-transformations-in-machine-learning/figure_9-1024x450.png)

Note how the observations are more evenly distributed along the red line in the precedent image.

## Arcsin transformation

Before diving into generalized power transformations, let’s have a quick look at the arcsin transformation. The arcsin transformation, also called the arcsin square root transformation, or angular transformation, takes the form of arcsin(sqrt(x)) where x is a real number between 0 and 1.

The arcsin square root transformation helps in dealing with probabilities, percentages, and proportions. It aims to stabilize the variance of the variable and return more evenly distributed (Gaussian-looking) values.

As you can imagine, there are plenty of examples of variables that could be suitable candidates for the arcsin transformation, like those from the breast cancer dataset from Scikit-learn.

You can see how a bunch of these variables show skewed distributions in their raw state:

![Histograms showing the distribution of the variables in sklearn's cancer dataset]({{ site.baseurl }}/assets/images/posts/variance-stabilizing-transformations-in-machine-learning/figure_10-1024x614.png)

And after the arcsin transformation the values are more evenly distributed:

![Histograms showing the distribution of the breast cancer dataset variables after applying the arcsin transformation.]({{ site.baseurl }}/assets/images/posts/variance-stabilizing-transformations-in-machine-learning/figure_11-1024x614.png)

This was probably a transformation that was out of your radar, and truth be told, it is rarely used. But here it is, a function that has been widely studied in the past.

## Power transformations

While variance-stabilizing transformations for some parametric families of distributions, such as the Poisson and binomial distributions, are well-known, some methods of data analysis rely on trial and error, such as searching through power transformations to find a suitable fixed transformation.

Power functions are mathematical formulations like this: X = X^lambda where lambda can take any value.

The square and cube root transformations are special cases of power transformations where lambda is 1/2 or 1/3, respectively. The reciprocal transformation is also a power transformation where lambda is -1. So hey, all along we’ve been discussing power transformations!

The challenge in choosing a power transformation resides in finding a suitable value for the parameter lambda that returns variables whose values are more evenly distributed.

We discussed the special cases of the square root and reciprocal transformation in the previous paragraphs because they are suitable for specific variables, or better say, specific distributions. In reality, we don’t really manually try exponents to see which one works best, because the Box-Cox transformation, which we will discuss in the next paragraph, automatically finds the parameter lambda for us.

![Illustration of power transformations]({{ site.baseurl }}/assets/images/posts/variance-stabilizing-transformations-in-machine-learning/figure_12.png)

As general guidance, if data is right-skewed (i.e. more observations around lower values), use lambda <1. If data is left-skewed (i.e. more observations around higher values), use lambda >1.

## Box-Cox transformation

The Box-Cox transformation is a generalization of the power family of transformations, and it is defined by:

![The Box-Cox transformation equation]({{ site.baseurl }}/assets/images/posts/variance-stabilizing-transformations-in-machine-learning/figure_13.png)

where X is the variable and λ is the transformation parameter.

The Box-Cox transformation can be used for transformations that we discussed before, including no transformation (λ = 1), the logarithm (λ = 0), the reciprocal (λ = -1), the square root (when λ = 0.5), and the cube root.

In the Box-Cox transformation, several values of the parameter λ are evaluated using maximum likelihood, and the λ that returns the best transformation is selected. The Box-Cox transformation is usually the preferred choice for machine learning practitioners, because it is not necessary to think about which transformation to apply to which variable.

The only caveat with the Box-Cox transformation is that it was designed only for positive variables. So, if your variables contain negative values, you can either shift the distribution by adding a constant, or use the Yeo-Johnson transformation.

## Yeo-Johnson transformation

The Yeo-Johnson transformation is an extension of the Box-Cox transformation that is no longer constrained to positive values. In other words, the Yeo-Johnson transformation can be used on variables with zero and negative values as well as positive values.

The Yeo-Johnson transformation is defined as follows:

![The Yeo-Johnson transformation equation ]({{ site.baseurl }}/assets/images/posts/variance-stabilizing-transformations-in-machine-learning/figure_14-1024x308.png)

In short, if the variable X is strictly positive, then, the Yeo-Johnson transformation is the same as the Box-Cox power transformation of X + 1. If X is strictly negative, then the Yeo-Johnson transformation is the Box-Cox transformation of (-X + 1) but with power 2 — λ.

If the variable has positive and negative values, then the transformation is a mixture of these 2 functions, so different powers are used for the positive and negative values of the variable. If you ask me, it’s a bit of a mess, but as long as it works…

## Why, which, how and when to transform variables?

We started the blog post with the following questions:

- Why do we transform random variables?
- Which variables should we transform?
- Which transformations should we use for variance stabilization?
- And, do we need to transform variables to train any machine learning algorithm?

By now, I think we have answers to all of these questions.

### Why do we transform variables?

In order to make data meet the assumptions of certain statistical models, typically analysis of variance (ANOVA) and linear regression models, and thus be able to draw accurate or reliable conclusions from the data analysis.

### Which variables should we transform?

In general, those random variables which show distributions that are not normal.

### Which transformations should we use?

There are a bunch of transformations that we can use. If we are keen to understand our transformed variables, we might prefer to do some data analysis and select which transformation to apply to which variable based on what we discussed throughout the blog post. For example, we would apply square root to counts, arcsin to fractions, and reciprocal to ratios. We would apply log to variables with observations accumulating on lower values, and for everything else, other power transformations.

In practice, to speed things up, we just go for Box-Cox or Yeo-Johnson, which consider all of the above transformations, and choose the transformation automatically. But beware, automation does not always resolve the issue!

Sometimes, applying transformations blindly creates an issue. So it is always good practice to plot the variables after the transformation, and be sure that we have obtained the expected result. I know, I sound like grandma.

Take a look for example at the following figure taken from the scikit-learn documentation:

[![Histograms showing various theoretical distributions and their shape after the Box-Cox and Yeo-Johnson transformations]({{ site.baseurl }}/assets/images/posts/variance-stabilizing-transformations-in-machine-learning/figure_15.png)](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_map_data_to_normal.html)

Applying transformations to variables that are already normally distributed does not really change the distribution (see the lilac plots), so there is no need to do that.

On the other hand, the Box-Cox and Yeo-Johnson transformations may not return Gaussian shaped distributions after the transformation, as in the extreme examples of the bimodal (green) and uniform (black) distributions.

Where am I going with this? Although it may be tempting to automatically transform variables and train models, it may be worth taking the time to analyze the transformations and understand the data in our datasets and what we are feeding to our models.

Finally, do we need to transform variables to train any machine learning algorithm?

No. These transformations were studied and designed for their use with linear models. So, if you want to train non-linear models like decision tree-based algorithms or nearest neighbors, you might as well skip this step.

## Python implementation of variance stabilizing transformations

I wrote a lot about variance stabilizing transformations, but I haven’t really shown you how to implement these transformations in Python, have I?

Applying these transformations with Python is really easy. We can do so with Numpy as follows:

```
import numpy as np
data["variable_log"] = np.log(data["variable_original"])
```

For the reciprocal, we would use `np.reciprocal()`, for the square root `np.sqrt()`, and for the Power `np.exp(data[“variable_original”], lambda)`, where lambda is the desired exponent of the transformation.

For BoxCox and Yeo-Johnson, we would use scipy.stats:

```
import scipy.stats as stats
X_tf["new_var"], param = stats.boxcox(X["original_var"])
X_tf["new_var"], param = stats.yeojohnson(X["original_var"])
```

where param is the suitable lambda found by the transformation.

Yet, with Numpy and scipy.stats we need to modify one variable at a time. We can transform various variables simultaneously by utilizing Scikit-learn, or the library for which I am the maintainer: Feature-engine.

So for example, if we want to apply the Box-Cox transformation with Feature-engine, we would do the following:

```
from feature_engine.transformation import BoxCoxTransformer
boxcox = BoxCoxTransformer()boxcox.fit(X_train)
train_transformed = boxcox.transform(X_train)
test_tranformed = boxcox.transform(X_test)
```

With Scikit-learn we would do:

```
from sklearn.preprocessing import PowerTransformer
transformer = PowerTransformer(method="box-cox", standardize=False)
boxcox.fit(X_train)
train_transformed = boxcox.transform(X_train)
test_tranformed = boxcox.transform(X_test)
```

There are differences between the Scikit-learn and Feature-engine implementations which I discuss in [this article](https://towardsdatascience.com/practical-code-implementations-of-feature-engineering-for-machine-learning-with-python-f13b953d4bcd).

I also highlight the differences between Numpy, scipy.stats, Scikit-learn and Feature-engine in [our online course](https://www.courses.trainindata.com/p/feature-engineering-for-machine-learning) and [book](https://packt.link/0ewSo).

[![Feature Engineering for Machine Learning course]({{ site.baseurl }}/assets/images/posts/feature-scaling-in-machine-learning/feature-engineering-machine-learning-course-1024x576.jpg)](https://www.trainindata.com/p/feature-engineering-for-machine-learning)

## References

For more information on variable transformation and variance stabilizing transformation, check the following resources:

- [Feature Engineering for Machine Learning](https://www.courses.trainindata.com/p/feature-engineering-for-machine-learning) – Online course
- [Python Feature Engineering Cookbook](https://www.packtpub.com/en-us/product/python-feature-engineering-cookbook-9781835883587) (also available on [Amazon](https://www.amazon.com/Python-Feature-Engineering-Cookbook-complete/dp/B0DBQDG7SG))
- [Scikit-learn documentation](https://scikit-learn.org/stable/modules/preprocessing.html#mapping-to-a-gaussian-distribution)
- [Feature-engine documentation](https://feature-engine.readthedocs.io/en/latest/user_guide/transformation/index.html)
- [Transformations: an introduction](https://www.stata.com/users/njc/topichlp/transint.hlp)
