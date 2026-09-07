---
layout: post
title: "Machine Learning Fundamentals"
author: priyansh
description: "Machine learning fundamentals help to tackle real-world problems, enabling accurate model selection, evaluation, troubleshooting."
excerpt: "Machine learning fundamentals help to tackle real-world problems, enabling accurate model selection, evaluation, troubleshooting."
categories: [Machine Learning]
image: assets/images/posts/machine-learning-fundamentals/machine-learning-fundamentals.png
---

At the heart of the digital revolution lies machine learning—a powerful tool shaping the future of innovation. At its core, machine learning allows computers to learn from data and make decisions without explicit programming.

Understanding the fundamentals of machine learning is the first step towards starting your journey into the fields of machine learning and data science. It is crucial for effectively solving real-world problems using appropriate techniques and models, enabling accurate evaluation, troubleshooting, and fostering innovation in the field.

So let’s dive into the fundamentals of machine learning.

> Advance your data science and machine learning skills with our [comprehensive expert led courses](https://www.trainindata.com/courses).

[![Advance your data science and machine learning skills with our comprehensive and expert led courses.]({{ site.baseurl }}/assets/images/posts/complete-guide-to-platt-scaling/Rectangle-7.png)](https://www.trainindata.com/courses)

## **What is Machine Learning?**

In today’s digital age, machines can perform tasks that were once thought to be solely in the realm of human expertise. How can machines carry out these tasks? Thanks to machine learning.

Machine learning is a field of computer science that consists of developing procedures that enable computers to learn from data without being explicitly programmed. These procedures are called algorithms.

In simple terms, machine learning allows computers to learn from data and make decisions based on what they’ve learned. It’s like teaching a computer to recognize faces in photos, understand spoken language, translate texts, or even play games like chess or Go—all without being explicitly programmed to do so.

### Machine Learning, Deep Learning and Artificial intelligence

You probably hear these terms a lot. How are they related?

Machine learning is a subset of artificial intelligence that focuses on developing algorithms that enable computers to learn from data.

Deep learning is a specific type of machine learning that uses neural networks to learn complex patterns in data. Artificial neural networks are modeled after the workings of the human brain.. It is said that a neural network can approximate, i.e., learn, any mathematical function, and therefore, their learning potential is enormous.

![machine learning, deep learning and artificial intelligence.]({{ site.baseurl }}/assets/images/posts/machine-learning-fundamentals/machine-learning-diagram.png)

Artificial intelligence (AI) is a broader concept involving any technique or system that tries to mimic human intelligence. That includes machine learning and deep learning as specific approaches within the field.

Data science is an interdisciplinary field that employs scientific methods and machine learning algorithms to extract insights and knowledge from structured and unstructured data.

## **What Can Machine Learning Do?**

Machine learning allows computers to recognize patterns in data, understand language, identify objects in images or videos, make recommendations, and predict future outcomes based on past data.

In fact, machine learning is revolutionizing numerous industries with its ability to analyze vast amounts of data and extract valuable insights. These are some of the key applications:

> Advance your data science and machine learning skills with our [comprehensive expert led courses](https://www.trainindata.com/courses).

[![Advance your data science and machine learning skills with our comprehensive and expert led courses.]({{ site.baseurl }}/assets/images/posts/complete-guide-to-platt-scaling/Rectangle-7.png)](https://www.trainindata.com/courses)

### Natural Language Processing (NLP)

NLP enables computers to understand, interpret, and generate human language. Thanks to NLP, computers can detect sentiment, translate and make text summaries.

Generative AI consists of more modern algorithms that allow computers to return human-like text. These algorithms are called “Generative Pretrained Transformers” or GPT. They are the state-of-the-art models for NLP.

### Computer Vision

Computer vision teaches computers to interpret and analyze information from images and videos. It enables machines to “see” and “understand” the world.

Computer vision is used in facial recognition for security systems and authentication, and in self-driving cars for detecting pedestrians, traffic signs, and other objects on the road. Additionally, it’s used in healthcare for diagnosing diseases from X-ray images and MRI scans.

### Predictive Analytics

Predictive analytics empowers computers to learn patterns from past data, and use them to forecast future trends, behaviors, or outcomes.

Data scientists use predictive analytics across industries, for example, to detect fraud, assess credit risk, understand and anticipate customer churn, forecast energy demand, and optimize the supply chain, among many other applications.

### Recommendation Systems

Recommendation systems are algorithms that analyze user preferences. They analyze past behavior, for example, past purchases, viewed films, or listened to and liked songs, to suggest personalized content, products, or services that the customer might be interested in.

Recommender systems are used in streaming services like Spotify or Netflix and in e-commerce like Amazon.

### Speech Recognition

Speech recognition involves converting spoken language into text. Once it is in the form of text, we can use NLP to allow computers to understand it.

Speech recognition is used in virtual assistants and customer services to understand and respond to users and customers.

### Other Applications of Machine Learning

Other areas, like robotics—powered by reinforcement learning—are common examples where machine learning plays a pivotal role.

As you can see, machine learning offers boundless applications in the real world. These are possible thanks to different types of machine learning methodologies. What are these methodologies?

> Advance your data science and machine learning skills with our [comprehensive expert led courses](https://www.trainindata.com/courses).

[![Advance your data science and machine learning skills with our comprehensive and expert led courses.]({{ site.baseurl }}/assets/images/posts/complete-guide-to-platt-scaling/Rectangle-7.png)](https://www.trainindata.com/courses)

## **Types of Machine Learning Algorithms**

Machine learning methodologies can be broadly categorized into two main types: supervised learning and unsupervised learning.

### **Supervised Learning**

Supervised learning involves training a model on labeled data, where each input is associated with an output. The goal of supervised learning is to learn a mapping function from input variables to output variables. This allows the algorithm to make predictions or decisions when given new, unseen data.

As we see in the diagram, initially we have a training set containing many observations, and each observation is labeled. Some are triangles, some are circles, and some are squares. We use that data to train a machine-learning algorithm. The model learns to match observations to shapes based on their characteristics. And later on, we can give new observations to the model, and it will be able to tell us which shape they have.

**![Supervised learning]({{ site.baseurl }}/assets/images/posts/machine-learning-fundamentals/supervised-learning.png)**

We can use supervised learning for regression and for classification.

#### **Regression**

Regression models predict continuous values. For example, predicting house prices based on features like square footage, number of bedrooms, and location is an example of a regression.

Popular algorithms for regression are linear regression, polynomial regression, decision tree regression, random forest regression, and support vector regression.

#### **Classification**

Classification models predict discrete outcomes, or categories. For instance, classifying emails as spam or non-spam based on their content is an example of classification.

Popular algorithms for classification are Logistic Regression, Naive Bayes, Support Vector Machines, Decision Trees, Random Forest Classifiers, and K-Nearest Neighbors (KNN).

### **Unsupervised Learning**

Unsupervised learning is a type of machine learning where the algorithm learns patterns and structures from unlabeled data. Unlike supervised learning, there are no predefined labels for unsupervised learning tasks. Instead, the algorithm seeks to discover hidden patterns or groupings within the data.

In the following diagram, we pass a dataset without labels to a machine learning model, which, by analyzing the intrinsic data patterns, learns to group observations based on their similarities:

**![Unsupervised learning - diagram]({{ site.baseurl }}/assets/images/posts/machine-learning-fundamentals/unsupervised-learning-1024x1024.png)**

Unsupervised learning has many applications. It can be used in clustering to find groups of similar observations. It can be used to simplify the data representation through dimensionality reduction. It can also be used to find anomalies.

#### **Clustering**

Clustering algorithms group similar data points together into clusters. The goal is to identify natural groupings or clusters in the data without any prior knowledge of their labels. The grouping is done by identifying similar patterns among variables.

Clustering can be used, for example, in customer segmentation to group together customers with similar purchasing behaviors. Some machine learning techniques used for clustering are K-Means Clustering, Hierarchical Clustering, and DBSCAN.

#### **Dimensionality Reduction**

Dimensionality reduction techniques aim to reduce the number of features in a dataset while preserving its essential information.

Principal Component Analysis is a popular dimensionality reduction technique that projects high-dimensional data into a lower dimension while preserving as much information as possible. This can help visualize and analyze complex datasets more effectively.

#### **Anomaly Detection**

Anomaly detection with unsupervised learning involves identifying unusual patterns or outliers in data without labeled examples. By analyzing the inherent structure and distribution of the data, unsupervised learning algorithms detect deviations or irregularities that stand out from the typical patterns, thus flagging potential anomalies.

Anomaly detection can be done by clustering and finding observations that do not fit in any cluster, by determining distributions and flagging outliers, or by using specific machine learning techniques, like one-class support vector machines or isolation forests.

## **Fundamentals of Machine Learning**

As you can see, machine learning has many applications in the real world, thanks to different types of machine learning methodologies. However, the fundamentals of machine learning are the same across applications and algorithms.

These machine learning basics include key components such as data, algorithms, training, testing, and evaluation techniques, which are essential for building effective models that generalize well to new, unseen data.

Let’s flesh these components out one by one.

### **Data in Machine Learning**

Machines learn patterns, make predictions, and generate insights from data. Data is essential for model performance, decision-making, and optimization. In fact, the field of data science is devoted to analyzing, processing, and preparing data, either for machine learning or to extract insight to drive decisions.

Data comes in many forms. We can have tables with numbers, images, or text. Images and texts are self-explanatory. Tabular data, however, has different flavors. Let’s discover some of them.

#### **Features and Labels**

Tabular data comes in the form of tables, where each row is an observation and each column is a feature or attribute.

Features, also called variables, are individual measurable properties or characteristics of the data being analyzed. For example, height is a feature, weight is another feature, as is color, vehicle make, city of residence, and so on.

Features serve as input variables for machine learning algorithms and can be numeric, categorical, or binary in nature. They provide the information necessary for the algorithm to learn patterns and make predictions or decisions.

![Features and labels in tabular data.]({{ site.baseurl }}/assets/images/posts/machine-learning-fundamentals/features-label.png)

Labels, also known as targets or responses, are the outcomes or values we want to predict. In a dataset of house prices, features and variables may include square footage, number of bedrooms, and location, while the label would be the actual sale price of the house.

#### **Numerical and Categorical Data**

Numerical data consists of numerical values that represent quantities or measurements. Examples of numerical variables are the number of rooms in a house, the median income, and the blood pressure, among others.

![Numerical and categorical variables in tabular data.]({{ site.baseurl }}/assets/images/posts/machine-learning-fundamentals/numerical-features.png)

Categorical data consists of categories or labels that represent qualitative attributes or characteristics. Examples of categorical features are gender, marital status, vehicle make, city of residence, and so on.

#### **Data Preprocessing**

The data that is collected either by automated sensors, machines, or systems is not suitable in its raw format to train machine learning models. Instead, data scientists devote a lot of time to preparing data to train machine learning models.

Data preprocessing is done to convert the raw data into a processable form that can be fed to a machine-learning model for training and making predictions. In fact, data preprocessing is the initial step in data analysis and machine learning projects.

Data preprocessing includes among other things, the following:

- Cleaning data, handling missing values and outliers, and removing duplicates.
- Scaling or normalizing data for uniformity.
- Encoding categorical data to numerical format which the machine can understand.
- Transforming variables to meet model assumptions.
- Extract features from complex structures, like texts, transactions or time series.
- Create new features that capture business knowledge.

#### **Exploratory Data Analysis**

Data preprocessing goes hand in hand with exploratory data analysis (EDA). Through EDA, data scientists seek to understand data patterns, correlations, and trends to gain insights into the structure, characteristics, and relationships between features.

Visualizations, graphs, and plots are actively used during EDA. This step is crucial for data-driven decision-making and hypothesis-testing. EDA also aids in creating predictive features and optimizing model performance.

### **Training and Validation Data Sets**

After data preprocessing and EDA, we are ready to start training machine learning models. To train machine learning models, we typically split the original data set into two or more sets: a training set, a testing set, and a validation set.

**Training data**

The training dataset is used to train the machine learning model by adjusting its parameters based on the input features and corresponding target labels.

**Validation data**

This set is used to evaluate and adjust a model during training. It acts like pseudo-test data, which provides an independent measure of how well the model generalizes to new data and makes adjustments to improve its effectiveness.

![Training set, testing set and validation set.]({{ site.baseurl }}/assets/images/posts/machine-learning-fundamentals/training-testing-sets.png)

**Test data**

This set is used to evaluate the final performance of a trained machine learning model, providing independent examples with input features and target labels that the model has not seen during training or validation. It serves as an unbiased measure to assess the model’s effectiveness in real-world scenarios.

### **Model Training**

With the data ready, it is time to train and evaluate the machine learning models. Model training involves feeding the training data into a machine learning algorithm to adjust its parameters and optimize its performance.

During model training and evaluation, it’s important to watch out for two common pitfalls: overfitting and underfitting.

#### **Overfitting & Underfitting**

Overfitting occurs when a model learns the training data too well, capturing noise or irrelevant patterns that do not generalize to new data. This leads to poor performance on unseen data.

Underfitting occurs when a model is too simplistic to capture the underlying patterns in the data. This leads to poor performance both on the training and test datasets.

#### **Bias & Variance**

Overfitting and underfitting are related to the trade-off between bias and variance in model performance. Bias refers to the error due to overly simplistic assumptions in the model, and variance relates to the model’s sensitivity to fluctuations in the training data.

**Bias** represents the error introduced by the model’s assumptions or simplifications. High-bias models underfit the data, leading to poor performance on both training and test datasets.

**Variance** represents the sensitivity of our model to a given data point. High-variance models may overfit the data, capturing noise or irrelevant patterns and failing to generalize to new data.

A good model should strike a balance between bias and variance, known as the “*bias-variance tradeoff.”*

### **Hyperparameters**

Hyperparameters are like settings or configurations that govern how a machine learning model operates. These parameters are not learned from the data but are rather adjusted to control the learning of a model.

Hyperparameters can be considered like the knobs of the machine learning model, which we can adjust to make changes to how the model fits the data. Examples of hyperparameters are the maximum depth of a decision tree, the number of trees in a random forest, or the kernel type in SVM.

Methods like grid search and random search are used to find the optimal values for these hyperparameters in a process called [hyperparameter optimization](https://www.blog.trainindata.com/hyperparameter-tuning-for-machine-learning/) to achieve the best performance from a model.

### **Cross-validation**

Cross-validation is a technique used to assess the performance and generalization ability of machine learning models. It involves dividing the dataset into multiple subsets, training the model on different combinations of these subsets, and evaluating its performance on the remaining data, aiding in obtaining a more reliable estimate of the model’s performance.

K-fold cross-validation is a popular cross-validation technique. The dataset is divided into K equal-sized subsets (folds). The model is trained K times, each time using K-1 folds for training and the remaining fold for validation. This ensures that each data point is used for validation exactly once. The final performance is calculated by averaging the results from the K validation runs.

### **Model Evaluation**

To assess the performance of a model, we use evaluation metrics. These metrics measure the error in the model’s predictions. “Error” in machine learning refers to the difference between the predicted values generated by a model and the actual values observed in the dataset. The smaller the error, the better the performance of the model.

There are evaluation metrics for regression and for classification models.

#### **Regression Metrics**

There are several metrics that help us determine the performance of a regression model. Here, I describe the most common ones.

**Mean Squared Error (MSE)**: Measures the average squared difference between the predicted and actual values. A smaller MSE indicates better model performance.

**Root Mean Squared Error (RMSE)**: Similar to MSE but takes the square root of the average squared difference. It’s easier to interpret since it’s in the same units as the target variable.

**Mean Absolute Error (MAE)**: Measures the average absolute difference between the predicted and actual values. It provides a more interpretable measure of error compared to MSE.

**R-squared**: Indicates how well the independent variables in a regression model explain the variation in the dependent variable. R-qsuared values vary between 0 and 1, with higher values indicating better model fit.

#### **Classification Metrics**

These are the most common evaluation metrics for classification:

**Accuracy**: Measures the proportion of correctly classified instances.

**Precision**: measures the proportion of true positive predictions out of all positive predictions made by the model. It focuses on the accuracy of positive predictions.

**Recall**: Measures the proportion of true positive predictions out of all actual positive instances in the dataset. It focuses on the model’s ability to capture all positive instances.

**F1 Score**: The harmonic mean of precision and recall, the F1 score provides a balance between precision and recall.

**ROC Curve (Receiver Operating Characteristic Curve)**: A graphical plot that illustrates the trade-off between true positive rate (TPR) and false positive rate (FPR) across different threshold values. The higher the area under the ROC curve, the better the performance.

**Confusion matrix**: a table that summarizes the performance of a classification model by comparing actual and predicted class labels. It provides insights into the model’s true positive, true negative, false positive, and false negative predictions.

# **Conclusion**

Machine learning is revolutionizing how we approach digital challenges. It empowers computers to learn autonomously, uncover patterns in data, and transform industries with predictive insights. By grasping the machine learning basics, we open doors to endless possibilities, enabling collaboration between humans and machines for a brighter, more innovative future.

We’ve gathered a list of [data science and machine learning courses](https://www.blog.trainindata.com/data-science-and-machine-learning-courses/) and [machine learning books](https://www.blog.trainindata.com/machine-learning-books-for-beginners/) that can get you started on your journey.

[![Advanced machine learning courses at train in data.]({{ site.baseurl }}/assets/images/posts/data-science-prerequisites/image-for-newsletter-3-1024x536.png)](https://www.trainindata.com/p/advanced-machine-learning)

If you’ve already taken your first steps, you can boost your skills with our [advanced machine learning](https://www.trainindata.com/p/advanced-machine-learning) courses.

Happy learning and good luck!
