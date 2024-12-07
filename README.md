# Supervised Learning in Python
Welcome to the Supervised Learning in Python repository! This repository will feature projects and studies on supervised learning algorithms implemented in Python. The main goal is to understand, implement, and compare various supervised learning techniques.

## Planned Topics
The following algorithms and topics will be covered step by step in this repository.

k-Nearest Neighbors (KNN),
Decision Trees,
Random Forest,
Logistic Regression,
Support Vector Machines (SVM),
Naive Bayes,
Comparison of Classification Algorithms,
Linear Regression,
Multiple Linear Regression,
Polynomial Regression

## Repository Contents
### 1. K-Nearest Neighbors (KNN)

#### KNN.py
A classification project where the load_breast_cancer dataset from the sklearn library is used. The dataset is analyzed and modeled using the KNN algorithm.

#### RegressionKNN.py
A regression-focused implementation of KNN. A custom dataset of random numbers is generated, and KNN is applied for regression tasks.


### 2. Desicion Trees (DT)
#### DT_1.py
This project demonstrates the creation and evaluation of a Decision Tree model using the popular sklearn.iris dataset.A Decision Tree classifier was created and trained on the dataset.The Decision Tree structure was visualized.Features were ranked from most to least important based on their contribution to the model.

#### DT_2.py
This file consists of an analysis with the Decision Tree algorithm using the Iris dataset. Different combinations of features were tested to see which features give better results when used together. During the analysis, the classification boundaries were visualized and the results were interpreted.

#### DT_3.py
This project involves predicting the target variable with a Decision Tree Regressor model using the Diabetes dataset. The model is trained on data split into training and test data and its performance is evaluated by Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) metrics.

#### DT_4.py
This project aims to make predictions using a Decision Tree Regression model on a noisy data set. The dataset consists of 80 random samples and contains values generated by a sine function for each sample. In addition, a random noise is added to every 5th value. The performance of the model is compared with two decision tree regressors with different max_depth parameters. The visualization visually presents how the model works and the impact of different depths on the predictions.

### Random Forest(rf)
#### RF_1.py
This project performs face recognition classification with Random Forest classifier using Olivetti Faces dataset. The faces in the dataset are classified with the Random Forest algorithm. After separating the training and test data, the accuracy of the model is calculated and the results are visualized. The project is an example of how to use the Random Forest algorithm in face recognition.

#### RF_2.py
This Python code uses the Random Forest algorithm to solve a regression problem with the California Housing dataset. The dataset is taken from the sklearn library and split 80% training and 20% testing. The model is trained on the training data and then makes predictions on the test data. Model performance is evaluated by the mean squared error (MSE) and root mean squared error (RMSE) of the predictions to the true values. The code provides a regression example for house price forecasting.

### Logistic Regression(LgR.py)
This Python script builds a logistic regression model with Heart Disease data from the UCI Machine Learning Repository. The data is extracted with the ucimlrepo library and missing values are removed. The features and target are split into 90% training and 10% testing. The logistic regression model is trained using sklearn and its accuracy on the test data is calculated. The code provides a simple classification example for heart disease detection.


Stay tuned for updates as more algorithms and projects will be added soon!
