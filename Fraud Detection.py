# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 22:06:35 2017

@author: Collins Nyagaya
"""

# load initial libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read in the data and visualize header
credit = pd.read_csv("creditcard.csv", na_values=["?"])
credit.head()

# Section 2: Data Preprocessing

# Feature "time" is the number of seconds elapsed between each transaction and 
# the first transaction in the dataset. Feature 'Amount' is the transaction 
# Amount, while feature 'Class' is the target variable and with value 1 = fraud 
# and 0 otherwise. As noted earlier, the other variables v1-v28 are principal 
# components of the original data.

credit.shape

# We see the data has two classes - per data description, 1=fraud, 0 otherwise
pd.unique(credit["Class"].values.ravel())
# number of classes
print("Number of classes:", len(pd.unique(credit["Class"].values.ravel())))

# As expected, the dataset suffers from severe class imbalance. 
# We see that class of interest (fraudulent transactions) is less than 1% - 
# from the visualization.
# examining the class distribution
cr = credit["Class"].value_counts()/credit["Class"].count()
fig = plt.figure(figsize=(10,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Target/Class')
ax1.set_ylabel('Percentage')
ax1.set_title("Class Distribution")
cr.plot(kind='bar', grid = True)

# print percentages of the two classes
fraud_count = len(credit[credit["Class"]==1])
not_fraud_count = len(credit[credit["Class"]==0])

percent_not_fraud = not_fraud_count/(not_fraud_count + fraud_count)
fraud_percent = fraud_count/(not_fraud_count + fraud_count)
print("Not fraud transactions percent: ", percent_not_fraud*100)
print("Fraudulent transactions percent: ", fraud_percent*100)


fraud = credit[credit["Class"]==1]
normal = credit[credit["Class"]==0]

plt.figure(figsize=(10,6))
plt.subplot(121)
fraud.Amount.plot.hist(title="Fraudulent Transactions")
plt.subplot(122)
normal.Amount.plot.hist(title="Non-Fraudulent Transactions")

# review column names
credit.columns
# .info provides data type for each variable, as well as indicates if there 
# are null values in the data
credit.info()

# another confirmation there are missing values
credit.isnull().sum()

# Though not meaningful, the summary stats indicate that the minimum transaction 
# amount is 0.00 dollars while maximum amount is about 26,000 dollars. 
# The mean amount is 88.00 dollars, and we also do see that 75 percent of all 
# transactions were less than 77.00 dollars.

# though not meaningful, we get the summary statistics of the data
credit.describe(include="all")
# assigning the predictor variables to object X
X = credit[['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']]
X.head(5)

# assigning target variable to object y
y = credit.Class

# we want all variable to be on a common scale. 
# Standardization, rather than normalization, has been proven to be more effective
# as it essentially normalizes the variables to mean 0 and standard deviation 1 
# thus achieving a normal distribution.
from sklearn.preprocessing import StandardScaler
X["Amount_scl"] = StandardScaler().fit_transform(X["Amount"].values.reshape(-1, 1))
X = X.drop(["Time", "Amount"], axis=1)
X.head(10)

# with data standardized, we split into train and test set, both X and Y sets
# we retain 30% to test the model, using train_test_split function
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.3, 
                                                    random_state=0)

# printing the split data
print("Predictor tain set: ", X_train.shape)
print("Predictor test set: ", X_test.shape)
print("Target train set: ", y_train.shape)
print("Target test set: ", y_test.shape)


# Modeling

# The data is clearly class-imbalanced. While there are a number of techniques 
# to dealing with class imbalance issues, we will first build a few models 
# without rebalancing, and assess model quality, afterwhich we will make 
# certain decisions on how to balance the dataset

# defining a function to measure model performance
from sklearn import metrics

def measure_performance(X, y, clf, show_accuracy=True, show_classification_report=True, show_confussion_matrix=True):
    y_pred = clf.predict(X)   
    if show_accuracy:
         print("Accuracy:{0:.3f}".format(metrics.accuracy_score(y, y_pred)),"\n")
    if show_classification_report:
        print("Classification report")
        print(metrics.classification_report(y, y_pred),"\n")
      
    if show_confussion_matrix:
        print("Confussion matrix")
        print(metrics.confusion_matrix(y, y_pred),"\n")

### Logistic Regression - imbalanced data

# import logistic function from sklearn
# and initialize the logistic reg object
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1.0, random_state=0)
lr.fit(X_train, y_train)

# make predictions based on test set
lrpred_test = lr.predict(X_test)

# import classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, lrpred_test))

from sklearn.metrics import confusion_matrix
# initialize the confusion matrix
lr_cm = (confusion_matrix(y_test, lrpred_test))

# visualize the confusion matrix
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
plt.matshow(lr_cm, cmap = plt.cm.Blues)
plt.title("Logistic Regression Confusion Matrix\n")
plt.ylabel("Actual")
plt.xlabel("Predicted")
for y in range(lr_cm.shape[0]):
    for x in range(lr_cm.shape[1]):
        plt.text(x, y, '{}'.format(lr_cm[y, x]),
                horizontalalignment = 'center',
                verticalalignment = 'center',)
plt.show()


# Random Forest - imbalanced data

# fitting a base random forest on class imbalance data
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf = rf.fit(X_train, y_train)

# measure performance, on the original unbalanced test set
measure_performance(X_test, 
                    y_test, 
                    rf, 
                    show_confussion_matrix=True, 
                    show_classification_report=True)


# Data Balancing

# From the simple logistic regression and random forest, it is clear any 
# algorithm trained on the class-imbalanced data will probably have lower 
# recall values. It is imperative we employ techniques for dealing with class 
# imbalance. General techniques include:
    
# 1. Resampling techniques - over/under sampling
# 2. Different performance metrics besides Accuracy: recall/precision, 
# sensitivity, F-measure, AUCPR and Kappa.
# 3. SMOTE - synthetically generate additional features of rare class
# 4. Threshold moving
# 5. Ensemble techniques
# 6. Collect additional data - often expensive, not practical!

# For the modeling tasks included here, we implemented techniques 1 
# (undersampling), 2, and 5 (ensemble technique).


# libraries needed for undersampling
from imblearn.over_sampling import SMOTE
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler

# initialize the undersampler
rus = RandomUnderSampler(random_state=0)
# we undersample the original TRAINING data, while leaving the original test 
# set as unseen data
X_res, y_res = rus.fit_sample(X_train, y_train)
# while faily small, we get a balanced sample that includes 70 percent of 
# fraudulent transactions
print('Resampled dataset shape {}'.format(Counter(y_res)))

# Random Forest - balanced data

# With the data balanced, using gridsearch, we fit a hyperparameter tuned 
# random forest on the undersampled data. We then validate model peformance on 
# the original class-imbalance test set.

# initialize random forest estimator
rf_test = RandomForestClassifier(n_estimators=100)
# import libraries needed
from time import time
from scipy.stats import randint as sp_randint
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# use a full grid over all parameters
param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 4, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# run grid search
grid_search = GridSearchCV(rf_test, param_grid=param_grid)
start = time()
grid_search.fit(X_res, y_res)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))

# getting the best grid search parameters
grid_search.best_params_

# fit the final model
from sklearn.ensemble import RandomForestClassifier
rf_final = RandomForestClassifier(n_estimators=100, 
                            min_samples_split=2, 
                            max_depth=None, 
                            max_features=10,
                           min_samples_leaf=3,
                           criterion="entropy",
                           bootstrap=False)
rf_final = rf_final.fit(X_res, y_res)

measure_performance(X_test, 
                    y_test, 
                    rf_final, 
                    show_confussion_matrix=True, 
                    show_classification_report=True)
