# Credit Risk Analysis

## Overview 

Fast Lending, a peer to peer lending company wants to use machine learning to predict credit risk to provide a quicker and more reliable loan experience. In this analysis we will use sampling methods and train machine learning models to analyze credit data and predict default rates in order to rationalize loan approval criteria. First we will use resampling methods to predict credit risk. We will oversample data using the Random Over Sampling and Synthetic Minority Oversampling Technique (SMOTE), then we will undersample using Cluster Centroids algorithm. We will then train a logistic regression model and evaluate the model with an accuracy score, a confusion matrix and a classification report. Secondly, we will use the Synthetic Minority Oversampling Technique Edited Nearest Neighbors (SMOTEENN) approach to predict risk. This approach combines SMOTE and Edited Nearest Neighbors algorithms. We will similarly train the logistic regression model and evaluate the effectiveness of the model and we will also compare the results against resampling algorithms. Lastly we will use Ensemble Classifiers to predict credit risk. We will compare two different ensemble classifiers, Balanced Random Forest Classifier and Easy Ensemble Classifier. Using both these algorithms we will resample the dataset, train the ensemble classifier and evaluate the effectiveness of the model and compare these between between the two classifiers. In this analysis the main metric we may want to look at is the False Negative rate because as a these are high risk loans predicted as low risk loans. Recalll for high risk loans would be a good indicator of this as well. By classifying bad loans as good loans, we are increasing our risk of not getting paid back the loan, affecting our profitibility.

## Results 

### Resampling Methods 

* Created our train and target variable, and divided our observations using a train-test split
* Analyzed outcomes of the Logistic Regression analysis using Naive Random and SMOTE Oversampling, Cluster Centroids Undersampling, and SMOTEENN combination over and undersampling 
* Naive Random Oversampling methods produced an accuracy score of 64.73%, 70 true positives, 31 false negatives, 6813 false positives and 10291 true negatives, precision of 0.01 and recall of 0.69 for high risk and precision of 1.00 and recall of 0.60 for low risk. The method did not perform well

![alt Text](https://github.com/mobinapiracha/Credit_Risk_Analysis/images/Naive_Random_Sampling.PNG)

* SMOTE oversampling methods produced an accuracy score of 66.21%, 64 true positives, 37 false negatives, 5291 false positives and 11813 true negatives, precision of 0.01 and recall of 0.63 for high risk and precision of 1.00 and recall of 0.69 for low risk. The method did not perform well
* The Cluster Centroids Undersampling produced an accuracy score of 54.43%, 70 true positives, 31 false negatives, 10340 false positives and 6764 true negatives, precision of 0.01 and recall of 0.69 for high risk and precision of 1.00 and recall of 0.40 for low risk. The method did not perform well
* Combination Sampling with SMOTEENN produced an accuracy score of 67.75%, 79 true positives, 22 false negatives, 7305 false positives, 9799 true negatives, precision of 0.01 and recall of 0.78 for high risk and precision of 1.00 and recall of 0.57 for low risk. This method performed better relative to other sampling methods but overall did not perform well.

### Ensemble Learners 

* We used Ensemble learners such as Balanced Random Forest and Easy Ensemble AdaBoost Classifier to see if these methods predict better results. 
* Balanced Random Forest produced an accuracy score of 78.85%, 71 true positives, 31 false negatives, 2153 false positives and 14951 true negatives. A precision of 0.03 and recall of 0.7 for high risk and precision of 1.00 and recall of 0.87 for low risk, the model has performed much better than the results above. Easy Ensemble Classifier performs even better with a an accuracy score of 93%, 93 true positives, 8 false negatives, 983 false positives and 16121 true negatives. A precision score of 0.09 and recall of 0.92 for high risk and a precision of 1.00 and a recall of 0.94 for low risk. This model performs the best out of all the classifiers. 

## Summary 

A problem with this dataset was the quantity of low risk loans relative to high risk loan, this can cause an imbalance in accuracy scores. Therefore, it is important to look at the confusion matrix and classification report as accuracy scores can be deceiving. However, we find that resampling methods have performed poorly, with accuracy scores no greater than 70% for any of the oversampling and undersampling, as well as combination sampling done. A lot of misclassifications, especially since the recall for high risk loans still below 0.8 for all methods. SMOTEENN method however, performed the best among all resampling method with a recall of 0.78 for high risk loans.However, like all other methods it performed poorly on all other metrics and misclassified many low risk loans as high risk loans. 

Ensemble learners, especially Easy Ensemble AdaBoost Classifier despite dealing with a very low number of high risk observations managed to perform much better than resampling with logistic regression. Balanced Random Forest performs relatively average with a recall of 0.70 for and an accuracy score of 78.85%. However, Easy Ensemble Classifier turns out to perform the best among all classifiers only misclassifying 8 out of 93 high risk loans as low risk which shows with the recall of 0.92. It also performs much better than other classifiers only misclassifying 983 low risk loans as 983 out of 17104 total low risk loans, which shows with a recall rate of 0.94 for low risk loans. 

Therefore, I would recommend the Easy Ensemble AdaBoost Classifier machine learning model to help classify loans due to its very high recall for both low risk and most importantly high risk loans. This model would help minimize risk and ensure that fewer loans are provided to individuals with high risk of defaulting. 
