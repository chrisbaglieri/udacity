#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# limit training set sizes
# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100] 

clf = SVC(C=10000, kernel = 'rbf')
clf.fit(features_train, labels_train)

pred = clf.predict(features_test)

# accuracy of our algorithm
accuracy = accuracy_score(pred, labels_test)
print accuracy

# particular predictions
print pred[10]
print pred[26]
print pred[50]

# summing up the positives gives us the number of chris emails
c_predicted = sum(pred)
print c_predicted

#########################################################