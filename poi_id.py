#!/usr/bin/python

import sys
import pickle
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

sys.path.append('../tools/')
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

### Load the dictionary containing the dataset
with open("final_project_dataset_windows.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

# Original Features that made the top 10 using selectKbest
original_features_list = ['poi', 'total_stock_value', 'exercised_stock_options', 'bonus', 'salary', 'deferred_income',
                          'long_term_incentive', 'total_payments', 'restricted_stock', 'shared_receipt_with_poi',
                          'expenses']

### Task 2: Remove outliers

# Loading the dictionary as dataframe
first_key = list(data_dict.keys())[0]
df_enron = pd.DataFrame.from_dict(data_dict, orient='index', columns=data_dict[first_key].keys())

# Correct the datatypes
for col in df_enron.columns:
    if (col != 'poi') & (col != 'email_address'):
        df_enron[col] = df_enron[col].where(df_enron[col] != 'NaN', np.nan)
        df_enron[col] = df_enron[col].astype('float')

# correct Robert Belfer and Sanjay Bhatnagar entries
financial_features = ['salary', 'bonus', 'long_term_incentive', 'deferred_income', 'deferral_payments', 'loan_advances',
                      'other', 'expenses', 'director_fees', 'total_payments', 'exercised_stock_options',
                      'restricted_stock', 'restricted_stock_deferred', 'total_stock_value']

for pos in range((len(financial_features) - 1), 1, -1):
    df_enron[financial_features[pos]]['BHATNAGAR SANJAY'] = df_enron[financial_features[pos - 1]]['BHATNAGAR SANJAY']
    df_enron[financial_features[pos]]['BELFER ROBERT'] = df_enron[financial_features[pos - 1]]['BELFER ROBERT']

# Remove the Total line of the dataset
df_enron.drop(['TOTAL'], inplace=True)


### Task 3: Create new feature(s)

# Build the features that made to the top 10 features
def get_email_ratio(poi_messages, all_messages):
    '''
    Function to build new columns based on the ratio of messages from a POI to a person and from a person to POI
    Edited from Udacity Class

    output:
        df_edited - dataframe with the aditional columns

    '''

    if all_messages != 0:
        fraction = poi_messages / all_messages
    else:
        fraction = 0.

    return fraction


# to_poi_ratio - features evaluated on Udacity class
# shared_poi_ratio - features created by the student

# initiate the list to store the values
to_poi_ratio = []
shared_poi_ratio = []

# iterate the dataframe for each person
for person in df_enron.index:
    # get the values to run the built fuction get_email_rate
    to_poi = df_enron.loc[person]['from_this_person_to_poi']
    from_messages = df_enron.loc[person]['from_messages']

    shared_poi = df_enron.loc[person]['shared_receipt_with_poi']
    to_messages = df_enron.loc[person]['to_messages']

    # store the proportion return on the list
    to_poi_ratio.append(get_email_ratio(to_poi, from_messages))
    shared_poi_ratio.append(get_email_ratio(shared_poi, to_messages))

df_enron['to_poi_ratio'] = to_poi_ratio
df_enron['shared_poi_ratio'] = shared_poi_ratio

features_list = ['poi', 'total_stock_value', 'exercised_stock_options', 'bonus', 'salary', 'to_poi_ratio',
                 'deferred_income', 'long_term_incentive', 'shared_poi_ratio', 'total_payments', 'restricted_stock']

### Store to my_dataset for easy export below.

# missing values will be filled with 0
df_enron.fillna(0., inplace=True)

my_dataset = df_enron.to_dict('index')

### Extract features and labels from dataset for local testing
original_data = featureFormat(my_dataset, original_features_list, sort_keys=True)
original_labels, original_features = targetFeatureSplit(original_data)

data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
#from time import time


print(f'{" Naive Bayes ":=^30}')
from sklearn.naive_bayes import GaussianNB
pipe = Pipeline([('scaler', MinMaxScaler()), ('PCA', PCA(svd_solver='auto', whiten=True)), ('naive', GaussianNB())])
param_grid = ([{'PCA__n_components': [1, 3, 5, 7]}])

print(f'{" Original Data ":=^20}')
#t0 = time()
clf = GridSearchCV(pipe, param_grid, scoring='recall').fit(features, labels).best_estimator_
#print(f'Trained in {time() - t0} s')
test_classifier(clf, my_dataset, original_features_list)
'''
Models tried, but with lower performance
print(f'{" SVM ":=^30}')
## SVM Model
from sklearn.svm import SVC

pipe = Pipeline([('scaler', MinMaxScaler()), ('PCA', PCA(whiten=True ,svd_solver ='auto')), ('svm', SVC())])
param_grid = ([{'PCA__n_components': [1, 3, 5, 7], 'svm__kernel': ['rbf'],
                'svm__C': [10, 1000, 10000], 'svm__gamma': [0.01, 0.1, 0.6, 1]},
               {'PCA__n_components': [1, 2, 5, 7], 'PCA__whiten':[True, False], 'svm__kernel': ['poly'],
                'svm__C': [10, 1000, 10000], 'svm__gamma': [0.01, 0.1, 0.6, 1],
                'svm__degree': [1, 3, 5, 7]}])

print(f'{" Original Data ":=^20}')
t0 = time()
clf = GridSearchCV(pipe, param_grid, scoring='recall').fit(features, labels).best_estimator_
print(f'Trained in {time() - t0} s')
test_classifier(clf, my_dataset, original_features_list)

print(f'{" K-Nearest Neighbours ":=^30}')
### K-Nearest Neighbours
from sklearn.neighbors import KNeighborsClassifier
pipe = Pipeline([('scaler', MinMaxScaler()), ('PCA', PCA(whiten=True, svd_solver='auto')),
                 ('neigh', KNeighborsClassifier())])
param_grid = ([{'PCA__n_components': [1, 3, 5, 7], 'neigh__n_neighbors': [2, 5, 7, 10, 50]}])

print(f'{" Original Data ":=^20}')
t0 = time()
clf = GridSearchCV(pipe, param_grid, scoring='recall').fit(features, labels).best_estimator_
print(f'Trained in {time() - t0} s')
test_classifier(clf, my_dataset, original_features_list)

print(f'{" Adaboost ":=^30}')
# Adaboost
from sklearn.ensemble import AdaBoostClassifier
pipe = Pipeline([('scaler', MinMaxScaler()), ('PCA', PCA(svd_solver='auto', whiten=True)), ('ada', AdaBoostClassifier())])
param_grid = ([{'PCA__n_components': [1, 2, 5, 7], 'ada__n_estimators': [5, 20, 40, 60, 100],
                'ada__learning_rate': [0.5, 1, 2, 3, 5]}])

print(f'{" Original Data ":=^20}')
t0 = time()
clf = GridSearchCV(pipe, param_grid, scoring='recall').fit(features, labels).best_estimator_
print(f'Trained in {time() - t0} s')
test_classifier(clf, my_dataset, original_features_list)
'''
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

pipe = Pipeline([('scaler', MinMaxScaler()), ('PCA', PCA(svd_solver='auto', n_components=5, whiten=True)),
                ('naive', GaussianNB())])

print(f'{" With New Features ":=^30}')
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)
clf = pipe.fit(features, labels)
test_classifier(clf, my_dataset, features_list)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
dump_classifier_and_data(clf, my_dataset, features_list)
