#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

###############################################################################
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
###############################################################################
features_list = ['poi','deferral_payments','total_stock_value',
                 'exercised_stock_options','long_term_incentive',
                 'shared_receipt_with_poi','ratio_to_messages',
                 'ratio_from_messages']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

###############################################################################
### Task 2: Remove outliers (Total data point)
###############################################################################
data_dict.pop( "TOTAL", 0 )
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)

###############################################################################
### Task 3: Create new feature(s)
###############################################################################
# New Feature
import pandas as pd
import numpy as np

df = pd.DataFrame.from_dict(data_dict,orient='index')
df['ratio_to_messages'] = \
  df['from_this_person_to_poi']/df['to_messages'].astype(float)
df['ratio_from_messages'] = \
  df['from_poi_to_this_person']/df['from_messages'].astype(float)

df = df.replace(np.nan,'NaN', regex=True)

data_dict = pd.DataFrame.to_dict(df, orient='index')
#new_features_list.remove('poi')
#new_features_list.insert(0,'poi')
#print data_dict
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)

labels, features = targetFeatureSplit(data)

###############################################################################
### Plot poi and non-poi features using a boxplot
###############################################################################
import matplotlib.pyplot as plt
poi_feature = []
non_poi_feature = []

for point in data:
    if int(point[0]) == 1:
        poi_feature.append(point[1])
    elif int(point[0]) == 0:
        non_poi_feature.append(point[1])

plt.figure()
plt.boxplot([poi_feature, non_poi_feature], labels=["Non-POI", "POI"])
plt.ylabel("feature")
plt.show()

###############################################################################
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
###############################################################################
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit

pipe = make_pipeline(MinMaxScaler(), SelectKBest(), PCA(), GaussianNB())#\
                     #DecisionTreeClassifier())
#print pipe.steps[2]

params = dict(selectkbest__k = range(4, 8),
              pca__n_components = range(1, 4),
              pca__svd_solver = ['auto']#,
              #decisiontreeclassifier__criterion = ['gini', 'entropy'],
              #decisiontreeclassifier__splitter = ['best', 'random'],
              #decisiontreeclassifier__min_samples_split = [2, 4],
              #decisiontreeclassifier__min_samples_leaf = [1, 2, 4],
              #decisiontreeclassifier__max_depth = [None, 5, 10, 15]              
              )

cv = StratifiedShuffleSplit(n_splits=20, test_size = 0.3, random_state = 42)

gs = GridSearchCV(pipe, param_grid=params, cv=cv, scoring='f1')

###############################################################################
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
###############################################################################

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.20, random_state=42)

# Select K Best Feature Scores
selectkbest = SelectKBest(k=3)
selectkbest = selectkbest.fit(features_test, labels_test)
print selectkbest.scores_

#clf = clf.fit(features_train, labels_train)
gs = gs.fit(features_train,labels_train)
clf = gs.best_estimator_
#print gs.best_params_
#print clf.score_
#print gs.best_estimator_

# fit and calculate the accuracy of the classifier
from sklearn.metrics import classification_report

pred = clf.predict(features_test)
acc = clf.score(features_test, labels_test)
report = classification_report(labels_test, pred)
print "Classifier Accuracy: ", acc
print report

# Calculate the precision and recall of the classifier
pred = np.array(pred)
print "Index of predicted POIs: ", np.where(pred == 1.0)
labels_test = np.array(labels_test)
print "Index of actual POIs: ", np.where(labels_test == 1.0)

###############################################################################
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
###############################################################################

dump_classifier_and_data(clf, my_dataset, features_list)