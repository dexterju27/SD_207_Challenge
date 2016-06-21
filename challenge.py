import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import sparse
import pandas as pd
import unidecode
import csv 
import io

train_fname = 'train.csv'
test_fname = 'test.csv'
X = []
y = []
test = []
with io.open(train_fname, encoding = 'utf-8') as f:
     for line in f: 
        line = line.replace("\\xa0", "")
        line = line.replace("\\n", "")
        line = line.lower()
        y.append(int(line[0]))
        X.append(line[5:-6])
y = np.array(y)

with io.open(test_fname, encoding = 'utf-8') as f:
     for line in f: 
        line = line.replace("\\xa0", "")
        line = line.replace("\\n", "")
        line = line.lower()
        # y.append(int(line[0]))
        test.append(line)
# y = np.array(y)
print test[0]
print('n_samples : %d' % len(X))

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X)
X_test_counts = count_vect.transform(test)
print X_train_counts.shape
count_vect.vocabulary_.get(u'fuck')
from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
X_test_tf = tf_transformer.transform(X_test_counts)
print X_train_tf.toarray().shape
print len(char_counts)
# X_train_tf = sp.sparse.hstack((X_train_tf, char_counts))

# This dataset is way to high-dimensional. Better do PCA:
pca = PCA(n_components=2050)

# Maybe some original features where good, too?
# election = SelectKBest(k=500)

# combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])
combined_features = FeatureUnion([("pca", pca)])

# Use combined features to transform dataset:
X_features = combined_features.fit(X_train_tf.toarray(), y).transform(X_train_tf.toarray())
# test_features = combined_features.transform(X_test_tf)

from sklearn.ensemble import BaggingClassifier
from sklearn.cross_validation import cross_val_score
clf = SVC(kernel='linear', gamma=2)
bagging = BaggingClassifier(clf,max_samples=0.5, max_features=0.5, n_estimators=20)
print cross_val_score(bagging, X_train_tf.toarray(), y)
