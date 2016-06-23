import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import unidecode
import csv 
import io
import nltk
import string
from time
import re
from nltk import wordpunct_tokenize
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk import bigrams
from nltk.stem.snowball import EnglishStemmer
# definition
def readdata(filename):
    X=[]
    y=[]
    with io.open(filename, encoding = 'utf-8') as f:
        for line in f:
            y.append(line[0])
            X.append(line[5:])
    return y,X    


# get argument list using sys module

# sys.stdout.write( "will read file from :" + filename + "and save result to: " + result)
# sys.stdout.flush() 
print "programme starting!"

y, X = readdata('train.csv')

from sklearn.feature_extraction.text import CountVectorizer
bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=1)
dict_result  = bigram_vectorizer.fit_transform(X)

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.feature_selection import chi2,SelectKBest
from sklearn.feature_extraction.text import TfidfTransformer

tf_transformer = TfidfTransformer(use_idf=False)
clf = SVC(kernel='linear', gamma=2)
select = SelectKBest(chi2)
pipe = Pipeline(steps=[('chi2', select), ("tf-idf",tf_transformer ),('SVM', clf)])
para = {"chi2__k" : np.arange(50, 3100, 200), "tf-idf__use_idf" : [True, False]}
estimator = GridSearchCV(pipe,para,n_jobs = 4)
print("result withbigram !")
print("Performing grid search...")
print("pipeline:", [name for name  in pipe.steps])
print("parameters:")
print(para)
t0 = time()
estimator.fit(dict_result.toarray(), y)
print("done in %0.3fs" % (time() - t0))
print()

print("Best score: %0.3f" % estimator.best_score_)
print("Best parameters set:")
best_parameters = estimator.best_estimator_.get_params()
for param_name in sorted(para.keys()):
	print("\t%s: %r" % (param_name, best_parameters[param_name]))

