import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import unidecode
import csv 
import io
import nltk
import string
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
filename  = sys.argv[0]
result = sys.argv[1]
print "will read file from :" + filename + "and save result to: " + result;

y, X = readdata(filename)

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
estimator = GridSearchCV(pipe,para)
estimator.fit(dict_result.toarray(), y)
sys.stdout.write("result withbigram !")
sys.stdout.write( "best score "+ estimator.best_score_)
sys.stdout.flush() 
f = open(result, 'w')
f.write("The best test score is " + estimator.best_score_ + " with " +estimator.best_params_)

