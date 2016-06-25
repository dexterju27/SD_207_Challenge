
# coding: utf-8

# ### Attention
# #### Il faut installer enchant pour corriger des mots
# #### istaller nltk et aussi ajouer un wordnet en lancant  nltk.download()
# #### aussi j'ai remplace le ficher de stopswords par un autre version plus complete

# In[246]:

# -*- coding: <utf-8> -*-
## pip install pyenchant, ntlk
get_ipython().magic(u'matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import unidecode
import csv 
import math
import io
import nltk
import string
import re
from nltk import wordpunct_tokenize
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk import bigrams
from nltk.stem.snowball import EnglishStemmer


# In[247]:

from nltk.corpus import wordnet
# nltk.download()


# In[248]:

def removehtmltags(line):    
    p=re.compile(r"<.*?>")
    line = p.sub(' ', line)
    line = line.replace('&nbsp;',' ')   
    return line


# In[249]:

def removebadtoken(line):
    rep = {'\\xa0': ' ', '\\xc2': ' ', '\\n': ' ', '\r': ''}
    rep = dict((re.escape(k), v) for k, v in rep.items())
    pattern = re.compile("|".join(rep.keys()))
    line = pattern.sub(lambda m: rep[re.escape(m.group(0))], line)
    return line   


# In[250]:

def commentnormalizer(line):
    line = "".join([ch for ch in line if ch not in string.punctuation])
    line = re.sub("\\d+(\\.\\d+)?", "NUM", line)
    return line


# In[251]:

def Tokenization(line):
    line=line.lower()
    tokenlist = wordpunct_tokenize(line)
    return tokenlist    


# In[252]:

def removestopwords(tokenList):
    stopwordList=np.genfromtxt('stopwords.txt',dtype='str')
    filteredWords = [w for w in tokenList if not w in stopwordList]
    return filteredWords    


# In[253]:

def wordreplace(word):
    if wordnet.synsets(word):
        return word
    repl_word=re.compile(r'(\w*)(\w)\2(\w*)').sub(r'\1\2\3',word)
    if repl_word!=word:
        return wordreplace(repl_word)
    else:
        return repl_word
def repeatreplace(tokenList):
    for i in range(len(tokenList)): 
        word=tokenList[i]
        word=wordreplace(word)
        tokenList[i]=word
    return tokenList


# In[254]:

repeatreplace(["fuuuuck","shiittt"])


# In[255]:

def commentStemmer(tokenList):
    stemmer = EnglishStemmer()
    for i in range(len(tokenList)):
        tokenList[i] = stemmer.stem(tokenList[i])
    return tokenList 


# In[256]:

import enchant
from nltk.metrics import edit_distance
def commentcorrect(tokenList):
    dict_name="en"
    spell_dict=enchant.Dict(dict_name)
    max_dist=2
    for i in range(0,len(tokenList)):
        if spell_dict.check(tokenList[i]):
            tokenList[i]=tokenList[i]
        else:
            suggestions=spell_dict.suggest(tokenList[i])
            if suggestions and edit_distance(tokenList[i],suggestions[0])<=max_dist:
                tokenList[i]=suggestions[0]
            else:
                tokenList[i]=tokenList[i]
    return tokenList        


# In[257]:

commentcorrect(["sh?t","languege"])


# In[258]:

## badwords in the bad words list
badList=[]
with io.open('badwords.txt', newline = '\n') as f:
    for line in f:     
        badList.append(line[:len(line)-1])
def encountebadwords(tokenList):  
    count=0
    for w in tokenList:
        if w in badList:
            count=count+1
    return  count


# In[512]:


X_li=[]
with io.open('X_train_preprocess.txt', newline = '\n') as f:
    for line in f:
        X_li.append(line[:len(line)-1])


# In[513]:

print np.shape(X_li)


# In[526]:

Data=(X_li+X_test)
a=Data[0]


# In[531]:

a=[["a"],["ff"],["sd"]]
pr


# In[260]:

### seem as creat unique word list
def calculateTermFrequency(tokenList):
    dictionary={}
    for i in range(0,len(tokenList)):
        if tokenList[i] in dictionary:
            dictionary[tokenList[i]] += 1
        else:
            dictionary[tokenList[i]] = 1
    return dictionary  


# In[262]:

## calculate the number of unique words in the whole doc
def createWordList(X):
    wordContent = []
    wordCount = dict()
    for line in X:
        for word in line:
            if word in wordCount.keys():
                wordCount[word] = wordCount[word] + 1 + (word in badList)
            else:
                wordCount[word] = 1 + (word in badList)
    return wordCount.keys(), wordCount
def words2Vectors(wordList, data):
    vec = [0] * len(wordList)
    for word in data:
        if word in wordList:
            vec[wordList.index(word)] = vec[wordList.index(word)] + 1  + (word in badList) 
            # count bad words twice
    return vec
def termFrequency(X):
    wordsList=createWordList(X)[0]
    X_Vector=[0]*len(X)
    for i in range(len(X)):
        X_Vector[i]=words2Vectors(wordsList,X[i])
    X_Vector=np.ravel(X_Vector).reshape((np.shape(X_Vector)[0],np.shape(X_Vector)[1]))
    return X_Vector 


# In[263]:

def convertToBigrams(tokenList):
    bigramList = []
    for i in range(len(tokenList)):
        if i == 0:
            continue
        bigram = tokenList[i-1] + ' ' + tokenList[i]
        bigramList.append(bigram)
    return bigramList 
def createbiWordList(X):
    biwordContent = []
    biwordCount = dict()
    for line in X: 
        for biword in convertToBigrams(line):
            if biword in biwordCount.keys():
                biwordCount[biword] = biwordCount[biword] + 1
            else:
                biwordCount[biword] = 1 
    return biwordCount.keys(), biwordCount
def biwords2Vectors(biwordList, data):
    vec = [0] * len(biwordList)
    for biword in convertToBigrams(data):
        if biword in biwordList:
            vec[biwordList.index(biword)] = vec[biwordList.index(biword)] + 1 
    return vec
def Bigramtransform(X):
    biwordsList=createBiWordList(X)[0]
    X_biVector=[0]*len(X)
    for i in range(len(X)):
        X_biVector[i]=biwords2Vectors(biwordsList,X[i])
    X_biVector=np.ravel(X_biVector).reshape((np.shape(X_biVector)[0],np.shape(X_biVector)[1]))
    return X_biVector  


# In[265]:

## TF_idf Transform
import scipy.sparse as sp
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import normalize

def _document_frequency(X):
    if sp.isspmatrix_csr(X):
        return bincount(X.indices, minlength=X.shape[1])
    else:
        return np.diff(sp.csc_matrix(X, copy=False).indptr)
class TfidfTransformer():
    def __init__(self, norm='l2', use_idf=True, smooth_idf=True,
                 sublinear_tf=False):
        self.norm = norm
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf

    def fit(self, X, y=None):
        if not sp.issparse(X):
            X = sp.csc_matrix(X)
        if self.use_idf:
            n_samples, n_features = X.shape
            df = _document_frequency(X)
            
            # perform idf smoothing if required
            df += int(self.smooth_idf)
            n_samples += int(self.smooth_idf)

            # original formula is  tf* idf, now we use (tf* (1+idf)) === log+1 instead of
            # log makes sure terms with zero idf don't get suppressed entirely.
            idf = np.log(float(n_samples) / df) + 1.0
            self._idf_diag = sp.spdiags(idf,
                                        diags=0, m=n_features, n=n_features)

        return self

    def transform(self, X, copy=True):
        if hasattr(X, 'dtype') and np.issubdtype(X.dtype, np.float):
            X = sp.csr_matrix(X, copy=copy)
        else:
            X = sp.csr_matrix(X, dtype=np.float64, copy=copy)

        n_samples, n_features = X.shape

        if self.sublinear_tf:
            np.log(X.data, X.data)
            X.data += 1

        if self.use_idf:
            check_is_fitted(self, '_idf_diag', 'sorry,idf vector is not fitted-_-')

            expected_n_features = self._idf_diag.shape[0]
            if n_features != expected_n_features:
                raise ValueError("Oh,input has n_features=%d while the model"
                                 " has been trained with n_features=%d,please check it first" % (
                                     n_features, expected_n_features))
            # why *= += can't work in python??????
            X = X * self._idf_diag

        if self.norm:
            X = normalize(X, norm=self.norm, copy=False)

        return X

    def fit_transform(self, X, y=None, **fit_params):
        if y is None:
            return self.fit(X, **fit_params).transform(X)
        else:
            return self.fit(X, y, **fit_params).transform(X)


# In[369]:

def commentlength(train_filename):
    y,X=readdata(train_filename)
    L=[]
    for i in range(0,len(X)):
        line=removehtmltags(X[i])
        line=removebadtoken(line)
        line=commentnormalizer(line)
        tokenList=Tokenization(line)
        L.append(np.shape(tokenList)[0])
    return L


# In[574]:

def commentlengthtest(filename):
    X=readtestdata(filename)
    L=[]
    for i in range(0,len(X)):
        print i
        line=removehtmltags(X[i])
        line=removebadtoken(line)
        line=commentnormalizer(line)
        tokenList=Tokenization(line)
        L.append(np.shape(tokenList)[0])
    return L


# In[303]:

def readtraindata(filename):
    X=[]
    y=[]
    with io.open(filename, encoding = 'utf-8') as f:
        for line in f:
            y.append(line[0])
            X.append(line[5:])
    return y,X   
def processingtrain(train_filename):
    y,X=readdata(train_filename)
    Count=[]
    for i in range(0,len(X)):
        line=removehtmltags(X[i])
        line=removebadtoken(line)
        line=commentnormalizer(line)
        tokenList=Tokenization(line) 
        tokenList=removestopwords(tokenList)
        tokenList=repeatreplace(tokenList)
        tokenList=commentcorrect(tokenList)
#         tokenList=commentStemmer(tokenList)       
        countbad =encountebadwords(tokenList)
        Count.append(countbad)
        print i
        X[i]=tokenList 
    return y,X,Count,


# In[312]:

import time
def readtestdata(testfile):
    X_test=[]
    with io.open(testfile, encoding = 'utf-8') as f:
        for line in f:
            X_test.append(line[:])
    return X_test

def processingTest(test_fname):   
    X_test=readtestdata(test_fname)
    Count=[]
    t0 = time.time()
    for i in range(0,len(X_test)):
        line=removehtmltags(X_test[i])
        line=removebadtoken(line)
        line=commentnormalizer(line)
        tokenList=Tokenization(line) 
        tokenList=removestopwords(tokenList)
        tokenList=repeatreplace(tokenList)
        tokenList=commentcorrect(tokenList)
#         tokenList=commentStemmer(tokenList)
        countbad =encountebadwords(tokenList)
        Count.append(countbad)
        X_test[i]=tokenList
    print time.time()-t0
    return X_test,Count    

