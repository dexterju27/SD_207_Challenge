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

def removehtmltags(line):    
    p=re.compile(r"<.*?>")
    line = p.sub(' ', line)
    line = line.replace('&nbsp;',' ')   
    return line

def removebadtoken(line):
    rep = {'\\xa0': ' ', '\\xc2': ' ', '\\n': ' ', '\r': ''}
    rep = dict((re.escape(k), v) for k, v in rep.items())
    pattern = re.compile("|".join(rep.keys()))
    line = pattern.sub(lambda m: rep[re.escape(m.group(0))], line)
    return line   

def Tokenization(line):
    line=line.lower()
    tokenlist = wordpunct_tokenize(line)
    return tokenlist    

def removestopwords(tokenList):
    stopwordList=np.genfromtxt('stopwords.txt',dtype='str')
    filteredWords = [w for w in tokenList if not w in stopwordList]
    return filteredWords    


def commentnormalizer(line):
    line = "".join([ch for ch in line if ch not in string.punctuation])
    line = re.sub("\\d+(\\.\\d+)?", "NUM", line)
    return line

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

def processingtrain(filename):
    y,X_train=readdata(filename)
    Count=[]
    for i in range(0,len(X_train)):
        lines=removehtmltags(X_train[i])
        lines=removebadtoken(lines)
        lines=commentnormalizer(lines)
        lines=Tokenization(lines) 
        lines=removestopwords(lines)
        lines=repeatreplace(lines)
        lines=commentcorrect(lines)
        X_train[i]=lines
    return y,X_train


def readTestFile(test_fname):
    X_test = []
    with io.open(test_fname, encoding = 'utf-8') as f:
        for line in f:
            X_test.append(line[3:-6].lower())
        print len(X_test)
    return X_test

def processingTest(test_fname):
    test_content = readTestFile(test_fname)
    Count=[]
    for i in range(0,len(test_content)):
        lines=removehtmltags(test_content[i])
        lines=removebadtoken(lines)
        lines=commentnormalizer(lines)
        lines=Tokenization(lines) 
        lines=removestopwords(lines)
        lines=repeatreplace(lines)
        lines=commentcorrect(lines)
        test_content[i]=lines
    return test_content

y,X_train_text = processingtrain("train.csv")
X_test_text = processingTest("test.csv")

def words2Vectors(wordList, data):
    vec = [0] * len(wordList)
    for word in data:
        if word in wordList:
            vec[wordList.index(word)] = vec[wordList.index(word)] + 1 # + (word in badList) 
            # count bad words twice
    return vec

def createWordList(X):
    wordContent = []
    wordCount = dict()
    for line in X:
        for word in line:
            if word in wordCount.keys():
                wordCount[word] = wordCount[word] + 1 # + (word in badList)
            else:
                wordCount[word] = 1 # + (word in badList)

    return wordCount.keys(), wordCount
wordlist, count = createWordList(X_train_text)
vectors_X_train = words2Vectors(wordlist,X_train_text)
vectors_X_test = words2Vectors(wordlist,X_test_text)
print " the length of word list" + len(wordlist)

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.feature_selection import chi2,SelectKBest
from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=True)
clf = SVC(kernel='linear', gamma=2)
select = SelectKBest(chi2, 850)
pipe = Pipeline(steps=[('chi2', select), ("tf-idf",tf_transformer ),('SVM', clf)])
pipe.fit(vectors_X_train, y)
y_pre = pipe.predict(vectors_X_test)
np.save("y_pre.txt",y_pre)