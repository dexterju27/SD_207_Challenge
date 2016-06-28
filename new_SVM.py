import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import unidecode
import csv
import io
import nltk
import string
# definition

import numpy as np
X = []
y = []
with open('train.csv') as f:
    for line in f:
        y.append(int(line[0]))
        X.append(line[5:-6])
y = np.array(y)

y[y == 0] = -1.


def loadDataSet(test_fileName, train_fileName):
    test = []
    train = []
    with io.open(test_fileName, encoding = 'utf-8') as f:
        for line in f:
            lineArr = line.strip().split('\t')
            test.append(lineArr)
    with io.open(train_fileName, encoding = 'utf-8') as f:
        for line in f:
            lineArr = line.strip().split('\t')
            train.append(lineArr)
    return test,train

test, train = loadDataSet('X_test.txt', 'X_train.txt')


def words2Vectors(wordList, data):
    vec = [0] * len(wordList)
    for word in data:
        if word in wordList:
            vec[wordList.index(word)] = vec[wordList.index(word)] + 1 # + (word in badList)
            # count bad words twice
    return vec

def createWordList(X, minchar):
    wordContent = []
    wordCount = dict()
    for line in X:
        for word in line:
            if len(word) > minchar:
                if word in wordCount.keys():
                    wordCount[word] = wordCount[word] + 1 # + (word in badList)
                else:
                    wordCount[word] = 1 # + (word in badList)
            else:
                continue
    return wordCount.keys(), wordCount



wordlist, count = createWordList(train, 2)
vectors_X_train = []
vectors_X_test = []
for line in train:
    vectors_X_train.append(words2Vectors(wordlist, line))
for line in test:
    vectors_X_test.append(words2Vectors(wordlist, line))

from numpy import *

def cakcWs(alphas, X, y ):
    X = np.mat(X)
    y = np.mat(y).T
    n, t = X.shape
    print  X.shape
    w = np.zeros((t,1))
    for i in np.arange(n):
        w += np.multiply(alphas[i]*y[i], X[i,:].T)
    return w




def selectJrand(i,m):
    j=i #we want to select any J not equal to i
    while (j==i):
        j = int(random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

class optStruct:
    def __init__(self,dataMatIn, classLabels, C, toler):  # Initialize the structure with the parameters
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = mat(zeros((self.m,2))) #first column is valid flag

def calcEk(oS, k):
    fXk = float(multiply(oS.alphas,oS.labelMat).T*(oS.X*oS.X[k,:].T)) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def selectJ(i, oS, Ei):         #this is the second choice -heurstic, and calcs Ej
    maxK = -1; maxDeltaE = 0; Ej = 0
    oS.eCache[i] = [1,Ei]  #set valid #choose the alpha that gives the maximum delta E
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:   #loop through valid Ecache values and find the one that maximizes delta E
            if k == i: continue #don't calc for i, waste of time
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej
    else:   #in this case (first time around) we don't have any valid eCache values
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

def updateEk(oS, k):#after any alpha has changed update the new value in the cache
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]

def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i, oS, Ei) #this has been changed from selectJrand
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H: return 0
        eta = 2.0 * oS.X[i,:]*oS.X[j,:].T - oS.X[i,:]*oS.X[i,:].T - oS.X[j,:]*oS.X[j,:].T
        if eta >= 0: return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS, j) #added this for the Ecache
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])#update i by the same amount as j
        updateEk(oS, i) #added this for the Ecache                    #the update is in the oppostie direction
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[i,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[i,:]*oS.X[j,:].T
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[j,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[j,:]*oS.X[j,:].T
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1
    else: return 0

def smoP(dataMatIn, classLabels, C, toler, maxIter):    #full Platt SMO
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler)
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:   #go over all
            for i in range(oS.m):
                alphaPairsChanged += innerL(i,oS)
            iter += 1
        else:#go over non-bound (railed) alphas
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
            iter += 1
        if entireSet: entireSet = False #toggle entire set loop
        elif (alphaPairsChanged == 0): entireSet = True
        print "iteration number: %d" % iter
        sys.stdout.flush()
    return oS.b,oS.alphas

class SVM():
    def __init__(self, C = 1., tol = 0.00001, maxIter= 10):
        # initial
        self.C = C
        self.tol = tol
        self.maxIter = maxIter
        # first col is flag, second col is the evalue
    def get_params(self, deep=True):
        return "c: +" + str(self.C) + str(self.tol)
    def fit(self, X, y):
        self.b, self.alphas = smoP(X, y, self.C, self.tol, self.maxIter)
        print "fit completed!"
        sys.stdout.flush()
        self.w = cakcWs(self.alphas, X,  y)

    def predict(self, X):
        print (np.mat(X)*np.mat(self.w)).shape
#         print self.w

        result = np.mat(X)*np.mat(self.w) + self.b
        print result
        result[result <= 0] = 0
        result[result > 0]  = 1
        return result.A1.astype('int')

    def score(self, X, y):
        return np.mean(self.predict(X) == y)



from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.feature_selection import chi2,SelectKBest
from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=True)
clf = SVM(C = 0.1)
select = SelectKBest(chi2, 2450)
pipe = Pipeline(steps=[('chi2', select), ("tf-idf",tf_transformer )])
X_fit = pipe.fit_transform(vectors_X_train, y)
test_fit = pipe.transform(vectors_X_test)
print "start fitting"
clf.fit(X_fit.toarray(), y)
y_pre = clf.predict(test_fit.toarray())
np.savetxt('y_pred.txt', y_pre, fmt='%s')
