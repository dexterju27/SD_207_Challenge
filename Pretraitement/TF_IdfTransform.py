
# coding: utf-8

# In[44]:

import numpy as np
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


# In[41]:

R.fit_transform(a).toarray()


# In[42]:

R=TfidfTransformer()


# In[43]:

a=[[0,0,1],[1,1,0],[0,2,0]]
R.fit(a)
R.transform(a).toarray()

