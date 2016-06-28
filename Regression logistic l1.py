# on calcule 
def proximal_l1(w, lam,rho):
    result = []
    lam = rho * lam
    for wi in w:
        if wi > lam:
            result.append(wi - lam)
        elif np.abs(wi) <= lam:
            result.append(0)
        else :
            result.append(wi + lam)
    return result

def function_f(w, X, Y, rho):
    X = np.c_[X, np.ones(X.shape[0])] # Add one cols
    sum_f = 0
    k = 0
    for xi, yi in zip(X, y):
        exp = np.exp(-1. * yi * xi.T.dot(w))
        sum_f = sum_f + np.log1p(exp) 
    sum_f = sum_f / y.shape[0]
    return sum_f 

# get value w (30,0) 
def grandient_f(w, X, Y, rho):
    X = np.c_[X, np.ones(X.shape[0])] # Add one cols
    sum_gradient = np.zeros(X.shape[1])
    for xi, yi in zip(X, Y):
        exp = np.exp(-1. * yi * xi.T.dot(w))
        sum_gradient = sum_gradient + xi* (-1.* yi) * (1. - (1. / (1. + exp)))   
    sum_gradient = sum_gradient / y.shape[0]
    return sum_gradient 
# return value of g is (31,) h is (31, 31)

# question 2.2
def optimisation_proximal(X,y,function_f, grandient_f, proximal_l1, rho,x0, rtol, maxloop, lam, beta):
    xk = x0
    fk_old = np.inf
    k = 0

    fk, grad_fk = function_f(xk, X, y, rho), grandient_f(xk,X,y,rho)
    while True :
        k = k + 1
        grad_fk = grandient_f(xk,X,y,rho)
        while True:  #lam change
            xk_grad = xk - lam * grad_fk
            prx = proximal_l1(xk_grad, lam, rho)
            Gt = (xk - prx) / lam
            lhand = function_f(xk - lam * Gt, X,y,rho)
            rhand = fk - lam * grad_fk.dot(Gt) + (0.5 * lam) * Gt.dot(Gt)
            if lhand <= rhand:
                break
            else:
                lam *= beta

        xk -= lam * Gt
        fk_old = fk
        fk, grad_fk = function_f(xk,X,y,rho), grandient_f(xk,X,y,rho)
        if np.linalg.norm(lam * np.array(proximal_l1(xk,lam,rho)) - np.zeros(xk.shape[0])) < rtol or k > maxloop: # stop condition
            print fk
            return xk
        
        
        
class Regression_logstic():
    def __init__(self,C = 1. ,rtol = 10**-10, maxloop=1000, lam=0.5, beta = 0.5):
        # initial
        self.rtol = rtol
        self.maxloop = maxloop
        self.lam = lam
        self.beta = beta
        self.rho = 1. / C
        # first col is flag, second col is the evalue
    def get_params(self, deep=True):
        return "c: +" + str(self.C) + str(self.tol)
    def fit(self, X, y):
        self.rho = 1./ y.shape[0]
        self.w = np.zeros(features.shape[1] + 1)
        self.w = optimisation_proximal(X, y, function_f, grandient_f, proximal_l1, self.rho,self.w , self.rtol ,self.maxloop,  self.lam, self.beta)  
    def predict(self, X):
        X = np.c_[X, np.ones(X.shape[0])] # Add one cols
        result = X.dot(self.w)
        result[result >0 ] = 1
        result[result < 0] = 0
        return result.astype('int').tolist()
    
    def score(self, X, y):
        return np.mean(self.predict(X) == y)