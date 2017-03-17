"""
	Grouped Fixed-Effects (GFE) from Bonhomme & Manresa (Econometrica 2015)
"""
import random
import numpy as np

from scipy.sparse import lil_matrix
from scipy.optimize import minimize

def gfe_alg_1_default_init(Y, X, XHet, N, T, G, ncov, nhetcov):
    # Initialization as descibed in the online appendix of Bonhomme & Manresa (2015)
    beta = 5*(np.random.rand(ncov)-0.5)
    betaHet = 5*(np.random.rand(G,nhetcov)-0.5)
    alpha = np.zeros((G,T))
    
    seq = random.sample(range(N), G)
    for g in range(G):
        for t in range(T):
            alpha[g,t] = Y[seq[g]*T+t] - X[seq[g]*T+t,:].dot(beta) - XHet[seq[g]*T+t,:].dot(betaHet[g,:])   
       
    return [beta, betaHet, alpha]

def gfe_alg_1 (Y, X, XHet, N, T, G, init = gfe_alg_1_default_init):
    Y = np.squeeze(np.array(Y, dtype = np.float64))   # squeezing Y here makes the following code more readable
                
    if X is None:
        X = np.zeros((N*T,0))        
    else:
        X = np.array(X, dtype = np.float64)

    if XHet is None:
        XHet = np.zeros((N*T,0)) 
    else:
        XHet = np.array(XHet, dtype = np.float64) 
        
    ncov = X.shape[1] 
    nhetcov = XHet.shape[1]
    
    # indices that split a single vector x into different parameter types (used by minimization)
    i1 = ncov
    i2 = ncov + nhetcov*G
    i3 = ncov + nhetcov*G + G*T
        
    # 1. Initialization 
    [beta, betaHet, alpha] = init(Y, X, XHet, N, T, G, ncov, nhetcov)
            
    # helper to compute the optimal grouping
    def grouping():
        # y_it - x_it*beta
        R = Y - X.dot(beta)
                        
        # subtract heterogeneous coefficients, alpha_g_t, elementwise power by 2, and sum rows
        Z = np.zeros((N,G))
        
        for g in range(0,G):    
            R_g = R - XHet.dot(betaHet[g,:])
            R_g = R_g.reshape((N,T))
            R_g = R_g - alpha[g,:]
            R_g = pow(R_g, 2)
            Z[:,g] = np.sum(R_g, axis=1)

        return np.argmin(Z, axis=1)
    
    # 2. Compute the optimal grouping
    ass = grouping()

    # helper to minimize the objective 
    def optimize():        
        # create a sparse matrix that translates group effect -> observation
        K_lil = lil_matrix((N*T,G*T))

        for i in range(0,N):
            for t in range(0,T):
                K_lil[i*T+t, ass[i]*T+t] = 1

        K_csr = K_lil.tocsr()
        K_csc = K_lil.tocsc() 
        
        # create a matrix that codes heterogeneous coefficients by group
        L_Het = np.zeros((N*T,G*nhetcov))  
        
        for i in range(0,N):
            for j in range(0,nhetcov):
                for t in range(0,T):
                    L_Het[i*T+t, j*G+ass[i]] = XHet[i*T+t,j]
            
        # compute the optimal parameters
        def f(x):
            # (y_it - x_it*beta - alpha_gi_t)^2 and sum it up
            return pow(Y - X.dot(x[0:i1]) - L_Het.dot(x[i1:i2]) - K_csr.dot(x[i2:]), 2.).sum()

        def grad(x):
            # pre-compute (y_it - x_it*beta - alpha_gi_t)
            m = Y - X.dot(x[0:i1]) - L_Het.dot(x[i1:i2]) - K_csr.dot(x[i2:])

            grad = np.zeros(i3)

            for i in range(0, i1):
                grad[i] = -2*np.inner(X[:,i], m)
                
            for i in range(i1, i2):
                grad[i] = -2*np.inner(L_Het[:,i-i1], m)

            h = -2*m*K_csc    
            for i in range(i2, i3):
                grad[i] = h[i-i2]

            return grad
        
        # it is crucial that we corretly code [beta, betaHet, alpha] -> x -> [beta, betaHet, alpha]
        res = minimize(f, np.append(np.append(beta, betaHet.transpose()), alpha), jac = grad, method = 'BFGS')
        
        return [res, res.x[0:i1], res.x[i1:i2].reshape((nhetcov,G)).transpose(), res.x[i2:].reshape((G,T))]

    # 3. Minimize the objective
    [res, beta, betaHet, alpha] = optimize()

    # 4. Repeat steps 2. and 3. until numerical convergence
    while(True):
        ass2 = grouping() 
        
        if (ass == ass2).all():
            break
            
        ass = ass2
        
        [res, beta, betaHet, alpha] = optimize()
                
    return [beta, betaHet, alpha, res.fun, ass]





def gfe(Y, X, XHet, N, T, G, init = gfe_alg_1_default_init, maxIter = 1000, hit = 5, bootstrap = False, nbootstrapRepl = 10):
    [beta, betaHet, alpha, fun, ass] = gfe_alg_1(Y, X, XHet, N, T, G, init = init)
    nconsecutive = 1
    itercount = 1
    
    # run algorithm 1 until we found the same smallest value 5 consecutive times or we reach maxIter    
    while(True):
        [beta1, betaHet1, alpha1, fun1, ass1] = gfe_alg_1(Y, X, XHet, N, T, G, init = init)
        
        if abs(fun-fun1)/fun < 1e-6:   # another time
            print "Hit again!"
            nconsecutive = nconsecutive+1
            
            if nconsecutive >= hit:
                print "Hit %d times, done!" % hit
                break 
        elif fun1 < fun:   # smaller value found
            [beta, betaHet, alpha, fun, ass] = [beta1, betaHet1, alpha1, fun1, ass1]
            nconsecutive = 1
            print fun
           
        # check maxIter
        itercount = itercount+1
        
        if itercount == maxIter:
            print "Maximum number of iterations reached!"
            break
            
    # bootstrap standard errors
    if bootstrap:
        # initialize the bootstrap replications with perturbations of the original estimates
        #def bootstrap_init(Y, X, XHet, N, T, G, ncov, nhetcov):            
        #    return [np.random.normal(beta, abs(beta)),  np.random.normal(betaHet, abs(betaHet)), np.random.normal(alpha, abs(alpha))]        

        print "Bootstrapping ..."
        #gfe_bootstrap(Y, X, XHet, N, T, G, beta, betaHet, alpha, init = init, maxIter = 1000, hit = hit, nRepl = nbootstrapRepl)


        return [beta, betaHet, alpha, fun, ass, beta_bootstrap, betaHet_bootstrap]   # bootstrap
            
    return [beta, betaHet, alpha, fun, ass]   # no bootstrap
            



def gfe_bootstrap(Y, X, XHet, N, T, G, init = gfe_alg_1_default_init, maxIter = 1000, hit = 5, nRepl = 10):
    """ Perform nRepl bootstrap replications.

        Returns: The estimated coefficients for each replication.
    """
    beta_repl = []
    betaHet_repl = []
    ass_repl = []

    print "Bootstrapping ..."
    Y = np.squeeze(np.array(Y, dtype = np.float64))
        
    if not(X is None):
        X = np.array(X, dtype = np.float64)
          
    if not(XHet is None):
        XHet = np.array(XHet, dtype = np.float64)
        
    for i in range(nRepl):
        # draw a subsample (with replacement)
        sample = []
            
        for j in np.random.randint(0, N, N):
            for t in range(T):
                sample.append(j*T+t)
            
        # create dataset with the subsample and replicate       
        X_sub = None
        if not(X is None):
            X_sub = X[sample,:]
                
        XHet_sub = None
        if not(XHet is None):
            XHet_sub = XHet[sample,:]
            
        beta_sub, betaHet_sub, alpha_sub, fun_sub, ass_sub = gfe(Y[sample], X_sub, XHet_sub, N, T, G, init = init, maxIter = maxIter, hit = hit)
            
        # groups assigned by the replication, or -1 if not present
        ass = []

        for j in range(N):
            if not(j*T in sample):
                ass.append(-1)
            else:
                index = next(i for i, v in enumerate(sample) if v == j*T)
                ass.append(ass_sub[index/T])

        # store result
        beta_repl.append(beta_sub)
        betaHet_repl.append(betaHet_sub)
        ass_repl.append(ass)

    # return what the bootstrap replications assigned
    return beta_repl, betaHet_repl, ass_repl




