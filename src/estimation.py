######################################################
#                    LIBRAIRIES                      #
######################################################
import numpy as np

######################################################
#                    FUNCTIONS                       #
######################################################

def phase_only(X):
    """ A function that computes phase only Estimator for covariance matrix estimation
        Inputs:
            * X = a matrix of size p*N with each observation along column dimension

        Outputs:
            * PO estimator 
    """
    n = X.shape[1]
    X = X/np.sqrt(abs(X)**2)
    return(np.dot(X,X.conj().T)/n)