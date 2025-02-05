######################################################
#                    LIBRAIRIES                      #
######################################################
import numpy as np

######################################################
#                    FUNCTIONS                       #
######################################################

def MM_LS_IPL(Sigma_tilde,maxIter):
    # MM LS cost function
    # Data :
    #       * Sigma_tilde : covariance estimation (possibily after a regularization step)
    #       * maxIter : maximum iteration of the gradient descent
    # Output : w_theta : complex vector of phasis 
    p = Sigma_tilde.shape[0]
    M = np.multiply(abs(Sigma_tilde),Sigma_tilde)
    
    w = np.ones((p,1))
    
    for i in range (maxIter):
        tilde_w = M@w 
        w = np.exp(1j*np.angle(tilde_w))
    return w


def MM_LS_SlIPL(w_overlap_past, lamda, Sigma_tilde, iter_max_MM):
    """ Majorization Minimization problem for least square (LS) optimization problem

        Inputs :
            * w_new : phases vector of new data
            * w_past : phases vector of past data
            * new_past_Psi_tilde : Coherence vector between past and new data
            * new_past_Sigma_tilde : Covariance vector between past and new data
            * new_Psi_tilde : Coherence matrix of the new data
            * new_Sigma_tilde : Covariance matrix of the new data
            * iter_max_MM : number of MM algorithm iterations

        Outputs : 
            * new_w : vector of the phases
    """
    l = w_overlap_past.shape[0] # overlap between temporal stacks
    p = Sigma_tilde.shape[0] # size of a temporal stack
    k = p - l # stride (i.e the number of new images)
    zeros_vector = np.zeros((k, 1))
    new_w = np.ones((p,1))
    w_past = np.concatenate((w_overlap_past, zeros_vector))
    M = 4 * np.multiply(abs(Sigma_tilde), Sigma_tilde)
    for _ in range(iter_max_MM):
        tilde_w = M@new_w + 2 * lamda * w_past
        new_w = np.exp(1j*np.angle(tilde_w))
    return new_w