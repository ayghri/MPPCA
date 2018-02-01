import numpy as np
import utils



class MPPCA(object):
    
    def __init__(self,num_mixtures,latent_dim,niter):
        
        self.num_mixtures = num_mixtures
        self.latent_dim = latent_dim
        self.niter = niter
    
    def initialize(self):
        self.pi, self.mu, self.W, self.sigma2 = utils.Kmeans(x)
    
    def fit(self,data):
        
        self.n_samples,self.dim = data.shape
        pi, mu, W, sigma2, niter = utils.Kmeans#

        p = len(sigma2)
   
        sigma2hist = np.zeros((p, niter))
        M = np.zeros((p, self.latent_dim, self.latent_dim))
        Minv = np.zeros((p, self.latent_dim, self.latent_dim))
        Cinv = np.zeros((p, self.dim, self.dim))
        logR = np.zeros((self.n_samples, p))
        R = np.zeros((self.n_samples, p))
       
    
        L = np.zeros(niter)
        for i in range(niter):
            print('.', end='')
            for c in range(p):
                sigma2hist[c, i] = sigma2[c]
    
                # M
                M[c, :, :] = sigma2[c]*np.eye(self.latent_dim) + np.dot(W[c, :, :].T, W[c, :, :])
                Minv[c, :, :] = np.linalg.inv(M[c, :, :])
    
                # Cinv
                Cinv[c, :, :] = (np.eye(self.dim)- np.dot(np.dot(W[c, :, :], Minv[c, :, :]), W[c, :, :].T)) / sigma2[c]
    
                # R_ni
                deviation_from_center = data - mu[c, :]
                logR[:, c] = ( np.log(pi[c])
                    + 0.5*np.log(
                        np.linalg.det(
                            np.eye(self.dim) - np.dot(np.dot(W[c, :, :], Minv[c, :, :]), W[c, :, :].T)
                        )
                    )
                    - 0.5*self.dim*np.log(sigma2[c])
                    - 0.5*(deviation_from_center * np.dot(deviation_from_center, Cinv[c, :, :].T)).sum(1)
                    )
    
            myMax = logR.max(axis=1).reshape((self.n_samples, 1))
            L[i] = (
                (myMax.ravel() + np.log(np.exp(logR - myMax).sum(axis=1))).sum(axis=0)
                - self.n_samples*self.dim*np.log(2*3.141593)/2.
                )
    
            logR = logR - myMax - np.reshape(np.log(np.exp(logR - myMax).sum(axis=1)), (self.n_samples, 1))
    
            myMax = logR.max(axis=0)
            logpi = myMax + np.log(np.exp(logR - myMax).sum(axis=0)) - np.log(self.n_samples)
            logpi = logpi.T
            pi = np.exp(logpi)
            R = np.exp(logR)
            for c in range(self.num_clusters):
                mu[c, :] = (R[:, c].reshape((self.n_samples, 1)) * data).sum(axis=0) / R[:, c].sum()
                deviation_from_center = data - mu[c, :].reshape((1, self.dim))
    
                SW = ( (1/(pi[c]*self.n_samples))
                    * np.dot((R[:, c].reshape((self.n_samples, 1)) * deviation_from_center).T,
                        np.dot(deviation_from_center, W[c, :, :]))
                    )
    
                Wnew = np.dot(SW, np.linalg.inv(sigma2[c]*np.eye(self.latent_dim) + np.dot(np.dot(Minv[c, :, :], W[c, :, :].T), SW)))
    
                sigma2[c] = (1/self.dim) * (
                    (R[:, c].reshape(self.n_samples, 1) * np.power(deviation_from_center, 2)).sum()
                    /
                    (self.n_samples*pi[c])
                    -
                    np.trace(np.dot(np.dot(SW, Minv[c, :, :]), Wnew.T))
                    )
    
                W[c, :, :] = Wnew
    
        return pi, mu, W, sigma2, R, L, sigma2hist


    def MPPCA_predict(self,X, pi, mu, W, sigma2):
        self.n_samples, self.dim = X.shape
        p = len(sigma2)
        _, self.latent_dim = W[0].shape
    
        M = np.zeros((p, self.latent_dim, self.latent_dim))
        Minv = np.zeros((p, self.latent_dim, self.latent_dim))
        Cinv = np.zeros((p, self.dim, self.dim))
        logR = np.zeros((self.n_samples, p))
        R = np.zeros((self.n_samples, p))
    
        for c in range(p):
            # M
            M[c, :, :] = sigma2[c] * np.eye(self.latent_dim) + np.dot(W[c, :, :].T, W[c, :, :])
            Minv[c, :, :] = np.linalg.inv(M[c, :, :])
    
            # Cinv
            Cinv[c, :, :] = (np.eye(self.dim)
                - np.dot(np.dot(W[c, :, :], Minv[c, :, :]), W[c, :, :].T)
                ) / sigma2[c]
    
            # R_ni
            deviation_from_center = X - mu[c, :]
            logR[:, c] = ( np.log(pi[c])
                + 0.5*np.log(
                    np.linalg.det(
                        np.eye(self.dim) - np.dot(np.dot(W[c, :, :], Minv[c, :, :]), W[c, :, :].T)
                    )
                )
                - 0.5*self.dim*np.log(sigma2[c])
                - 0.5*(deviation_from_center * np.dot(deviation_from_center, Cinv[c, :, :].T)).sum(1)
                )
    
        myMax = logR.max(axis=1).reshape((self.n_samples, 1))
        logR = logR - myMax - np.reshape(np.log(np.exp(logR - myMax).sum(axis=1)), (self.n_samples, 1))
        R = np.exp(logR)
    
        return R
    
class PPCA(object):
    
    def __init__(self,latent_dim):
        
        self.latent_dim = latent_dim
    
