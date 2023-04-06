import numpy as np
from getLogLikelihood import getLogLikelihood


def MStep(gamma, X):
    # Maximization step of the EM Algorithm
    #
    # INPUT:
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.
    # X              : Input data (NxD matrix for N datapoints of dimension D).
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # means          : Mean for each gaussian (KxD).
    # weights        : Vector of weights of each gaussian (1xK).
    # covariances    : Covariance matrices for each component(DxDxK).

    #####Insert your code here for subtask 6c#####
    # Get the sizes
    n_training_samples, dim = X.shape
    K = gamma.shape[1]

    # Create matrices
    means = np.zeros((K, dim))
    covariances = np.zeros((dim, dim, K))

    # Compute the weights pi_new
    Nk = gamma.sum(axis=0)
    weights = Nk / n_training_samples

    # Compute the means mu_new
    means = np.divide(gamma.T.dot(X), Nk[:, np.newaxis])

    # Compute the covariance sigma_new
    for i in range(K):
        auxSigma = np.zeros((dim, dim))
        for j in range(n_training_samples):
            meansDiff = X[j] - means[i]
            auxSigma = auxSigma + gamma[j, i] * np.outer(meansDiff.T, meansDiff)
        covariances[:, :, i] = auxSigma / Nk[i]

    logLikelihood = getLogLikelihood(means, weights, covariances, X)
    return weights, means, covariances, logLikelihood
