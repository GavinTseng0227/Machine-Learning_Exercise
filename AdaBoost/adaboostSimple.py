import numpy as np
from numpy.random import choice

from simpleClassifier import simpleClassifier
def adaboostSimple(X, Y, K, nSamples):
    # Adaboost with decision stump classifier as weak classifier
    #
    # INPUT:
    # X         : training examples (numSamples x numDim)
    # Y         : training lables (numSamples x 1)
    # K         : number of weak classifiers to select (scalar) 
    #             (the _maximal_ iteration count - possibly abort earlier
    #              when error is zero)
    # nSamples  : number of training examples which are selected in each round (scalar)
    #             The sampling needs to be weighted!
    #             Hint - look at the function 'choice' in package numpy.random
    #
    # OUTPUT:
    # alphaK 	: voting weights (K x 1) - for each round
    # para		: parameters of simple classifier (K x 2) - for each round
    #           : dimension 1 is j
    #           : dimension 2 is theta

    #####Insert your code here for subtask 1c#####
    # total number of training samples
    N, _ = X.shape

    # Initialize the classifier models
    j = np.zeros(K)
    theta = np.zeros(K)

    # voting weight for each classifier
    alpha = np.zeros(K)
    # uniform initialization of sample-weights
    w = (np.ones(N) / N).reshape(N, 1)

    # Iterate over all classifiers
    for k in range(K):

        # Sample data with weights
        # ravel(): flatten form high dim. to 1D
        index = choice(N, nSamples, True, w.ravel())

        X_sampled = X[index, :]
        Y_sampled = Y[index]

        # Train the weak classifier C_k
        j[k], theta[k] = simpleClassifier(X_sampled, Y_sampled)

        # placeholder for class predictions
        cY = (np.ones(N) * (-1)).reshape(N, 1)
        # classify
        cY[X[:, int(j[k] - 1)] > theta[k]] = 1

        # Calculate weighted error for given classifier
        temp = np.where([Y[i] != cY[i] for i in range(N)], 1, 0).reshape(N, 1)
        ek = np.sum(w * temp)

        # If the error is zero, the data set is correct classified - break the loop
        if ek < 1.0e-01:
            alpha[k] = 1
            break

        # Compute the voting weight for the weak classifier alpha_k
        alpha[k] = 0.5 * np.log((1 - ek) / ek)

        # Update the weights
        w = w * np.exp((-alpha[k] * (Y * cY)))
        w = w / sum(w)

    alphaK = alpha
    para = np.stack((j, theta), axis=1)

    return alphaK, para
