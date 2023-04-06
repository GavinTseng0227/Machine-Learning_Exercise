import numpy as np


def eval_adaBoost_simpleClassifier(X, alphaK, para):
    # INPUT:
    # para	: parameters of simple classifier (K x 2) - for each round
    #           : dimension 1 is j
    #           : dimension 2 is theta
    # alphaK    : classifier voting weights (K x 1)
    # X         : test data points (numSamples x numDim)
    #
    # OUTPUT:
    # classLabels: labels for data points (numSamples x 1)
    # result     : weighted sum of all the K classifier (numSamples x 1)

    #####Insert your code here for subtask 1c#####
    # number of classifiers
    K = para.shape[0]
    # number of test points
    N = X.shape[0]
    # prediction for each test point
    result = np.zeros(N)

    for k in range(K):
        # Initialize temporary labels for given j and theta
        cY = np.ones(N) * (-1)
        # Classify
        cY[X[:, int(para[k, 0] - 1)] > para[k, 1]] = 1

        # update results with weighted prediction
        result += alphaK[k] * cY

    # class-predictions for each test point
    classLabels = np.sign(result)

    return classLabels, result
