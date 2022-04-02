import numpy as np

def PCA(X,num_dim=None):
    X_pca, num_dim = X, len(X[0]) # placeholder

    # finding the projection matrix that maximize the variance (Hint: for eigen computation, use numpy.eigh instead of numpy.eig)
    #1 center the data
    mu = X.mean(axis=0)
    X = X-mu

    #2 Find the covariance:
    sigma = np.cov(np.transpose(X)) #I THINK that this should be a dxd matrix, not NXN.

    #3 find the eigenvalues and eivenvectors:
    lamb, w = np.linalg.eigh(sigma)

    #reverse order:
    #https://stackoverflow.com/questions/8092920/sort-eigenvalues-and-associated-eigenvectors-after-using-numpy-linalg-eig-in-pyt
    idx = lamb.argsort()[::-1]
    lamb = lamb[idx]
    w = w[:,idx]

    #!!!!Must change eigenvectors from columns to rows!!!
    w = w.T

    #4 find how many eigenvalues we will need to have 95% variance,
    #Using slide 23 of dimensionality reduction 1

    lambcounter = 0
    lambs = lamb.sum()
    #print(lambs)
    for i in range(len(lamb)):
        lambcounter += 1
        var = lamb[:lambcounter].sum()/lambs

        # print("Count: ", lambcounter)
        # print("variance: ", var)

        if var >= .95:
            break

    # select the reduced dimensions that keep >95% of the variance
    num_dim = lambcounter
    W = w[:lambcounter].T

    # project the high-dimensional data to low-dimensional one
    X_pca = X.dot(W)

    ###Testing: ###
    # print("X dim: ", X.shape)
    # print("Sigma dim: ", sigma.shape)
    # print("Lambda dim: ", lamb.shape)
    # print("w dim: ", w.shape)
    # print("W dim: ", W.shape)
    # print(X_pca.shape)

    return X_pca, num_dim