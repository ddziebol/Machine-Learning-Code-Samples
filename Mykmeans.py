# import libraries
import numpy as np


class Kmeans:
    def __init__(self, k=8):  # k is number of clusters
        self.num_cluster = k
        self.center = None
        self.error_history = []

    def run_kmeans(self, X, y):
        # initialize the centers of clusters as a set of pre-selected samples
        init_idx = [1, 200, 500, 1000, 1001, 1500, 2000, 2005]  # indices for the samples

        dim = (len(init_idx), len(X[0]))
        self.center = np.zeros(dim)
        for i in range(len(init_idx)):
            self.center[i] = X[init_idx[i]]

        num_iter = 0  # number of iterations for convergence

        # initialize cluster assignment
        prev_cluster_assignment = np.zeros([len(X), ]).astype('int')
        cluster_assignment = np.zeros([len(X), ]).astype('int')
        is_converged = False

        # iteratively update the centers of clusters till convergence
        while not is_converged:
            # iterate through the samples and compute their cluster assignment (E step)
            ##Per the gradescope read out, I am making these variables to store classes:
            dim = self.center.shape
            centers = np.zeros(self.center.shape)
            clustercount = np.zeros(dim[0]).astype('int')
            for i in range(len(X)):
                # use euclidean distance to measure the distance between sample and cluster centers
                edistances = []
                # Make a list of distances between X[i] and cluster centers:
                for j in range(len(self.center)):
                    edistances.append(np.linalg.norm(X[i] - self.center[j]))

                # determine the cluster assignment by selecting the cluster whose center is closest to the sample
                clusternumber = np.argmin(edistances)
                # update cluster assignment:
                cluster_assignment[i] = clusternumber
                # update class count:
                clustercount[clusternumber] = clustercount[clusternumber] + 1
                # update sum of all X in class:
                centers[clusternumber] = centers[clusternumber] + X[i]
                pass

            # print(cluster_assignment[:10])
            # print(clustercount)
            # print(Ncounts)
            # print(centers[0])
            # update the centers based on cluster assignment (M step)

            for i in range(len(self.center)):
                self.center[i] = np.divide(centers[i], clustercount[i])

            # compute the reconstruction error for the current iteration
            cur_error = self.compute_error(X, cluster_assignment)
            self.error_history.append(cur_error)

            # reach convergence if the assignment does not change anymore
            is_converged = True if (cluster_assignment == prev_cluster_assignment).sum() == len(X) else False
            prev_cluster_assignment = np.copy(cluster_assignment)
            num_iter += 1

        # compute the information entropy for different clusters
        Kclusters, Ncounts = np.unique(cluster_assignment, return_counts=True)
        # print(Kclusters)
        # print(Ncounts)

        PXKC = np.zeros((len(Kclusters), 4))
        # Count each class in each cluster:
        for i in range(len(X)):
            PXKC[cluster_assignment[i]][0] += 1
            PXKC[cluster_assignment[i]][(np.where([0, 8, 9] == y[i])[0]) + 1] += 1
            pass
        print(PXKC)

        entropy = 0
        for i in range(len(PXKC)):
            Kentropy = 0
            for j in range(3):
                PX = PXKC[i][j + 1] / PXKC[i][0]
                if PX == 0:
                    log2PX = 0
                else:
                    log2PX = np.log2([PX])
                Kentropy += PX * log2PX  # Collects the entropy in one cluster for each class

                # print("numerator: ", PXKC[i][j+1])
                # print("Denominator: ", PXKC[i][0])
                # print("PX: ", PX)
                # print(log2PX)
                # print(PX*log2PX)

                pass
            entropy += Kentropy / 8  # each cluster entropy contributes 1/8 to total entropy

        entropy = entropy * -1  # make it positive per equation 1

        print("final error: ", self.error_history[-1])
        return num_iter, self.error_history, entropy

    def compute_error(self, X, cluster_assignment):
        # compute the reconstruction error for given cluster assignment and centers
        error = 0  # placeholder
        for i in range(len(X)):
            error = error + (np.linalg.norm(X[i] - self.center[cluster_assignment[i]]) ** 2)
        return error

    def params(self):
        return self.center
