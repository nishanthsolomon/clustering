from sklearn.cluster import KMeans
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
from collections import defaultdict
from datasetreader import *


def kmeans(data, K):
    X = np.array(data)
    cluster_numbers = [3, 5, 7]
    m = X.shape[0]  # number of training examples
    n = X.shape[1]  # number of features. Here n=2
    n_iter = 50
    Centroids = np.array([]).reshape(n, 0)
    cost = np.array([])

    # choosing random centriods
    for i in range(K):
        rand = rd.randint(0, m-1)
        Centroids = np.c_[Centroids, X[rand]]

    Output = {}
    for i in range(n_iter):
        EuclidianDistance = np.array([]).reshape(m, 0)
        # Calculatin Euclidian Distance with each cluster centriod
        for k in range(K):
            tempDist = np.sum((X-Centroids[:, k])**2, axis=1)
            EuclidianDistance = np.c_[EuclidianDistance, tempDist]
        C = np.argmin(EuclidianDistance, axis=1)+1
        # print(C)
        Y = {}
        for k in range(K):
            Y[k+1] = np.array([]).reshape(5, 0)
        for i in range(m):
            Y[C[i]] = np.c_[Y[C[i]], X[i]]
        for k in range(K):
            Y[k+1] = Y[k+1].T
        # Updating Centriods
        for k in range(K):
            Centroids[:, k] = np.mean(Y[k+1], axis=0)
        Output = Y
        # Calculating cost for Kmeans. Cost is obtained by sum of squares distance of each data point to its
        # corresponding cluster
        wcss = 0
        for k in range(K):
            wcss += np.sum((Output[k+1]-Centroids[:, k])**2)
        cost = np.append(cost, wcss)

    # Appling PCA with n_components 2.
    pcadf = pd.DataFrame(data=get_pca_data(), columns=[
                         'principal component 1', 'principal component 2'])
    # print(pcadf)
    O = {}
    for i in range(len(C)):
        cl = C[i]
        l = []
        l.append(pcadf.iloc[i, 0])
        l.append(pcadf.iloc[i, 1])
        if cl not in O:
            O[cl] = np.array(np.array(l))
        else:
            t = O[cl]
            O[cl] = np.vstack((t, np.array(l)))

    # Plotting Number of Iterations Vs Cost
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost")
    plt.plot(range(1, n_iter+1), cost)
    plt.title("Number of Iterations Vs Cost for KMeans K=" + str(K))
    plt.show()
    # Plotting clusters
    color = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'pink']
    labels = ['cluster1', 'cluster2', 'cluster3',
              'cluster4', 'cluster5', 'cluster6', 'cluster7']
    for k in range(K):
        plt.scatter(O[k+1][:, 0], O[k+1][:, 1], c=color[k], label=labels[k])
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.title("KMeans K="+str(K))
    plt.show()


if __name__ == "__main__":
    data = get_data()
    kmeans(data, 3)
    kmeans(data, 5)
    kmeans(data, 7)
