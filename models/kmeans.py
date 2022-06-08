import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from random import uniform

class myKmeans:
    def __init__(self, n_clusters, random_state=0, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.labels_ = []
    
    def fit(self, X_train):
        # Initiate centroids
        min_, max_ = np.min(X_train, axis=0), np.max(X_train, axis=0)
        self.centroids = [uniform(min_, max_) for _ in range(self.n_clusters)]
        
        iteration = 0
        prev_centroids = None
        while np.not_equal(self.centroids, prev_centroids).any() and iteration < self.max_iter:
            # Sort each datapoint, assigning to nearest centroid
            sorted_points = [[] for _ in range(self.n_clusters)]
            for x in X_train:
                dists = euclidean(x, self.centroids)
                centroid_idx = np.argmin(dists)
                sorted_points[centroid_idx].append(x)
            # Push current centroids to previous, reassign centroids as mean of the points belonging to them
            prev_centroids = self.centroids
            self.centroids = [np.mean(cluster, axis=0) for cluster in sorted_points]
            for i, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():  # Catch any np.nans, resulting from a centroid having no points
                    self.centroids[i] = prev_centroids[i]
            iteration += 1

        self.labels_ = self.predict(X_train)

    def predict(self, X):
        centroid_idxs = []
        for x in X:
            dists = euclidean(x, self.centroids)
            centroid_idx = np.argmin(dists)
            centroid_idxs.append(centroid_idx)
        return centroid_idxs

def euclidean(point, data):
    dists = []
    for d in data:
        dists.append(np.sqrt(np.sum((point - data)**2, axis=1)))
    return dists