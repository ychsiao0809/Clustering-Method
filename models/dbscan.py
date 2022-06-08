from sklearn import cluster, datasets, metrics
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import random

def euclidean(point, data):
    return np.sqrt(np.sum((point - data)**2))

class myDBSCAN:
    def __init__(self, eps=0.5, min_samples=2):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = []

    def check_core_point(self, X, idx_cur):
        # get points from given index
        x_cur = X[idx_cur]
        
        # check available points within radius
        idx_neigh = [] 
        for i, x in enumerate(X):
            if euclidean(x, x_cur) <= self.eps and (i != idx_cur):
                idx_neigh.append(i)
        
        # check how many points are present within radius
        if len(idx_neigh) >= self.min_samples:
            return (idx_neigh , True, False, False)    
        elif (len(idx_neigh) < self.min_samples) and len(idx_neigh) > 0:
            return (idx_neigh , False, True, False)    
        elif len(idx_neigh) == 0:
            return (idx_neigh , False, False, True)


    def fit(self, X):
        C = 1
        current_stack = set()
        unvisited = list(range(len(X)))
        clusters = []
        
        while (len(unvisited) != 0): #run until all points have been visited
            #identifier for first point of a cluster
            first_point = True
            #choose a random unvisited point
            current_stack.add(random.choice(unvisited))
        
            # Run until a cluster is complete
            while len(current_stack) != 0:
                curr_idx = current_stack.pop()
                
                #check if point is core, neighbour or border
                neigh_indexes, iscore, isborder, isnoise = self.check_core_point(X, curr_idx)
                if (isborder and first_point):
                    # first border point assign as noise
                    clusters.append((curr_idx, 0))
                    for neigh_idx in neigh_indexes:
                        if neigh_idx not in [_idx for _idx, _class in clusters]:
                            clusters.append((neigh_idx, 0))
                    unvisited.remove(curr_idx)
                    unvisited = [e for e in unvisited if e not in neigh_indexes]
                    continue
                    
                unvisited.remove(curr_idx)
                # look up unvisited point
                neigh_indexes = set(neigh_indexes) & set(unvisited)
                
                if iscore:
                    first_point = False                
                    clusters.append((curr_idx, C))
                    current_stack.update(neigh_indexes)
                    continue
                elif isborder:
                    clusters.append((curr_idx, C))
                    continue
                elif isnoise:
                    clusters.append((curr_idx, 0))                
                    continue
                    
            # Increment cluster number
            if not first_point:
                C+=1
        
        clusters.sort(key=lambda x:x[0])
        
        self.labels_ = [clas for (clus, clas) in clusters]
        return self.labels_