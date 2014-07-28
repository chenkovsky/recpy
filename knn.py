__author__ = 'chenkovsky'

import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
import time
class UserBasedKNNRecommender:
    """
    class for user based knn recommender.
    when doing recommendation, it first select neighbors,
    and calculate the similarity between neighbors and current user.
    then $$E(X[user][item]) = \sum_{u in neighbors} X[u][item]*similarity[user][u]$$.
    return the highest ranked items.
    """
    def __init__(self, X, neighbors_size = 10, metric = 'euclidean', kdtree_leaf_size = 30):
        """
        item based knn recommender
        :param X: np.matrix, the element in matrix should >= 0. X[user][item] == 0 means the user hasn't voted the item.
        :param neighbors_size: use how many neighbors' voting to decide the user's vote
        :param metric: how to meter the distance between users.
        :param kdtree_leaf_size: leaf size of kdtree which is used to query neighbors.
        if the size of item set is very large, it may be a better choice
        """
        self.X = X
        self.neighbors_size = neighbors_size if len(self.X) > neighbors_size else len(self.X)
        self.kdt = KDTree(X, leaf_size=kdtree_leaf_size, metric= metric)
    def recommend(self, id, items_num = 10, except_voted = True):
        """
        recommend item for user
        :param id: user id, row num of the matrix.
        :param items_num: return at most how many items.
        :param except_voted: whether remove the items that user already has.
        :return: the list of the highest ranked items.
        """
        record = np.squeeze(np.asarray(self.X[id]))
        #print("-----record-----")
        #print(record)
        #print("-----end-----")
        #select possible items
        neighbors = self.kdt.query([record], k=self.neighbors_size,return_distance=False)
        #distances = neighbors[0]
        neighbor_ids = neighbors.flatten()
        #print("-----neighbor ids-----")
        #print(neighbor_ids)
        #print("-----end-----")
        #maybe we can skip select rest_items_ids
        #rest_item_ids = np.array([])
        #neighbor_matrix = []
        #for idx in range(0, len(distances)):
        #    np.union1d(rest_item_ids, np.nonzero(self.X[neighbor_ids[idx]]))
        #    neighbor_matrix.append(np.squeeze(np.asarray(self.X[neighbor_ids[idx]])))
        #rest_item_ids = np.setdiff1d(rest_item_ids, np.nonzero(record))

        #calculate item priority
        #1.calculate similarity
        m = self.X[neighbor_ids]
        #print("-----neighbors' matrix-----")
        #print(m)
        #print("-----end-----")
        #m = np.matrix(neighbor_matrix)
        def cosine_simi(u,v):
            print(u)
            #return None
            print(v)
            u = np.array(u)
            return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
        similarity = np.ma.apply_along_axis( cosine_simi, 1, m, record)
        #print("-----similarity-----")
        #print(similarity)
        #print("-----end-----")
        total_similarity = np.sum(similarity)
        #print("-----total similarity-----")
        #print(total_similarity)
        #print("-----end-----")
        #print("-----sim*neighbor_matrix-----")
        #print(np.dot(similarity, m))
        #print("-----end-----")
        expect_votes = np.dot(similarity, m)/total_similarity
        #print("-----expect_votes-----")
        #print(expect_votes)
        #print("-----end-----")
        id_idx = np.lexsort((-expect_votes,))
        #print("-----id_idx_sorted-----")
        #print(id_idx)
        #print("-----end-----")
        if (except_voted):
            #print("-----record_non_zero-----")
            #print(np.nonzero(record)[0])
            #print("-----end-----")
            ret =[]
            for id in id_idx:
                if not id in np.nonzero(record)[0]:
                    ret.append(id)

            #id_idx = np.setdiff1d(id_idx, np.nonzero(record)[0])
            id_idx = ret
        print("-----id_idx_except-----")
        print(id_idx)
        print("-----end-----")
        return id_idx[0:items_num]

class ItemBasedKNNRecommender:
    """
    class for item based KNN recommender.
    when doing recommendation, it calculates the similarity between items and user voted items.
    $$similarity[i][j] = X[:][i]*X[:][j]/|X[:][i]|/|X[:][j]|$$.
    Then use current user's votes to estimate votes for rest items.
    $$X[u][i] = (\sum_{j\in voted_items} similarity[i][j]*X[u][j])/(\sum_{j \in voted_items} similarity[i][j])$$.
    """
    def __init__(self, X, lazy = False): #, neighbors_size = 10
        """
        item based knn recommender
        :param X: np.matrix, the element in matrix should >= 0. X[user][item] == 0 means the user hasn't voted the item.
        :param lazy: whether calculate similarity at recommend time,
        if the size of item set is very large, it may be a better choice
        """
        self.X_ori = X
        self.X = X/np.linalg.norm(X, axis = 0)
        self.lazy = lazy
        if not lazy:
            self.similarity = self.X.transpose()*self.X
        #self.neighbors_size = neighbors_size
    def recommend(self, id, items_num = 10, except_voted = True):
        """
        recommend item for user
        :param id: user id, row num of the matrix.
        :param items_num: return at most how many items.
        :param except_voted: whether remove the items that user already has.
        :return: the list of the highest ranked items.
        """
        print(time.strftime("start recommend: %Y-%m-%d %X",time.localtime()))
        record = np.squeeze(np.asarray(self.X[id]))
        print(time.strftime("get record:%Y-%m-%d %X",time.localtime()))
        #print("-----record-----")
        #print(record)
        #print("-----end-----")
        voted = np.argwhere(record > 0).flatten()
        print(time.strftime("voted: %Y-%m-%d %X",time.localtime()))
        #print("-----voted-----")
        #print(voted)
        #print("-----end-----")

        if self.lazy:
            other_user = np.nonzero(np.asarray(np.max(self.X[:,voted], axis = 1)).flatten())[0]
            print(time.strftime("other user:%Y-%m-%d %X",time.localtime()))
            #print("-----other user-----")
            #print(other_user)
            #print("-----end-----")
            candidate_item_ids = np.argwhere(np.asarray(np.max(self.X[other_user,:], axis = 0)).flatten() > 0).flatten()
            #np.argwhere(np.asarray(np.max(matrix[[1,2],:], axis = 0)).flatten() > 0)[0]
            print(time.strftime("candidate item ids:%Y-%m-%d %X",time.localtime()))
            print("-----candidate item ids-----")
            print(candidate_item_ids)
            print("-----end-----")

            m = self.X[:, candidate_item_ids]
            #m = self.X[:, candidate_item_ids]/ np.linalg.norm(self.X[:, candidate_item_ids], axis=0)[:,None]
            print(time.strftime("m:%Y-%m-%d %X",time.localtime()))
            #print("-----m-----")
            #print(m)
            #print("-----end-----")
            voted_m = self.X[:,voted]
            #voted_m = self.X[:,voted]/ np.linalg.norm(self.X[:,voted],axis=0)[:,None]
            print(time.strftime("voted_m:%Y-%m-%d %X",time.localtime()))
            #print("-----voted_m-----")
            #print(voted_m)
            #print("-----end-----")
            sim = voted_m.transpose()*m
        else:
            sim = self.similarity[voted, :]
        expect_votes = np.squeeze(np.asarray(self.X_ori[id, voted] * sim))
        print(time.strftime("expect_votes:%Y-%m-%d %X",time.localtime()))
        #print("-----expect_votes-----")
        #print(expect_votes)
        #print("-----end-----")
        id_idxes = expect_votes.argsort()[::-1]
        print(time.strftime("id_idxes:%Y-%m-%d %X",time.localtime()))
        #print("-----id_idxes-----")
        #print(id_idxes)
        #print("-----end-----")
        if self.lazy:
            candidate_item_ids =  candidate_item_ids[id_idxes]
        else:
            candidate_item_ids = np.array(range(0,np.size(self.X,1)))[id_idxes]
        print(time.strftime("candidate_item_ids:%Y-%m-%d %X",time.localtime()))
        if (except_voted):
            #print("-----record_non_zero-----")
            #print(np.nonzero(record)[0])
            #print("-----end-----")
            ret =[]
            for id in candidate_item_ids:
                if not id in np.nonzero(record)[0]:
                    ret.append(id)
            candidate_item_ids = ret
        return ret[0:items_num]





