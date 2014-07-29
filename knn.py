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
    def __init__(self, X, lazy = False, neighbors_size = 10, kdt = None):
        """
        item based knn recommender
        :param X: np.matrix, the element in matrix should >= 0. X[user][item] == 0 means the user hasn't voted the item.
        :param lazy: whether calculate similarity at recommend time.
        :param neighbors_size: use how many neighbors' voting to decide the user's vote
        :param kdt: KDTree used to calculate neighbors
        """
        self.X_ori = X
        #print("-----ori_X-----")
        #print(X)
        #print("-----end-----")
        self.X = X/np.linalg.norm(X, axis = 1)[:,None]
        #print("-----Xnorm-----")
        #print(self.X)
        #print("-----end-----")
        self.neighbors_size = neighbors_size if len(self.X) > neighbors_size else len(self.X)
        self.lazy = lazy
        self.use_kdtree = not kdt is None
        if not lazy and self.use_kdtree:
            raise Exception("kdtree is used for lazily calculate similarity between users. Cannot use kdtree in non lazy mode.")
        if lazy and self.use_kdtree:
            #print("-----ini kdtree-----")
            self.kdt = kdt
            #self.kdt = KDTree(self.X_ori, leaf_size=30, metric= 'euclidean')
            #print("-----end-----")
        if not lazy:
            self.similarity = self.X * self.X.transpose()
    def recommend(self, id, items_num = 10, except_voted = True, ret_expect_vote = False):
        """
        recommend item for user
        :param id: user id, row num of the matrix.
        :param items_num: return at most how many items.
        :param except_voted: whether remove the items that user already has.
        :return: the list of the highest ranked items.
        """
        #print(time.strftime("start recommend:%Y-%m-%d %X",time.localtime()))
        record = np.squeeze(np.asarray(self.X_ori[id]))
        voted = np.argwhere(record > 0).flatten()
        #print("-----voted-----")
        #print(voted)
        #print("-----end-----")
        if not self.lazy:
            neighbor_ids = np.squeeze(np.asarray(self.similarity[id])).argsort()[::-1][0:self.neighbors_size+1]
        if self.use_kdtree:
            neighbor_ids = self.kdt.query([record], k=self.neighbors_size+1,return_distance=False).flatten()
        else:
            neighbor_ids = np.nonzero(np.asarray(np.max(self.X[:,voted], axis = 1)).flatten())[0]
        neighbor_ids = neighbor_ids[neighbor_ids != id]
        #print("-----neighbor ids-----")
        #print(neighbor_ids)
        #neighbor_ids = np.array([0, 1, 2, 3, 5, 6])
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
        #print(time.strftime("get neighbor matrix:%Y-%m-%d %X",time.localtime()))
        #m = self.X[neighbor_ids]
        #print("-----neighbors' matrix-----")
        #print(m)
        #print("-----end-----")
        #m = np.matrix(neighbor_matrix)
        #def cosine_simi(u,v):
        #    print(u)
        #    #return None
        #    print(v)
        #    u = np.array(u)
        #    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
        #print(time.strftime("calc similarity:%Y-%m-%d %X",time.localtime()))
        #similarity = np.ma.apply_along_axis( cosine_simi, 1, m, record)
        if self.lazy:
            similarity = self.X[id]*self.X[neighbor_ids].transpose()
        else:
            similarity = self.similarity[id][:, neighbor_ids]
        #print("-----similarity-----")
        #print(similarity)
        #print("-----end-----")
        #print(time.strftime("calc sum of similarity:%Y-%m-%d %X",time.localtime()))
        total_similarity = np.sum(similarity)
        #print("-----total similarity-----")
        #print(total_similarity)
        #print("-----end-----")
        #print("-----sim*neighbor_matrix-----")
        #print(np.dot(similarity, self.X_ori[neighbor_ids]))
        #print("-----end-----")
        #print(time.strftime("expect_votes:%Y-%m-%d %X",time.localtime()))
        expect_votes =  np.squeeze(np.asarray(np.dot(similarity, self.X_ori[neighbor_ids])/total_similarity))
        #print("-----expect_votes-----")
        #print(expect_votes)
        #print("-----end-----")
        #print(time.strftime("sort votes:%Y-%m-%d %X",time.localtime()))
        candidate_item_ids = np.lexsort((-expect_votes,))
        #print("-----id_idx_sorted-----")
        #print(id_idx)
        #print("-----end-----")
        #print("-----candidate_item_ids_before_except-----")
        #print(candidate_item_ids)
        #print("-----end-----")
        if (except_voted):
            candidate_item_ids = [a  for a in candidate_item_ids if not a in voted]
        #print("-----candidate_item_ids_except-----")
        #print(candidate_item_ids)
        #print("-----end-----")
        #print(time.strftime("end:%Y-%m-%d %X",time.localtime()))
        return (candidate_item_ids[0:items_num], list(expect_votes[candidate_item_ids]) if ret_expect_vote else None)

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
        :param lazy: whether calculate similarity at recommend time.
        """
        self.X_ori = X
        #print("-----ori matrix-----")
        #print(self.X_ori)
        #print("-----end-----")
        self.X = X/np.linalg.norm(X, axis = 0)
        #print("-----normed-----")
        #print(self.X)
        #print("-----end-----")
        self.lazy = lazy
        if not lazy:
            self.similarity = self.X.transpose()*self.X
            #print("-----similarity-----")
            #print(self.similarity)
            #print("-----end-----")
        #self.neighbors_size = neighbors_size
    def recommend(self, id, items_num = 10, except_voted = True, ret_expect_vote = False):
        """
        recommend item for user
        :param id: user id, row num of the matrix.
        :param items_num: return at most how many items.
        :param except_voted: whether remove the items that user already has.
        :return: the list of the highest ranked items.
        """
        #print(time.strftime("start recommend: %Y-%m-%d %X",time.localtime()))
        record = np.squeeze(np.asarray(self.X[id]))
        #print(time.strftime("get record:%Y-%m-%d %X",time.localtime()))
        #print("-----record-----")
        #print(record)
        #print("-----end-----")
        voted = np.argwhere(record > 0).flatten()
        #print(time.strftime("voted: %Y-%m-%d %X",time.localtime()))
        #print("-----voted-----")
        #print(voted)
        #print("-----end-----")

        if self.lazy:
            other_user = np.nonzero(np.asarray(np.max(self.X[:,voted], axis = 1)).flatten())[0]
            #print(time.strftime("other user:%Y-%m-%d %X",time.localtime()))
            #print("-----other user-----")
            #print(other_user)
            #print("-----end-----")
            candidate_item_ids = np.argwhere(np.asarray(np.max(self.X[other_user,:], axis = 0)).flatten() > 0).flatten()
            #np.argwhere(np.asarray(np.max(matrix[[1,2],:], axis = 0)).flatten() > 0)[0]
            #print(time.strftime("candidate item ids:%Y-%m-%d %X",time.localtime()))
            #print("-----candidate item ids-----")
            #print(candidate_item_ids)
            #print("-----end-----")

            m = self.X[:, candidate_item_ids]
            #m = self.X[:, candidate_item_ids]/ np.linalg.norm(self.X[:, candidate_item_ids], axis=0)[:,None]
            #print(time.strftime("m:%Y-%m-%d %X",time.localtime()))
            #print("-----m-----")
            #print(m)
            #print("-----end-----")
            voted_m = self.X[:,voted]
            #voted_m = self.X[:,voted]/ np.linalg.norm(self.X[:,voted],axis=0)[:,None]
            #print(time.strftime("voted_m:%Y-%m-%d %X",time.localtime()))
            #print("-----voted_m-----")
            #print(voted_m)
            #print("-----end-----")
            sim = voted_m.transpose()*m
        else:
            sim = self.similarity[voted, :]
        #print("-----sim-----")
        #print(sim)
        #print("-----end-----")
        #total_sim = np.sum(sim, axis =0)
        #print("-----total_sim-----")
        #print(total_sim)
        #print("-----end-----")
        #print("-----exp-----")
        #print(self.X_ori[id, voted] * sim)
        #print("-----end-----")
        #print("-----sim/total_sim-----")
        #print(sim/total_sim)
        #print("-----end-----")
        # it seems that divided by total_sim is not a good choice.
        # for example.
        # if user A only voted item X.
        # similarity between other items and X is [s_1,s_2...s_n]
        # so every items's expect vote is [s_1*vote[X],s_2*vote[X],...s_n*vote[X]]
        # and divided by total_sim
        # expect_votes is [vote[X], vote[X], ...]
        expect_votes = np.squeeze(np.asarray(self.X_ori[id, voted] * sim))#/total_sim
        #print(time.strftime("expect_votes:%Y-%m-%d %X",time.localtime()))
        #print("-----expect_votes-----")
        #print(expect_votes)
        #print("-----end-----")
        id_idxes = expect_votes.argsort()[::-1]
        #print(time.strftime("id_idxes:%Y-%m-%d %X",time.localtime()))
        #print("-----id_idxes-----")
        #print(id_idxes)
        #print("-----end-----")
        if self.lazy:
            candidate_item_ids =  candidate_item_ids[id_idxes]
        else:
            candidate_item_ids = np.array(range(0,np.size(self.X,1)))[id_idxes]
        expect_votes =  sorted(expect_votes, reverse = True)
        #print(time.strftime("candidate_item_ids:%Y-%m-%d %X",time.localtime()))
        if (except_voted):
            expect_votes = [expect_votes[idx] for idx, a in enumerate(candidate_item_ids) if not a in voted]
            candidate_item_ids = [a for a in candidate_item_ids if not a in voted]
        return (candidate_item_ids[0:items_num], list(expect_votes[0:items_num])if ret_expect_vote else None)





