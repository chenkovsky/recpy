__author__ = 'chenkovsky'

import pandas as pd
import numpy as np
from . import knn
class TestRecommender:
    def setUp(self):
        data = {1: {1: 3.0, 2: 4.0, 3: 3.5, 4: 5.0, 5: 3.0},
         2: {1: 3.0, 2: 4.0, 3: 2.0, 4: 3.0, 5: 3.0, 6: 2.0},
         3: {2: 3.5, 3: 2.5, 4: 4.0, 5: 4.5, 6: 3.0},
         4: {1: 2.5, 2: 3.5, 3: 2.5, 4: 3.5, 5: 3.0, 6: 3.0},
         5: {2: 4.5, 3: 1.0, 4: 4.0},
         6: {1: 3.0, 2: 3.5, 3: 3.5, 4: 5.0, 5: 3.0, 6: 1.5},
         7: {1: 2.5, 2: 3.0, 4: 3.5, 5: 4.0}}
        df =pd.DataFrame(data)
        m = np.matrix(df)
        m = m.transpose()
        self.matrix = np.nan_to_num(m)
    def testUserBasedKNNRecommender(self):
        rec = knn.UserBasedKNNRecommender(self.matrix)
        assert(rec.recommend(4) == [4,0,5])
    def testItemBasedKNNRecommender(self):
        rec = knn.ItemBasedKNNRecommender(self.matrix)
        assert(rec.recommend(4) == [4,0,5])
        rec = knn.ItemBasedKNNRecommender(self.matrix, lazy = True)
        assert(rec.recommend(4) == [4,0,5])