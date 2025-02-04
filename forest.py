import numpy as np
import pandas as pd

from tree import DecisionTreeRegressor

class randomforest(DecisionTreeRegressor):
    def __init__(self, data_train, maxdepth, num_features, num_trees):
        self.data_train = data_train
        self.maxdepth = maxdepth
        self.num_features = num_features
        self.num_trees = num_trees
        self.trees = []
        y_train0 = self.data_train[:,-1]
        data_arange1 = np.arange(len(self.data_train[0])-1)
        l_data_train = len(self.data_train)
        data_arange2 = np.arange(l_data_train)
        for i in range(self.num_trees):
            randfeatures = np.random.choice(data_arange1, size=self.num_features, replace=False)
            # self.data_train[np.random.choice(np.arange(len(self.data)), size=len(self.data)),randfeatures]
            rows = np.random.choice(data_arange2, size=l_data_train,replace=False)
            data_rows = self.data_train[rows]
            x_train = data_rows[:,randfeatures]
            y_train = y_train0[rows]
            dtc = DecisionTreeRegressor(x_train, y_train, max_depth=self.maxdepth)
            self.trees.append(dtc)
    def predict(self, x):
        predict = np.zeros(len(x))
        # x_train = self.data_train[:,:-1]
        for i in range(self.num_trees):
            curr_predict = self.trees[i].predict(x)
            predict += curr_predict
        predict = predict/self.num_trees
        return predict
    