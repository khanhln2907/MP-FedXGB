import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from DataBaseStructure import *
from TreeStructure import *
from FedXGBoostTree import VerticalFedXGBoostTree
from Common import PARTY_ID, logger, rank, clientNum
from VerticalXGBoost import LogLoss, LeastSquareLoss




class FedXGBoostClassifier():
    def __init__(self, rank, lossfunc, _lambda=1, _gamma=0.5, _epsilon=0.1, n_estimators=3, max_depth=3):
        if lossfunc == 'LogLoss':
            self.loss = LogLoss()
        else:
            self.loss = LeastSquareLoss()
        self._lambda = _lambda
        self._gamma = _gamma
        self._epsilon = _epsilon
        self.n_estimators = n_estimators  # Number of trees
        self.max_depth = max_depth  # Maximum depth for tree
        self.trees = []
        for _ in range(n_estimators):
            tree = VerticalFedXGBoostTree(
                                        lossfunc=self.loss,
                                       _lambda=self._lambda,
                                        _gamma=self._gamma,
                                       _epsilon=self._epsilon,
                                       _maxdepth=self.max_depth,
                                       clientNum=clientNum)
            self.trees.append(tree)

        self.data = []
        self.label = []
        self.dataBase = DataBase()

    def append_data(self, dataTable, fName = None):
        """
        Dimension definition: 
        -   dataTable   nxm: <n> users & <m> features
        -   name        mx1: <m> strings
        """
        self.dataBase = DataBase.data_matrix_to_database(dataTable, fName)
        logger.warning('Appended data feature %s to database of party %d', str(fName), rank)

    def append_label(self, labelVec):
        self.label = np.reshape(labelVec, (len(labelVec), 1))

    def print_info(self):
        featureListStr = '' 
        ret = self.dataBase.log()
        print(ret)

    def boost(self):
        # TODO: the data is passed in the method append data
        orgData = deepcopy(self.dataBase)
        y = self.label
        y_pred = np.zeros(np.shape(self.label))
        
        for i in range(self.n_estimators):     
            # Perform tree boosting
            dataFit = QuantiledDataBase(self.dataBase)
            self.trees[i].fit_fed(y, y_pred, i, dataFit)

            if i == self.n_estimators - 1: # The last tree, no need for prediction update.
                continue
            else:
                update_pred = self.trees[i].predict_fed(orgData)
            if rank == PARTY_ID.ACTIVE_PARTY:
                update_pred = np.reshape(update_pred, (self.dataBase.nUsers, 1))
                y_pred += update_pred

    def predict_fed(self, X, fName = None):
        y_pred = None
        data_num = X.shape[0]
        # Make predictions
        testDataBase = DataBase.data_matrix_to_database(X, fName)
        for tree in self.trees:
            # Estimate gradient and update prediction
            update_pred = tree.predict_fed(testDataBase)
            if y_pred is None:
                y_pred = np.zeros_like(update_pred).reshape(data_num, -1)
            if rank == 1:
                update_pred = np.reshape(update_pred, (data_num, 1))
                y_pred += update_pred
        return y_pred
