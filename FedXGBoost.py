import numpy as np
from gpg import Data
from SSCalculation import *
import matplotlib.pyplot as plt
from DataBaseStructure import *
from TreeStructure import *

from VerticalXGBoost import VerticalXGBoostClassifier
from FedXGBoostTree import VerticalFedXGBoostTree
from Common import logger




class FedXGBoostClassifier(VerticalXGBoostClassifier):
    def __init__(self, rank, lossfunc, splitclass, _lambda=1, _gamma=0.5, _epsilon=0.1, n_estimators=3, max_depth=3):
        super().__init__(rank, lossfunc, splitclass, _lambda, _gamma, _epsilon, n_estimators, max_depth)

        self.trees = []
        for _ in range(n_estimators):
            tree = VerticalFedXGBoostTree(rank=self.rank,
                                       lossfunc=self.loss,
                                       splitclass=self.splitclass,
                                       _lambda=self._lambda,
                                        _gamma=self._gamma,
                                       _epsilon=self._epsilon,
                                       _maxdepth=self.max_depth,
                                       clientNum=clientNum)
            self.trees.append(tree)

        self.data = []
        self.label = []
        self.dataBase = DataBase()

    def append_data(self, dataTable, featureName = None):
        """
        Dimension definition: 
        -   dataTable   nxm: <n> users & <m> features
        -   name        mx1: <m> strings
        """
        # TODO: This is just for now to let the code run
        self.data = dataTable.copy()
        #y = dataTable[:, -1]
        #self.label = np.reshape(y, (data_num, 1))
        self.getAllQuantile()
        
        # Implementing the database
        nFeatures = len(dataTable[0])
        if(featureName is None):
            featureName = ["Rank_{}_Feature_".format(rank) + str(i) for i in range(nFeatures)]
        
        assert (len(featureName) is nFeatures) # The total amount of columns must match the assigned name 
        
        for i in range(len(featureName)):
            self.dataBase.append_feature(FeatureData(featureName[i], dataTable[:,i]))

        logger.warning('Appended data')


    def append_label(self, labelVec):
        self.label = np.reshape(labelVec, (len(labelVec), 1))


    def print_info(self):
        featureListStr = '' 
        self.dataBase.print_info()

    def boostDepr(self):
        data_num = self.data.shape[0]
        X = self.data
        y = self.label
        y_pred = np.zeros(np.shape(self.label))
        for i in range(self.n_estimators):
            logger.info("Iter: %d. Amount of splitting candidates: %d", i, self.maxSplitNum)
            for key, value in self.quantile.items():
                logger.info("Quantile %d: ", key)
                logger.info("{}".format(' '.join(map(str, value))))
            
            tree = self.trees[i]
            tree.data, tree.maxSplitNum, tree.quantile = self.data, self.maxSplitNum, self.quantile
            #print((np.array(self.quantile)))
            y_and_pred = np.concatenate((y, y_pred), axis=1)

            # Perform tree boosting
            tree.fit(y_and_pred, i, self.data, self.quantile)
            if i == self.n_estimators - 1: # The last tree, no need for prediction update.
                continue
            else:
                update_pred = tree.predict(X)
            if self.rank == 1:
                update_pred = np.reshape(update_pred, (data_num, 1))
                y_pred += update_pred

    def boost(self):
        # TODO: the data is passed in the method append data
        data_num = self.data.shape[0]
        X = self.dataBase.get_data_matrix()
        if(rank == 2):
            print("data ", np.shape(self.data))
            print("X ", np.shape(X))
        orgData = self.data.copy()

        y = self.label
        y_pred = np.zeros(np.shape(self.label))
        
        
        for i in range(self.n_estimators):
            
            
            self.trees[i].data, self.trees[i].maxSplitNum, self.trees[i].quantile = self.data, self.maxSplitNum, self.quantile
            y_and_pred = np.concatenate((y, y_pred), axis=1)

            # Perform tree boosting
            dataFit = QuantiledDataBase(self.dataBase)
            self.trees[i].fit(y_and_pred, i, dataFit)

            if i == self.n_estimators - 1: # The last tree, no need for prediction update.
                continue
            else:
                #print("hello")
                update_pred = self.trees[i].predictTrain(orgData)
            if self.rank == 1:
                update_pred = np.reshape(update_pred, (data_num, 1))
                y_pred += update_pred

