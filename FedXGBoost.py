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

    def appendData(self, dataTable, featureName = None):
        """
        Dimension definition: 
        -   dataTable   nxm: <n> users & <m> features
        -   name        mx1: <m> strings
        """
        # TODO: This is just for now to let the code run
        tmp = dataTable.copy()
        self.data = tmp[:,:-1]
        data_num = self.data.shape[0]
        y = dataTable[:, -1]
        self.label = np.reshape(y, (data_num, 1))
        self.getAllQuantile()
        
        # Implementing the database
        nFeatures = len(dataTable[0])
        if(featureName is None):
            featureName = ["Rank_{}_Feature_".format(rank) + str(i) for i in range(nFeatures)]
        
        assert (len(featureName) is nFeatures) # The total amount of columns must match the assigned name 
        
        for i in range(len(featureName)):
            self.dataBase.appendFeature(FeatureData(featureName[i], dataTable[:,i]))

        logger.warning('Appended data')


    def printInfo(self):
        featureListStr = '' 
        self.dataBase.printInfo()

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
        X = self.dataBase.getDataMatrix()
        if(rank == 2):
            print("data ", np.shape(self.data))
            print("X ", np.shape(X))

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
                update_pred = self.trees[i].predictTrain(X)
            if self.rank == 1:
                update_pred = np.reshape(update_pred, (data_num, 1))
                y_pred += update_pred


def test():
    data = pd.read_csv('./iris.csv').values

    zero_index = data[:, -1] == 0
    one_index = data[:, -1] == 1
    zero_data = data[zero_index]
    one_data = data[one_index]
    train_size_zero = int(zero_data.shape[0] * 0.8)
    train_size_one = int(one_data.shape[0] * 0.8)
    X_train, X_test = np.concatenate((zero_data[:train_size_zero, :-1], one_data[:train_size_one, :-1]), 0), \
                      np.concatenate((zero_data[train_size_zero:, :-1], one_data[train_size_one:, :-1]), 0)
    y_train, y_test = np.concatenate((zero_data[:train_size_zero, -1].reshape(-1,1), one_data[:train_size_one, -1].reshape(-1, 1)), 0), \
                      np.concatenate((zero_data[train_size_zero:, -1].reshape(-1, 1), one_data[train_size_one:, -1].reshape(-1, 1)), 0)

    X_train_A = X_train[:, 0].reshape(-1, 1)
    X_train_B = X_train[:, 2].reshape(-1, 1)
    X_train_C = X_train[:, 1].reshape(-1, 1)
    X_train_D = X_train[:, 3].reshape(-1, 1)
    X_test_A = X_test[:, 0].reshape(-1, 1)
    X_test_B = X_test[:, 2].reshape(-1, 1)
    X_test_C = X_test[:, 1].reshape(-1, 1)
    X_test_D = X_test[:, 3].reshape(-1, 1)
    splitclass = SSCalculate()
    model = FedXGBoostClassifier(rank=rank, lossfunc='LogLoss', splitclass=splitclass)

    # np.concatenate((X_train_A, y_train))
    if rank == 1:
        #print("Test A", len(X_train_A), len(X_train_A[0]), len(y_train), len(y_train[0]))
        #print("Test A", X_train_A.shape[0], len(X_train_A[0]), len(y_train), len(y_train[0]))
        model.appendData(np.concatenate((X_train_A, y_train), 1))
    elif rank == 2:
        #print("Test", len(X_train_B), len(X_train_B[0]), len(y_train), len(y_train[0]))
        model.appendData(np.concatenate((X_train_B, y_train), 1))
    elif rank == 3:
        model.appendData(np.concatenate((X_train_C, y_train), 1))
    elif rank == 4:
        model.appendData(np.concatenate((X_train_D, y_train), 1))
    else:
        model.appendData(np.concatenate((X_train_A, y_train), 1))

    model.printInfo()


    model.boost()
    
    # b = FLVisNode(model.trees)
    # b.display()

    if rank == 1:
        y_pred = model.predict(X_test_A)
    elif rank == 2:
        y_pred = model.predict(X_test_B)
    elif rank == 3:
        y_pred = model.predict(X_test_C)
    elif rank == 4:
        y_pred = model.predict(X_test_D)
    else:
        model.predict(np.zeros_like(X_test_A))

    if rank == 1:
        y_pred = 1.0 / (1.0 + np.exp(-y_pred))
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
        result = y_pred - y_test
        print(np.sum(result == 0) / y_pred.shape[0])
        # for i in range(y_test.shape[0]):
        #     print(y_test[i], y_pred[i], y_ori[i])
    pass







from VerticalXGBoost import main2

def main4():
    #data = pd.read_csv('./GiveMeSomeCredit/cs-training.csv')
    data = pd.read_csv('./GiveMeSomeCredit/cs-training-small.csv')
    data.dropna(inplace=True)
    data = data[['SeriousDlqin2yrs',
       'RevolvingUtilizationOfUnsecuredLines', 'age',
       'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
       'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
       'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
       'NumberOfDependents']].values
    ori_data = data.copy()
    # Add features
    # for i in range(1):
    #     data = np.concatenate((data, ori_data[:, 1:]), axis=1)
    data = data / data.max(axis=0)

    ratio = 10000 / data.shape[0]


    zero_index = data[:, 0] == 0
    one_index = data[:, 0] == 1
    zero_data = data[zero_index]
    one_data = data[one_index]
    zero_ratio = len(zero_data) / data.shape[0]
    one_ratio = len(one_data) / data.shape[0]
    num = 7500
    train_size_zero = int(zero_data.shape[0] * ratio) + 1
    train_size_one = int(one_data.shape[0] * ratio)
    X_train, X_test = np.concatenate((zero_data[:train_size_zero, 1:], one_data[:train_size_one, 1:]), 0), \
                      np.concatenate((zero_data[train_size_zero:train_size_zero+int(num * zero_ratio)+1, 1:], one_data[train_size_one:train_size_one+int(num * one_ratio), 1:]), 0)
    y_train, y_test = np.concatenate(
        (zero_data[:train_size_zero, 0].reshape(-1, 1), one_data[:train_size_one, 0].reshape(-1, 1)), 0), \
                      np.concatenate((zero_data[train_size_zero:train_size_zero+int(num * zero_ratio)+1, 0].reshape(-1, 1),
                                      one_data[train_size_one:train_size_one+int(num * one_ratio), 0].reshape(-1, 1)), 0)

    X_train_A = X_train[:, :2]
    X_train_B = X_train[:, 2:4]
    X_train_C = X_train[:, 4:7]
    X_train_D = X_train[:, 7:]
    X_test_A = X_test[:, :2]
    X_test_B = X_test[:, 2:4]
    X_test_C = X_test[:, 4:7]
    X_test_D = X_test[:, 7:]

    splitclass = SSCalculate()
    model = FedXGBoostClassifier(rank=rank, lossfunc='LogLoss', splitclass=splitclass, max_depth=3, n_estimators=3, _epsilon=0.1)

    start = datetime.now()
     # np.concatenate((X_train_A, y_train))
    if rank == 1:
        #print("Test A", len(X_train_A), len(X_train_A[0]), len(y_train), len(y_train[0]))
        #print("Test A", X_train_A.shape[0], len(X_train_A[0]), len(y_train), len(y_train[0]))
        model.appendData(np.concatenate((X_train_A, y_train), 1))
    elif rank == 2:
        #print("Test", len(X_train_B), len(X_train_B[0]), len(y_train), len(y_train[0]))
        model.appendData(np.concatenate((X_train_B, y_train), 1))
    elif rank == 3:
        model.appendData(np.concatenate((X_train_C, y_train), 1))
    elif rank == 4:
        model.appendData(np.concatenate((X_train_D, y_train), 1))
    else:
        model.appendData(np.concatenate((X_train_A, y_train), 1))

    model.printInfo()


    model.boost()

    if rank == 1:
        y_pred = model.predict(X_test_A)
    elif rank == 2:
        y_pred = model.predict(X_test_B)
    elif rank == 3:
        y_pred = model.predict(X_test_C)
    elif rank == 4:
        y_pred = model.predict(X_test_D)
    else:
        model.predict(np.zeros_like(X_test_A))

    if rank == 1:
        y_pred = 1.0 / (1.0 + np.exp(-y_pred))
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
        result = y_pred - y_test
        print(np.sum(result == 0) / y_pred.shape[0])
        # for i in range(y_test.shape[0]):
        #     print(y_test[i], y_pred[i], y_ori[i])
    pass




try:
    test()
   # main4()

    

except Exception as e:
  logging.error("Exception occurred", exc_info=True)

