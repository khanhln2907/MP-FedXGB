from curses.ascii import SI
from posixpath import split
import random
from tkinter import N
from gpg import Data
from numpy import concatenate
from VerticalXGBoost import *
from SSCalculation import *
import matplotlib.pyplot as plt
import logging
from DataBaseStructure import *
from TreeStructure import *

from Common import logger, TreeNodeType

class PARTY_ID:
    ACTIVE_PARTY = 1


class MSG_ID:
    MASKED_GH = 99
    RAW_SPLITTING_MATRIX = 98
    OPTIMAL_SPLITTING_INFO = 97

def compute_splitting_score(SM, GVec, HVec, lamb):
    G = sum(GVec)
    H = sum(HVec)
    GRVec = np.matmul(SM, GVec)
    HRVec = np.matmul(SM, HVec)
    GLVec = G - GRVec
    HLVec = H - HRVec
    # logger.info("Received from party {} \n".format(partners) + \
    #     "GR: " + str(sumGRVec.T) + "\n" + "HR: " + str(sumHRVec.T) +\
    #     "\nGL: " + str(sumGLVec.T) + "\n" + "HL: " + str(sumHLVec.T))  

    L = (GLVec*GLVec / (HLVec + lamb)) + (GRVec*GRVec / (HRVec + lamb)) - (G*G / (H + lamb))
    return L.reshape(-1)

class SplittingInfo:
    def __init__(self) -> None:
        self.bestSplitScore = -np.Infinity
        self.bestSplitParty = None
        self.bestSplittingVector = None

    def log(self, logger):
        logger.info("Best Splitting Score: L = %.2f, Selected Party %s",\
                self.bestSplitScore, str(self.bestSplitParty))
        logger.debug("The optimal splitting vector: %s", str(self.bestSplittingVector))

class FedXGBoostSecureHandler:
    QR = []
    pass

    def generate_secure_kernel(mat):
        import scipy.linalg
        return scipy.linalg.qr(mat)

    def calc_secure_response(privateMat, rxKernelMat):
        n = len(rxKernelMat) # n users 
        r = len(rxKernelMat[0]) # number of kernel vectors
        Z = rxKernelMat
        return np.matmul((np.identity(n) - np.matmul(Z, np.transpose(Z))), privateMat)

    def generate_splitting_matrix(dataVector, quantileBin):
        n = len(dataVector) # Rows as n users
        l = len(quantileBin) # amount of splitting candidates

        retSplittingMat = []
        for candidateIter in range(l):
            v = np.zeros(n)
            for userIter in range(n):
                v[userIter] = (dataVector[userIter] > max(quantileBin[candidateIter]))  
            retSplittingMat.append(v)

        return retSplittingMat


class VerticalFedXGBoostTree(VerticalXGBoostTree):
    def __init__(self, rank, lossfunc, splitclass, _lambda, _gamma, _epsilon, _maxdepth, clientNum):
        super().__init__(rank, lossfunc, splitclass, _lambda, _gamma, _epsilon, _maxdepth, clientNum)

        self.root = FLTreeNode()

    def fitDepr(self, y_and_pred, tree_num, xData, sQuantile):
        super().fit(y_and_pred, tree_num)


        # Compute the gradients
        if self.rank == PARTY_ID.ACTIVE_PARTY: # Calculate gradients on the node who have labels.
            y, y_pred = self._split(y_and_pred)
            G = self.loss.gradient(y, y_pred)
            H = self.loss.hess(y, y_pred)
            logger.info("Computed Gradients and Hessians ")
            logger.debug("G {}".format(' '.join(map(str, G))))
            logger.debug("H {}".format(' '.join(map(str, H))))

            gh = np.concatenate((G, H), axis=1)
            nprocs = comm.Get_size()
            for partners in range(2, nprocs):   
                logger.info("Sending G, H to party %d", partners)         
                data = comm.send(G, dest = partners, tag = MSG_ID.MASKED_GH)
        
        elif rank != 0: # TODO: change this hard coded number
        #else:
            data = comm.recv(source=PARTY_ID.ACTIVE_PARTY, tag=MSG_ID.MASKED_GH)
            logger.info("Received G, H")         
            
            # Perform the secure Sharing of the splitting matrix
            # for featureID in range(len(sQuantile[0])):
            #     splitMat = FedXGBoostSecureHandler.generate_splitting_matrix(xData, sQuantile[0])
            if rank == 2:
                #print(xData)
                pass

    def fit(self, y_and_pred, treeID, qDataBase: QuantiledDataBase):
        logger.info("Tree is growing column-wise. Current column: %d", treeID)

        super().fit(y_and_pred, treeID)

        """
        This function computes the gradient and the hessian vectors to perform the tree construction
        """
        # Compute the gradients and hessians
        if self.rank == PARTY_ID.ACTIVE_PARTY: # Calculate gradients on the node who have labels.
            y, y_pred = self._split(y_and_pred)
            G = np.array(self.loss.gradient(y, y_pred)).reshape(-1)
            H = np.array(self.loss.hess(y, y_pred)).reshape(-1)
            logger.info("Computed Gradients and Hessians ")
            logger.debug("G {}".format(' '.join(map(str, G))))
            logger.debug("H {}".format(' '.join(map(str, H))))

            # nprocs = comm.Get_size()
            # for partners in range(2, nprocs):   
            #     data = comm.send(G, dest = partners, tag = MSG_ID.MASKED_GH)
            #     logger.info("Sent G, H to party %d", partners)         

            qDataBase.appendGradientsHessian(G, H) 

        else:
            # data = comm.recv(source=PARTY_ID.ACTIVE_PARTY, tag=MSG_ID.MASKED_GH)
            # logger.info("Received G, H from the active party")

            dummyG = np.zeros((qDataBase.nUsers,1))
            dummyH = np.zeros((qDataBase.nUsers,1))
            qDataBase.appendGradientsHessian(dummyG, dummyH)

        if(rank != 0):
            rootNode = FLTreeNode()
            self.grow(qDataBase, depth = 0, NodeDirection = TreeNodeType.ROOT, currentNode = rootNode)
            self.root = rootNode

            if((rank == 1)):
                treeInfo = self.root.get_string_recursive()
                logger.info("Tree Info:\n%s", treeInfo)
                print(treeInfo)

    def generate_leaf(self, gVec, hVec, lamb = 0.1):
        gI = sum(gVec) 
        hI = sum(hVec)
        ret = TreeNode(-1.0 * gI / (hI + lamb), leftBranch= None, rightBranch= None)
        return ret

    def grow(self, qDataBase: QuantiledDataBase, depth=0, NodeDirection = TreeNodeType.ROOT, currentNode : FLTreeNode = None):
        logger.info("Tree is growing depth-wise. Current depth: {}".format(depth) + " Node's type: {}".format(NodeDirection))

        if self.rank == PARTY_ID.ACTIVE_PARTY:
            sInfo = SplittingInfo()
            nprocs = comm.Get_size()        
            for partners in range(2, nprocs):   
                rxSM = comm.recv(source = partners, tag = MSG_ID.RAW_SPLITTING_MATRIX)

                # Find the optimal splitting score
                sumGRVec = np.matmul(rxSM, qDataBase.gradVec).reshape(-1)
                sumHRVec = np.matmul(rxSM, qDataBase.hessVec).reshape(-1)
                sumGLVec = sum(qDataBase.gradVec) - sumGRVec
                sumHLVec = sum(qDataBase.hessVec) - sumHRVec
                L = compute_splitting_score(rxSM, qDataBase.gradVec, qDataBase.hessVec, 0.01)

                logger.debug("Received SM from party {} and computed:  \n".format(partners) + \
                    "GR: " + str(sumGRVec.T) + "\n" + "HR: " + str(sumHRVec.T) +\
                    "\nGL: " + str(sumGLVec.T) + "\n" + "HL: " + str(sumHLVec.T) +\
                    "\nSplitting Score: {}".format(L.T))       
                
                # Optimal candidate of 1 partner party
                # Select the optimal candidates without all zeros or one elements of the splitting)
                isValid = False
                excId = np.zeros(L.size, dtype=bool)
                for id in range(len(L)):
                    splitVector = rxSM[id, :]

                    nL = np.count_nonzero(splitVector == 0.0)
                    nR = np.count_nonzero(splitVector == 1.0)
                    #print(len(splitVector), nR, nL)
                    thres = 0.2
                    isValid = ((nL/len(splitVector)) > thres) and ((nR/len(splitVector)) > thres)
                    if isValid:
                        pass
                    else:
                        excId[id] = True

                bestSplitId = 0
                tmpL = np.ma.array(L, mask=excId) # Mask the exception index
                bestSplitId = np.argmax(tmpL)
                splitVector = rxSM[bestSplitId, :]
                maxScore = L[bestSplitId]     
                # Select the optimal over all partner parties
                if (maxScore > sInfo.bestSplitScore):
                    sInfo.bestSplitScore = maxScore
                    sInfo.bestSplitParty = partners
                    bestCandidateIndex = bestSplitId
                    sInfo.bestSplittingVector = rxSM[bestCandidateIndex, :]
                    
            # Log the splitting info
            sInfo.log(logger)
            
            # Build Tree from the feature with the optimal index
            for partners in range(2, nprocs):
                    data = comm.send(sInfo, dest = partners, tag = MSG_ID.OPTIMAL_SPLITTING_INFO)
                    logger.info("Sent splitting info to clients {}".format(partners))
        elif (rank != 0):           
            # Perform the secure Sharing of the splitting matrix
            # qDataBase.printInfo()
            privateSM = qDataBase.get_merged_splitting_matrix()
            logger.debug("Secured splitting matrix with shape of {}".format(str(privateSM.shape)) \
                            + "\n {}".format(' '.join(map(str, privateSM))))
            logger.debug("Secured splitting matrix: \n %s", str(privateSM))


            # Send the splitting matrix to the active party
            txSM = comm.send(privateSM, dest = PARTY_ID.ACTIVE_PARTY, tag = MSG_ID.RAW_SPLITTING_MATRIX)
            logger.info("Sent the splitting matrix to the active party")         

            sInfo = comm.recv(source=PARTY_ID.ACTIVE_PARTY, tag = MSG_ID.OPTIMAL_SPLITTING_INFO)
            logger.info("Received the Splitting Info from the active party")   

        # Set the optimal split as the owner ID of the current tree node
        currentNode.owner = sInfo.bestSplitParty
        logger.info("Optimal splitting: %s", str(sInfo.bestSplittingVector))

        # Get the optimal splitting candidates and partition them into two databases
        if(sInfo.bestSplittingVector is not None):
            lD, rD = qDataBase.partition(sInfo.bestSplittingVector)
            logger.info("")
            logger.info("Database is partitioned into two quantiled databases!")
            logger.info("Original database: %s", qDataBase.get_info_string())
            logger.info("Left splitted database: %s", lD.get_info_string())
            logger.info("Right splitted database: %s \n", rD.get_info_string())

            maxDepth = 3
            # Construct the new tree if the gain is positive
            if (depth <= maxDepth) and (sInfo.bestSplitScore > 0):
            #if (depth <= maxDepth):
                depth += 1
                currentNode.leftBranch = FLTreeNode()
                currentNode.rightBranch = FLTreeNode()

                self.grow(lD, depth,NodeDirection = TreeNodeType.LEFT, currentNode=currentNode.leftBranch)
                self.grow(rD, depth, NodeDirection = TreeNodeType.RIGHT, currentNode=currentNode.rightBranch)
            
                tmpInfo = currentNode.get_string_recursive()
                logger.info("Live Tree Info at depth %d:\n%s", depth, tmpInfo)

            else:
                terNode = self.generate_leaf(qDataBase.gradVec, qDataBase.hessVec, lamb = 0.2)
                currentNode.weight = terNode.weight

                logger.warning("Reached max-depth or Gain is negative. Terminate the tree growing process ...")
                logger.info("Leaf Weight: %f", currentNode.weight)
                #return leafNode
        else:
            logger.warning("Splitting candidate is not feasible. Terminate the tree growing process and generate leaf ...")
            terNode = self.generate_leaf(qDataBase.gradVec, qDataBase.hessVec, lamb = 0.2)
            currentNode.weight = terNode.weight
            logger.info("Leaf Weight: %f", currentNode.weight)
            #return leafNode
              

    # def predict(self, data): # Encapsulated for many data
    #     data_num = data.shape[0]
    #     result = []
    #     for i in range(data_num):
    #         temp_result = self.classify(self.Tree, data[i].reshape(1, -1))
    #         if self.rank == 1:
    #             result.append(temp_result)
    #         else:
    #             pass
    #     result = np.array(result).reshape((-1, 1))
    #     return result


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
                update_pred = self.trees[i].predict(X)
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
    data = pd.read_csv('./GiveMeSomeCredit/cs-training.csv')
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
    #test()
    main4()

except Exception as e:
  logging.error("Exception occurred", exc_info=True)

