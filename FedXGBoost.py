from posixpath import split
import random
from numpy import concatenate
from VerticalXGBoost import *
from SSCalculation import *
import matplotlib.pyplot as plt
import logging
from DataBaseStructure import *
from TreeStructure import *


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
        self.bestSplitScore = 0
        self.bestSplitParty = None
        self.bestSplittingVector = None

    def log(self, logger):
        logger.info("Best Splitting Score: L = %.2f, Selected Party %d",\
                self.bestSplitScore, self.bestSplitParty)
        logger.debug("The optimal splitting vector: \n %s", str(self.bestSplittingVector))

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

        self.Tree = []

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
        super().fit(y_and_pred, treeID)

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
                data = comm.send(G, dest = partners, tag = MSG_ID.MASKED_GH)
                logger.info("Sent G, H to party %d", partners)         

            # Receive the splitting matrix and find the optimal splitting score
            
            sInfo = SplittingInfo()        
            for partners in range(2, nprocs):   
                rxSM = comm.recv(source = partners, tag = MSG_ID.RAW_SPLITTING_MATRIX)

                # Find the optimal splitting score
                sumGRVec = np.matmul(rxSM, G).reshape(-1)
                sumHRVec = np.matmul(rxSM, H).reshape(-1)
                sumGLVec = sum(G) - sumGRVec
                sumHLVec = sum(H) - sumHRVec
                L = compute_splitting_score(rxSM, G, H, 0.01)

                logger.info("Received SM from party {} and computed:  \n".format(partners) + \
                    "GR: " + str(sumGRVec.T) + "\n" + "HR: " + str(sumHRVec.T) +\
                    "\nGL: " + str(sumGLVec.T) + "\n" + "HL: " + str(sumHLVec.T) +\
                    "\nSplitting Score: {}".format(L.T))       
                maxScore = max(L)
                if maxScore > sInfo.bestSplitScore:
                    sInfo.bestSplitScore = maxScore
                    sInfo.bestSplitParty = partners
                    bestCandidateIndex = np.argmax(L)
                    sInfo.bestSplittingVector = rxSM[bestCandidateIndex, :]

            sInfo.log(logger)
            # Build Tree from the feature with the optimal index
            
            for partners in range(2, nprocs):
                data = comm.send(sInfo, dest = partners, tag = MSG_ID.OPTIMAL_SPLITTING_INFO)
                logger.info("Sent splitting info to clients {} \n".format(partners))

            
        elif rank != 0: # TODO: change this hard coded number
            data = comm.recv(source=PARTY_ID.ACTIVE_PARTY, tag=MSG_ID.MASKED_GH)
            logger.info("Received G, H from the active party")         
            
            # Perform the secure Sharing of the splitting matrix
            qDataBase.printInfo(logger)
            privateSM = qDataBase.get_merged_splitting_matrix()
            logger.debug("Secured splitting matrix with shape of {}".format(str(privateSM.shape)) \
                            + "\n {}".format(' '.join(map(str, privateSM))))
            logger.debug("Secured splitting matrix: \n %s", str(privateSM))


            # Send the splitting matrix to the active party
            txSM = comm.send(privateSM, dest = PARTY_ID.ACTIVE_PARTY, tag = MSG_ID.RAW_SPLITTING_MATRIX)
            logger.info("Sent the splitting matrix to the active party")         

            sInfo = comm.recv(source=PARTY_ID.ACTIVE_PARTY, tag = MSG_ID.OPTIMAL_SPLITTING_INFO)
            logger.info("Received the SplittingInfo from the active party")         


        if(rank != 0):
            # Get the optimal splitting candidates and partition them into two databases
            logger.info("Partition the database for the next build")         
            lD, rD = qDataBase.partion(sInfo.bestSplittingVector)
            qDataBase.printInfo(logger)
            lD.printInfo(logger)
            rD.printInfo(logger)
            

            

    def grow(self, dataBase, depth=1, maxDepth = 5):
        leftBranch = self.buildTree_ver4(shared_gl, shared_hl, shared_sl, depth + 1)
        rightBranch = self.buildTree_ver4(shared_gr, shared_hr, shared_sr, depth + 1)

        leftBranch = TreeNode()
        rightBranch = TreeNode() 

        if depth <= maxDepth:
            return TreeNodes
        else:
            return 
        return 


    #def build()


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
        self.dataBase.printInfo(logger)

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
    X_train_B = X_train[:, 1:4]
    X_train_C = X_train[:, 1].reshape(-1, 1)
    X_train_D = X_train[:, 3].reshape(-1, 1)
    X_test_A = X_test[:, 0].reshape(-1, 1)
    X_test_B = X_test[:, 1:4]
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



np.random.seed(10)
clientNum = 4
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


logger = logging.getLogger()
logName = 'Log/FedXGBoost_%d.log' % rank
file_handler = logging.FileHandler(logName, mode='w')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.setLevel(logging.DEBUG)

logger.warning("Hello World")



#test()


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
    if rank == 1:
        model.appendData(np.concatenate(X_train_A, y_train))
    elif rank == 2:
        model.appendData(np.concatenate(X_train_B, y_train))
    elif rank == 3:
        model.appendData(np.concatenate(X_train_C, y_train))
    elif rank == 4:
        model.appendData(np.concatenate(X_train_D, y_train))
    else:
        model.appendData(np.concatenate(X_train_A, y_train))

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

test()
#main4()

