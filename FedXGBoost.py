from numpy import concatenate
from VerticalXGBoost import *
from SSCalculation import *
import matplotlib.pyplot as plt
import logging

class PARTY_ID:
    ACTIVE_PARTY = 1


class MSG_ID:
    MASKED_GH = 99

class VerticalFedXGBoostTree(VerticalXGBoostTree):
    def __init__(self, rank, lossfunc, splitclass, _lambda, _gamma, _epsilon, _maxdepth, clientNum):
        super().__init__(rank, lossfunc, splitclass, _lambda, _gamma, _epsilon, _maxdepth, clientNum)

        self.Tree = []

    def fit(self, y_and_pred, tree_num):
        super().fit(y_and_pred, tree_num)

        # size = None
        # size_list = comm.gather(self.data.shape[1], root=2)  # Gather all the feature size.
        # if self.rank == 2:
        #     size = sum(size_list[1:])
        # self.featureNum = comm.bcast(size, root=2)  # Broadcast how many feature there are in total.
        # if self.rank == 2:
        #     #print('DCM, rank: ', self.rank)
        #     random_list = np.random.permutation(self.featureNum)
        #     start = 0
        #     for i in range(1, clientNum + 1):
        #         rand = random_list[start:start + size_list[i]]
        #         if i == 2:
        #             self.featureList = rand
        #         else:
        #             comm.send(rand, dest=i)  # Send random_list to all the client, mask their feature index.
        #         start += size_list[i]
        # elif self.rank != 0:    
        #     self.featureList = comm.recv(source=2)
        # self.setMapping()
        # shared_G, shared_H, shared_S = None, None, None


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
            #data = comm.bcast(gh, root = 1)
            #data = comm.bcast(H, root = 1)
            

        elif rank != 0: # TODO: change hard coded number
        #else:
            data = comm.recv(source=1, tag=MSG_ID.MASKED_GH)
            logger.info("Received G, H")         
            # Perform the secure Sharing of the splitting matrix


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

    def appendData(self, X, y):
        data_num = X.shape[0]
        self.label = np.reshape(y, (data_num, 1))
        self.data = X.copy()

        self.getAllQuantile()
        logger.warning('Rank %d appended data', self.rank)

    def boost(self):
        data_num = self.data.shape[0]
        X = self.data
        y = self.label
        y_pred = np.zeros(np.shape(self.label))
        for i in range(self.n_estimators):
            if self.rank == 1:
                logger.info("Iter: %d. Amount of splitting candidates: %d", i, self.maxSplitNum)
                for key, value in self.quantile.items():
                    logger.info("Quantile %d: ", key)
                    logger.info("{}".format(' '.join(map(str, value))))
            
            tree = self.trees[i]
            tree.data, tree.maxSplitNum, tree.quantile = self.data, self.maxSplitNum, self.quantile
            y_and_pred = np.concatenate((y, y_pred), axis=1)

            # Perform tree boosting
            tree.fit(y_and_pred, i)
            if i == self.n_estimators - 1: # The last tree, no need for prediction update.
                continue
            else:
                update_pred = tree.predict(X)
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

    if rank == 1:
        model.appendData(X_train_A, y_train)
    elif rank == 2:
        model.appendData(X_train_B, y_train)
    elif rank == 3:
        model.appendData(X_train_C, y_train)
    elif rank == 4:
        model.appendData(X_train_D, y_train)
    else:
        model.appendData(X_train_A, y_train)

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


test()
 