import numpy as np
import pandas as pd
from mpi4py import MPI
from datetime import *
#from Tree import *
from VerticalXGBoost import VerticalXGBoostClassifier #, VerticalXGBoostTree
from SSCalculation import *
import math
import time

np.random.seed(10)
clientNum = 4

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
def main1():
    print("My Rank ", rank)
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
    model = VerticalXGBoostClassifier(rank=rank, lossfunc='LogLoss', splitclass=splitclass)

    if rank == 1:
        model.fit(X_train_A, y_train)
        print('end 1')
    elif rank == 2:
        model.fit(X_train_B, np.zeros_like(y_train))
        print('end 2')
    elif rank == 3:
        model.fit(X_train_C, np.zeros_like(y_train))
        print('end 3')
    elif rank == 4:
        model.fit(X_train_D, np.zeros_like(y_train))
        print('end 4')
    else:
        model.fit(np.zeros_like(X_train_B), np.zeros_like(y_train))
        print('end 0')

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
        y_ori = y_pred.copy()
        y_pred = 1.0 / (1.0 + np.exp(-y_pred))
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
        result = y_pred - y_test
        print(np.sum(result == 0) / y_pred.shape[0])
        # for i in range(y_test.shape[0]):
        #     print(y_test[i], y_pred[i], y_ori[i])

main1()