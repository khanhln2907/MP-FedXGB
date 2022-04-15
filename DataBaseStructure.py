import numpy as np
from scipy import rand


class QuantileParam:
    epsilon = 0.02

class FeatureData:
    def __init__(self, name, dataVector) -> None:
        self.name = name
        self.data = np.array(dataVector)

class QuantiledFeature(FeatureData):
  
    def __init__(self, name, dataVector) -> None:
        super().__init__(name, dataVector)
        self.splittingMatrix, self.splittingCandidates = QuantiledFeature.quantile(self.data, QuantileParam)
        


    def quantile(fData: FeatureData, param: QuantileParam):
        splittingMatrix = []
        splittingCandidates = []

        # Credits to HikariX ... copy for convenience refactoring ...
        split_list = []
        data = np.copy(fData.data)
        idx = np.argsort(data)
        data = data[idx]
        value_list = sorted(list(set(list(data))))  # Record all the different value
        hess = np.ones_like(data)
        data = np.concatenate((data, hess), axis=0)
        sum_hess = np.sum(hess)
        last = value_list[0]
        i = 1
        if len(value_list) == 1: # For those who has only one value, do such process.
            last_cursor = last
        else:
            last_cursor = value_list[1]
        split_list.append((-np.inf, value_list[0]))
        while i < len(value_list):
            cursor = value_list[i]
            small_hess = np.sum(data[data <= last]) / sum_hess
            big_hess = np.sum(data[data <= cursor]) / sum_hess
            if np.abs(big_hess - small_hess) < param.epsilon:
                last_cursor = cursor
            else:
                judge = value_list.index(cursor) - value_list.index(last)
                if judge == 1: # Although it didn't satisfy the criterion, it has no more split, so we must add it.
                    split_list.append((last, cursor))
                    last = cursor
                else: # Move forward and record the last.
                    split_list.append((last, last_cursor))
                    last = last_cursor
                    last_cursor = cursor
            i += 1
        if split_list[-1][1] != value_list[-1]:
            split_list.append((split_list[-1][1], value_list[-1]))  # Add the top value into split_list.
        split_list = np.array(split_list)


        # Khanh goes on from here, I need the splittingCandidates
        splittingCandidates = [split_list[i][1] for i in range(len(split_list))]
        splittingMatrix = QuantiledFeature.generate_splitting_matrix(fData.data, splittingCandidates)

        # import matplotlib.pyplot as plt 
        # print(split_list)
        # plt.hist(fData.data)
        # plt.hist(sm[6])
        # plt.show()

        return splittingMatrix, splittingCandidates


        return splittingMatrix

    def generate_splitting_matrix(dataVector, splittingCandidates):
        """
        assign 0 if the data value is smaller than the splitting candidates (left node)
        assign 1 if the data value is bigger than the splitting candidates (left node)
        """
        splittingMatrix = []
        for i in range(len(splittingCandidates)):
            v =  1. * (dataVector > splittingCandidates[i])
            splittingMatrix.append(v)

        return np.array(splittingMatrix)


class DataBase:
    def __init__(self) -> None:
        self.data = []

    def appendFeature(featureData: FeatureData):
        pass


def testQuantile():
    vec = rand(100)
    handle = QuantiledFeature("RandomData", vec)
    print("Amount of Splitting Cadndidates: {}" .format(len(handle.splittingCandidates)))

np.random.seed(66)
testQuantile()