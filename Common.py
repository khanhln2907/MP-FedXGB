import numpy as np
from mpi4py import MPI
import logging

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
logger.setLevel(logging.INFO)

logger.warning("Hello World")

class TreeNodeType:
    ROOT = "Root"
    LEFT = "Left"
    RIGHT = "Right"
    LEAF = "Leaf"

class TreeEntity:
    def __init__(self) -> None:
        pass


class TreeNode(TreeEntity):
    def __init__(self) -> None:
        self.weight = 0
        self.leftBranch = None
        self.rightBranch = None


class PARTY_ID:
    ACTIVE_PARTY = 1


class MSG_ID:
    MASKED_GH = 99
    RAW_SPLITTING_MATRIX = 98
    OPTIMAL_SPLITTING_INFO = 97

class SplittingInfo:
    def __init__(self) -> None:
        self.bestSplitScore = -np.Infinity
        self.bestSplitParty = None
        self.bestSplittingVector = None

    def log(self, logger):
        logger.info("Best Splitting Score: L = %.2f, Selected Party %s",\
                self.bestSplitScore, str(self.bestSplitParty))
        logger.debug("The optimal splitting vector: %s", str(self.bestSplittingVector))



# class TreeLeaf(TreeEntity):
#     def __init__(self) -> None:
#         self.weight = 0



# class PARTY_ID:
#     ACTIVE_PARTY = 1


# class MSG_ID:
#     MASKED_GH = 99


# np.random.seed(10)
# clientNum = 4
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()


# logger = logging.getLogger()
# logName = 'Log/FedXGBoost_%d.log' % rank
# file_handler = logging.FileHandler(logName, mode='w')
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)
# logger.setLevel(logging.DEBUG)

# logger.warning("Hello World")


