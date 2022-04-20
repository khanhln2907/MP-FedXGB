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
logger.setLevel(logging.DEBUG)

logger.warning("Hello World")

class TreeNodeType:
    ROOT = "Root"
    LEFT = "Left"
    RIGHT = "Right"


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


