import numpy as np
import pandas as pd
from datetime import *
from math import ceil, log
import time

from Common import SplittingInfo, logger

class TreeNode:
    def __init__(self, weight = 0.0, leftBranch=None, rightBranch=None):
        self.weight = weight
        self.leftBranch = leftBranch
        self.rightBranch = rightBranch
        

    def logNode(self):
        logger.info("Child Node Addresses: L %d| R %d", id(self.leftBranch), id(self.rightBranch))

    def get_string_recursive(self):
        str = ""
        if not self.is_leaf():
            str += "[Addr: {} Child L: {} Child R: {} Weight: {}]".format(id(self), id(self.leftBranch), id(self.rightBranch), self.weight)
            str += "{}".format(self.get_private_info())
            str += " \nChild Info \nLeft Node: {} \nRight Node: {}".format(self.leftBranch.get_string_recursive(), self.rightBranch.get_string_recursive())
        else:
            str += "[TreeLeaf| Addr: {} Weight: {}]".format(id(self), self.weight)
        return str

    def get_private_info(self):
        return


    def is_leaf(self):
        return (self.leftBranch is None) and (self.rightBranch is None)

class FLTreeNode(TreeNode):
    def __init__(self, FID = 0, weight=0, nUsers = 0, leftBranch=None, rightBranch=None, ownerID = -1):
        super().__init__(weight, leftBranch, rightBranch)
        self.FID = FID
        self.owner = ownerID
        self.splittingInfo = SplittingInfo()
        self.nUsers = nUsers
        
    def get_private_info(self):
        return "\nOwner ID:{}".format(self.owner)

    def set_splitting_info(self, sInfo: SplittingInfo):
        self.owner = sInfo.bestSplitParty
        self.splittingInfo = sInfo

        
