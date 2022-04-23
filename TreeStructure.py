import numpy as np
import pandas as pd
from datetime import *
import math
import time

from Common import logger

class TreeNode:
    def __init__(self, weight = 0, leftBranch=None, rightBranch=None):
        self.weight = weight
        self.leftBranch = leftBranch
        self.rightBranch = rightBranch

    def logNode(self):
        logger.info("Child Node Addresses: L %d| R %d", id(self.leftBranch), id(self.rightBranch))

    def get_string_recursive(self):
        str = ""
        if(self.leftBranch is not None) and (self.rightBranch is not None):
            str += "[Addr: {} Child L: {} Child R: {} Weight: {}]".format(id(self), id(self.leftBranch), id(self.rightBranch), self.weight)
            str += "{}".format(self.get_private_info())
            str += " \nChild Info \nLeft Node: {} \nRight Node: {}".format(self.leftBranch.get_string_recursive(), self.rightBranch.get_string_recursive())
        else:
            str += "[TreeLeaf| Addr: {} Weight: {}]".format(id(self), self.weight)
        return str

    def get_private_info(self):
        return

    def show_tree_structure(self):



        pass

class FLTreeNode(TreeNode):
    def __init__(self, weight=0, leftBranch=None, rightBranch=None, ownerID = -1):
        super().__init__(weight, leftBranch, rightBranch)
        self.owner = ownerID

    def get_private_info(self):
        return "\nOwner ID:{}".format(self.owner)




class TreeLeaf:
    def __init__(self, weight) -> None:
        self.weight = weight


