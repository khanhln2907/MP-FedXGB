import numpy as np
import pandas as pd
from datetime import *
import math
import time


class TreeNode:
    def __init__(self, leftBranch=None, rightBranch=None):
        self.leftBranch = leftBranch
        self.rightBranch = rightBranch

class TreeLeaf:
    def __init__(self, weight) -> None:
        self.weight = weight


