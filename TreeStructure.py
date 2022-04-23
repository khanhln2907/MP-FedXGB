import numpy as np
import pandas as pd
from datetime import *
from math import ceil, log
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



class FLTreeNode(TreeNode):
    def __init__(self, weight=0, leftBranch=None, rightBranch=None, ownerID = -1):
        super().__init__(weight, leftBranch, rightBranch)
        self.owner = ownerID

    def get_private_info(self):
        return "\nOwner ID:{}".format(self.owner)




class TreeLeaf:
    def __init__(self, weight) -> None:
        self.weight = weight



def draw_tree(nodes: TreeNode, cell_width=2):
    node_width = cell_width + 2
    ws = "·"
    empty_cell = ws * cell_width
    lft_br = f"{(ws * (node_width-2))}╱{ws}"
    rgt_br = f"{ws}╲{(ws * (node_width-2))}"
    cap = f"┌{'─' * cell_width}┐"
    base = f"└{'─' * cell_width}┘"
    space_width = 2

    n = len(nodes)
    nodes = [str(node).zfill(cell_width) for node in nodes]
    tree_height = ceil(log(n + 1, 2))
    max_lvl_nodes = 2 ** (tree_height - 1)
    max_lvl_gaps = max_lvl_nodes - 1
    last_lvl_gap_w = node_width

    max_box_width = (max_lvl_nodes * node_width) + (max_lvl_gaps * last_lvl_gap_w)
    row_width = max_box_width + (space_width * 3)
    rows_n = (tree_height * 4) + 1
    empty_row = ws * (row_width+(cell_width))

    for row_i in range(rows_n):
        if row_i == 0 or row_i == rows_n - 1:
            print(empty_row)
        else:
            lvl_i = row_i // 4
            lvl_slots_n = 2 ** lvl_i
            lvl_gaps_n = lvl_slots_n - 1
            lvl_start = lvl_slots_n - 1
            lvl_stop = (lvl_start * 2) + 1

            lvl_gap_w = (2 ** (tree_height + 2 - lvl_i)) - node_width
            lvl_box_w = (lvl_slots_n * node_width) + (lvl_gaps_n * lvl_gap_w)
            lvl_margin_w = ((row_width - lvl_box_w) // 2) + 1
            lvl_margin = ws * lvl_margin_w
            lvl_gap = ws * lvl_gap_w

            lvl_fill = ""
            if lvl_i + 1 == tree_height:
                max_slots = (2 ** tree_height) - 1
                missing_n = max_slots - n
                lvl_fill_w = (missing_n * node_width) + (missing_n * lvl_gap_w)
                lvl_fill = ws * lvl_fill_w

            lvl_nodes = [f"│{node}│" for node in nodes[lvl_start:lvl_stop]]
            lvl_legend = empty_cell

            if row_i % 4 == 0:
                # branch row
                lvl_nodes = [lft_br if br % 2 == 0 else rgt_br for br in range(len(lvl_nodes))]
            elif (row_i + 1) % 4 == 0:
                # base row
                lvl_nodes = [base for _ in lvl_nodes]
            elif (row_i + 3) % 4 == 0:
                # cap row
                lvl_nodes = [cap for _ in lvl_nodes]
            else:
                lvl_legend = str(lvl_i).zfill(cell_width)

            print(f"{lvl_legend}{lvl_margin[cell_width:]}{lvl_gap.join(lvl_nodes)}{lvl_fill}{lvl_margin}")

