import numpy as np
from Common import logger, rank, comm, PARTY_ID, MSG_ID, TreeNodeType, SplittingInfo
from VerticalXGBoost import VerticalXGBoostTree
from TreeStructure import *
from DataBaseStructure import *
from TreeRender import FLVisNode

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



class VerticalFedXGBoostTree(VerticalXGBoostTree):
    def __init__(self, rank, lossfunc, splitclass, _lambda, _gamma, _epsilon, _maxdepth, clientNum):
        super().__init__(rank, lossfunc, splitclass, _lambda, _gamma, _epsilon, _maxdepth, clientNum)

        self.root = FLTreeNode()

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
        logger.info("Tree is growing column-wise. Current column: %d", treeID)

        super().fit(y_and_pred, treeID)

        """
        This function computes the gradient and the hessian vectors to perform the tree construction
        """
        # Compute the gradients and hessians
        if self.rank == PARTY_ID.ACTIVE_PARTY: # Calculate gradients on the node who have labels.
            y, y_pred = self._split(y_and_pred)
            G = np.array(self.loss.gradient(y, y_pred)).reshape(-1)
            H = np.array(self.loss.hess(y, y_pred)).reshape(-1)
            logger.info("Computed Gradients and Hessians ")
            logger.debug("G {}".format(' '.join(map(str, G))))
            logger.debug("H {}".format(' '.join(map(str, H))))

            # nprocs = comm.Get_size()
            # for partners in range(2, nprocs):   
            #     data = comm.send(G, dest = partners, tag = MSG_ID.MASKED_GH)
            #     logger.info("Sent G, H to party %d", partners)         

            qDataBase.appendGradientsHessian(G, H) 

        else:
            # data = comm.recv(source=PARTY_ID.ACTIVE_PARTY, tag=MSG_ID.MASKED_GH)
            # logger.info("Received G, H from the active party")

            dummyG = np.zeros((qDataBase.nUsers,1))
            dummyH = np.zeros((qDataBase.nUsers,1))
            qDataBase.appendGradientsHessian(dummyG, dummyH)

        if(rank != 0):
            rootNode = FLTreeNode()
            self.grow(qDataBase, depth = 0, NodeDirection = TreeNodeType.ROOT, currentNode = rootNode)
            self.root = rootNode

            #if((rank == 1)):
            treeInfo = self.root.get_string_recursive()
            logger.info("Tree Info:\n%s", treeInfo)
            #print(treeInfo)
        
            b = FLVisNode(self.root)
            b.display(treeID)

    def generate_leaf(self, gVec, hVec, lamb = 0.1):
        gI = sum(gVec) 
        hI = sum(hVec)
        ret = TreeNode(-1.0 * gI / (hI + lamb), leftBranch= None, rightBranch= None)
        return ret

    def grow(self, qDataBase: QuantiledDataBase, depth=0, NodeDirection = TreeNodeType.ROOT, currentNode : FLTreeNode = None):
        logger.info("Tree is growing depth-wise. Current depth: {}".format(depth) + " Node's type: {}".format(NodeDirection))

        if self.rank == PARTY_ID.ACTIVE_PARTY:
            sInfo = SplittingInfo()
            nprocs = comm.Get_size()        
            for partners in range(2, nprocs):   
                rxSM = comm.recv(source = partners, tag = MSG_ID.RAW_SPLITTING_MATRIX)

                # Find the optimal splitting score
                sumGRVec = np.matmul(rxSM, qDataBase.gradVec).reshape(-1)
                sumHRVec = np.matmul(rxSM, qDataBase.hessVec).reshape(-1)
                sumGLVec = sum(qDataBase.gradVec) - sumGRVec
                sumHLVec = sum(qDataBase.hessVec) - sumHRVec
                L = compute_splitting_score(rxSM, qDataBase.gradVec, qDataBase.hessVec, 0.01)

                logger.debug("Received SM from party {} and computed:  \n".format(partners) + \
                    "GR: " + str(sumGRVec.T) + "\n" + "HR: " + str(sumHRVec.T) +\
                    "\nGL: " + str(sumGLVec.T) + "\n" + "HL: " + str(sumHLVec.T) +\
                    "\nSplitting Score: {}".format(L.T))       
                
                # Optimal candidate of 1 partner party
                # Select the optimal candidates without all zeros or one elements of the splitting)
                isValid = False
                excId = np.zeros(L.size, dtype=bool)
                for id in range(len(L)):
                    splitVector = rxSM[id, :]

                    nL = np.count_nonzero(splitVector == 0.0)
                    nR = np.count_nonzero(splitVector == 1.0)
                    #print(len(splitVector), nR, nL)
                    thres = 0.2
                    isValid = ((nL/len(splitVector)) > thres) and ((nR/len(splitVector)) > thres)
                    if isValid:
                        pass
                    else:
                        excId[id] = True

                bestSplitId = 0
                tmpL = np.ma.array(L, mask=excId) # Mask the exception index
                bestSplitId = np.argmax(tmpL)
                splitVector = rxSM[bestSplitId, :]
                maxScore = L[bestSplitId]     
                # Select the optimal over all partner parties
                if (maxScore > sInfo.bestSplitScore):
                    sInfo.bestSplitScore = maxScore
                    sInfo.bestSplitParty = partners
                    sInfo.selectedCandidate = bestSplitId
                    sInfo.bestSplittingVector = rxSM[bestSplitId, :]
                    
            # Log the splitting info
            sInfo.log(logger)
            
            # Build Tree from the feature with the optimal index
            for partners in range(2, nprocs):
                    data = comm.send(sInfo, dest = partners, tag = MSG_ID.OPTIMAL_SPLITTING_INFO)
                    logger.info("Sent splitting info to clients {}".format(partners))
        elif (rank != 0):           
            # Perform the secure Sharing of the splitting matrix
            # qDataBase.printInfo()
            privateSM = qDataBase.get_merged_splitting_matrix()
            logger.debug("Secured splitting matrix with shape of {}".format(str(privateSM.shape)) \
                            + "\n {}".format(' '.join(map(str, privateSM))))
            logger.debug("Secured splitting matrix: \n %s", str(privateSM))


            # Send the splitting matrix to the active party
            txSM = comm.send(privateSM, dest = PARTY_ID.ACTIVE_PARTY, tag = MSG_ID.RAW_SPLITTING_MATRIX)
            logger.info("Sent the splitting matrix to the active party")         

            sInfo = comm.recv(source=PARTY_ID.ACTIVE_PARTY, tag = MSG_ID.OPTIMAL_SPLITTING_INFO)
            logger.info("Received the Splitting Info from the active party")   

        # Set the optimal split as the owner ID of the current tree node
        # If the selected party is me
        if(rank == sInfo.bestSplitParty):
            feature, value = qDataBase.find_fId_and_scId(sInfo.bestSplittingVector)
            sInfo.featureName = feature
            sInfo.splitValue = value
            currentNode.set_splitting_info(sInfo)

            # Remove the feature for the next iteration because this is already used
            #qDataBase.remove_feature(feature)
        else:
            currentNode.set_splitting_info(sInfo)
        
        logger.info("Optimal splitting: %s", str(sInfo.bestSplittingVector))

        # Get the optimal splitting candidates and partition them into two databases
        if(sInfo.bestSplittingVector is not None):
            lD, rD = qDataBase.partition(sInfo.bestSplittingVector)
            logger.info("")
            logger.info("Database is partitioned into two quantiled databases!")
            logger.info("Original database: %s", qDataBase.get_info_string())
            logger.info("Left splitted database: %s", lD.get_info_string())
            logger.info("Right splitted database: %s \n", rD.get_info_string())

            maxDepth = 3
            # Construct the new tree if the gain is positive
            if (depth <= maxDepth) and (sInfo.bestSplitScore > 0):
            #if (depth <= maxDepth):
                depth += 1
                currentNode.leftBranch = FLTreeNode()
                currentNode.rightBranch = FLTreeNode()

                self.grow(lD, depth,NodeDirection = TreeNodeType.LEFT, currentNode=currentNode.leftBranch)
                self.grow(rD, depth, NodeDirection = TreeNodeType.RIGHT, currentNode=currentNode.rightBranch)
            
                tmpInfo = currentNode.get_string_recursive()
                logger.info("Live Tree Info at depth %d:\n%s", depth, tmpInfo)

            else:
                endNode = self.generate_leaf(qDataBase.gradVec, qDataBase.hessVec, lamb = 0.2)
                currentNode.weight = endNode.weight

                logger.warning("Reached max-depth or Gain is negative. Terminate the tree growing process ...")
                logger.info("Leaf Weight: %f", currentNode.weight)
                #return leafNode
        else:
            logger.warning("Splitting candidate is not feasible. Terminate the tree growing process and generate leaf ...")
            endNode = self.generate_leaf(qDataBase.gradVec, qDataBase.hessVec, lamb = 0.2)
            currentNode.weight = endNode.weight
            logger.info("Leaf Weight: %f", currentNode.weight)
            #return leafNode
              

    def predictTrain(self, data): # Encapsulated for many data
        """
        Data matrix has the same format as the data appended to the database, includes the features' values
        
        """
        #result = super().predict(data)

        # Perform prediction for users with [idUser] --> [left, right, nextParty]
        data_num = data.shape[0]
        result = []
        for i in range(data_num):
            temp_result = self.classify(self.Tree, data[i].reshape(1, -1))
            if self.rank == 1:
                result.append(temp_result)
            else:
                pass
        result = np.array(result).reshape((-1, 1))

        ## Khanh goes from here
        if rank != 0:
            
            nUsers = data.shape[0]
            for i in range(nUsers):
                #self.classify_fed(i, data[i])
        
                pass

        #print(result)
        return result

    def classify_fed(self, userId, data):
        """
        This method performs the secured federated inferrence
        """

        logger.info("Classifying data of user %d. Data: %s", userId, str(data))

        if rank is PARTY_ID.ACTIVE_PARTY:
            curNode = self.root
            #while(curNode.leftBranch)
            # Iterate until we find the right leaf node
            depth = 0
            while(not curNode.is_leaf()):
                
                # Federate finding the direction for the next node
                partnerID = curNode.owner
                # Request the direction from the partner
                txRecordID = comm.send(depth, dest = curNode.owner, tag = MSG_ID.REQUEST_DIRECTION)
                logger.info("Sent the splitting matrix to the active party")


                curNode = curNode.rightBranch

            
            #print("Leaf weight", curNode.weight)
            selParty = self.root.owner



            pass
        elif rank != 0:
            pass




    def classify(self, tree, data):
        #print(data.shape)

        idx_list = []
        shared_idx = None
        final_result = 0
        if self.rank != 0:
            idx, result = self.getInfo(tree, data)
        

        # for i in range(1, clientNum + 1):
        #     if self.rank == i:
        #         shared_idx = self.split.SSSplit(idx, clientNum)
        #         temp = np.zeros_like(shared_idx[0])
        #         temp = np.expand_dims(temp, axis=0)
        #         shared_idx = np.concatenate([temp, shared_idx], axis=0)
        #     shared_idx = comm.scatter(shared_idx, root=i)
        #     idx_list.append(shared_idx)

        # final_idx = idx_list[0]
        # for i in range(1, clientNum):
        #     final_idx = self.split.SMUL(final_idx, idx_list[i], self.rank)
        # if self.rank == 0:
        #     result = np.zeros_like(final_idx)
        # temp_result = np.sum(self.split.SMUL(final_idx, result, self.rank))
        # temp_result = comm.gather(temp_result, root=1)
        # if self.rank == 1:
        #     final_result = np.sum(temp_result[1:])
        return super().classify(tree, data)





# class FedXGBoostSecureHandler:
#     QR = []
#     pass

#     def generate_secure_kernel(mat):
#         import scipy.linalg
#         return scipy.linalg.qr(mat)

#     def calc_secure_response(privateMat, rxKernelMat):
#         n = len(rxKernelMat) # n users 
#         r = len(rxKernelMat[0]) # number of kernel vectors
#         Z = rxKernelMat
#         return np.matmul((np.identity(n) - np.matmul(Z, np.transpose(Z))), privateMat)

#     def generate_splitting_matrix(dataVector, quantileBin):
#         n = len(dataVector) # Rows as n users
#         l = len(quantileBin) # amount of splitting candidates

#         retSplittingMat = []
#         for candidateIter in range(l):
#             v = np.zeros(n)
#             for userIter in range(n):
#                 v[userIter] = (dataVector[userIter] > max(quantileBin[candidateIter]))  
#             retSplittingMat.append(v)

#         return retSplittingMat


