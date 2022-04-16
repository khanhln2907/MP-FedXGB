from Tree import VerticalXGBoostTree
from Common import *


class VerticalFedXGBoostTree(VerticalXGBoostTree):
    def __init__(self, rank, lossfunc, splitclass, _lambda, _gamma, _epsilon, _maxdepth, clientNum):
        super().__init__(rank, lossfunc, splitclass, _lambda, _gamma, _epsilon, _maxdepth, clientNum)

        self.Tree = []

    def fit(self, y_and_pred, tree_num, xData, sQuantile):
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
            data = comm.recv(source=1, tag=MSG_ID.MASKED_GH)
            logger.info("Received G, H")         
            
            # Perform the secure Sharing of the splitting matrix
            # for featureID in range(len(sQuantile[0])):
            #     splitMat = FedXGBoostSecureHandler.generate_splitting_matrix(xData, sQuantile[0])
            if rank == 2:
                #print(xData)
                pass

