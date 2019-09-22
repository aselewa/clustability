import sys
import numpy as np

class clustability(object):
    '''
    Class representing cluster stability calculations
    '''

    def __init__(self, labelMatrix):

        self.n, self.k = labelMatrix.shape
        self.labelMatrix = labelMatrix

    def getStability(self):
        '''
        Computes the stability metric as well as other cluster features
        '''

        stability = []
        nClustList = []
        resList = []
        clustSizeList = []
        
        for i in range(self.k):
            
            nClusts = np.max(self.labelMatrix[:,i]) + 1
            currResScore = np.zeros(nClusts)
            clustSizeList.append(self.getIntersect(i,i).sum(axis=1))
            
            for j in range(self.k):
                if i != j:
                    shareMat = self.getIntersect(i,j)
                    c_j = shareMat.sum(axis=0)
                    N_l = (shareMat>0).sum(axis=1)
    
                    currResScore += (((shareMat/c_j).T/(N_l**2)).T).sum(axis=1)
            currResScore *= 1/self.k

            stability.append(currResScore)
            nClustList.append(np.arange(nClusts))
            resList.append(np.repeat(i, nClusts))
        
        stabilityMat = np.column_stack((np.concatenate(stability),
                                        np.concatenate(nClustList), 
                                        np.concatenate(resList), 
                                        np.concatenate(clustSizeList)))
        
        return stabilityMat

    def getIntersect(self, res1, res2):
        '''
        Helper function for evaluating intersection of clusters at different resolutions
        '''
        iVals = self.labelMatrix[:,res1]
        jVals = self.labelMatrix[:,res2]

        nClust = np.max(iVals) + 1            #clusters start at 0
        nClust_other = np.max(jVals) + 1

        shareMat = np.zeros((nClust, nClust_other))
        for i in range(nClust):
            for j in range(nClust_other):
                cellsInBoth = np.intersect1d(np.where(iVals == i), np.where(jVals == j))
                shareMat[i,j] = len(cellsInBoth)
       
        return shareMat
        




            
