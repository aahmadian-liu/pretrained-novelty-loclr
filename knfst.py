"""
 Kernel Null Foley-Sammon transform (KNFST) method (used for implementing local KNFST novelty detection)
"""

# Re-implemented based on the MATLAB code in https://github.com/cvjena/knfst/tree/master 

import numpy as np
from numpy.linalg import eig
from scipy.linalg import null_space


def calculateKNFST(K, labels):

    n,m = K.shape

    maxLabel = np.max(labels)
    minLabel = np.min(labels)
 
    centeredK = centerKernelMatrix(K)

    basisvecsValues,basisvecs = eig(centeredK)
    
    basisvecs = basisvecs[:,basisvecsValues > 1e-12]
    basisvecsValues = basisvecsValues[basisvecsValues > 1e-12]
    basisvecsValues = np.diag(1./np.sqrt(basisvecsValues))
    basisvecs = basisvecs@basisvecsValues
      
    L = np.zeros([n,n])
    for i in range(minLabel,maxLabel+1):
       sel=[[ ( (labels[i1]==i) & (labels[i2]==i) ) for i1 in range(m)] for i2 in range(n)]
       sel=np.array(sel)
       L[sel] = 1/np.sum(labels==i)

    M = np.ones([m,m])/m
    
    H = ((np.eye(M.shape[0],M.shape[1])-M)@basisvecs).T @ K @ ((np.eye(K.shape[0],K.shape[1]))-L)
    
    T = H @ H.T
    
    eigenvecs = null_space(T)
    if eigenvecs.shape[1] < 1:
      eigenvals,eigenvecs= eig(T)
      min_ID = np.argmin(eigenvals)
      eigenvecs = eigenvecs[:,min_ID]
    
    proj = ((np.eye(M.shape[0],M.shape[1])-M)@basisvecs)@eigenvecs

    return proj

def centerKernelMatrix(kernelMatrix):
  
    n = kernelMatrix.shape[0]

    columnMeans = np.mean(kernelMatrix,axis=0)
    matrixMean = np.mean(columnMeans)

    centeredKernelMatrix = kernelMatrix.copy()

    for k in range(0,n):
        centeredKernelMatrix[k,:] = centeredKernelMatrix[k,:] - columnMeans
        centeredKernelMatrix[:,k] = centeredKernelMatrix[:,k] - columnMeans.T

    return centeredKernelMatrix+matrixMean

def learn_oneClassNovelty_knfst(K):
    
    n = K.shape[0]

    K = np.block([ [K, np.zeros([n,1])], [np.zeros([1,n+1])] ])
    
    labels = np.block([ [np.ones([n,1])],[0] ])
    labels=labels.astype(int)
    labels=labels[:,0]

    proj = calculateKNFST(K,labels)
    targetValue = np.mean(K[labels==1,:] @ proj)
    proj=proj[0:n]

    return proj,targetValue

def test_oneClassNovelty_knfst(proj,targetValue, Ks):
# Ks: (n x m) kernel matrix containing similarities between n training samples and m test samples
    projectionVectors = Ks.T@proj
    
    diff = projectionVectors-np.ones([Ks.shape[1],1])*targetValue

    scores = np.sqrt(np.sum(diff*diff,1))

    return scores