#Purpose: Code that students fill in to implement Procrustes Alignment
#and the Iterative Closest Points Algorithm
import numpy as np

#Purpose: To compute the centroid of a point cloud
#Inputs:
#PC: 3 x N matrix of points in a point cloud
#Returns: A 3 x 1 matrix of the centroid of the point cloud
def getCentroid(PC):
    # mean of column vectors (axis 1) 
    return np.mean(PC, 1)[:, np.newaxis] 

#Purpose: Given an estimate of the aligning matrix Rx that aligns
#X to Y, as well as the centroids of those two point clouds, to
#find the nearest neighbors of X to points in Y
#Inputs:
#X: 3 x M matrix of points in X
#Y: 3 x N matrix of points in Y (the target point cloud)
#Cx: 3 x 1 matrix of the centroid of X
#Cy: 3 x 1 matrix of the centroid of corresponding points in Y
#Rx: Current estimate of rotation matrix for X
#Returns:
#idx: An array of size N which stores the indices 
def getCorrespondences(X, Y, Cx, Cy, Rx):
    #TODO: Update this: Right now everything maps to the first point in Y
    return np.zeros(X.shape[1], dtype=np.int64) #dummy value

#Purpose: Given correspondences between two point clouds, to center
#them on their centroids and compute the Procrustes alignment to
#align one to the other
#Inputs:
#X: 3 x M matrix of points in X
#Y: 3 x N matrix of points in Y (the target point cloud)
#Returns:
#A Tuple (Cx, Cy, Rx):
#Cx: 3 x 1 matrix of the centroid of X
#Cy: 3 x 1 matrix of the centroid of corresponding points in Y
#Rx: A 3x3 rotation matrix to rotate and align X to Y after
#they have both been centered on their centroids Cx and Cy
def getProcrustesAlignment(X, Y, idx):
    #TODO: Finish this: These are all dummy values
    Cx = np.zeros((3, 1))
    Cy = np.zeros((3, 1))
    R = np.eye(3) #3x3 Identity matrix (trivial rotation...needs to be updated)
    return (Cx, Cy, R)    

#Purpose: To implement the loop which ties together correspondence finding
#and procrustes alignment to implment the interative closest points algorithm
#Do until convergence (i.e. as long as the correspondences haven't changed)
#Inputs:
#X: 3 x M matrix of points in X
#Y: 3 x N matrix of points in Y (the target point cloud)
#MaxIters: Maximum number of iterations to perform, regardless of convergence
#Returns: A tuple of (CxList, CyList, RxList):
#CxList: A list of centroids of X estimated in each iteration (these
#should actually be the same each time)
#CyList: A list of the centroids of corresponding points in Y at each 
#iteration (these might be different as correspondences change)
#RxList: A list of rotations matrices Rx that are produced at each iteration
#This is all of the information needed to animate exactly
#what the ICP algorithm did
def doICP(X, Y, MaxIters):
    CxList = [np.zeros((3, 1))]
    CyList = [np.zeros((3, 1))]
    RxList = [np.eye(3)]
    #TODO: Fill the rest of this in    
    return (CxList, CyList, RxList)
