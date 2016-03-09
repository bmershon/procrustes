from PolyMesh import *
from Primitives3D import *
from OpenGL.GL import *
import sys
import re
import math
import time
import numpy as np
from scipy import sparse
import scipy.io as sio
from scipy.sparse.linalg import lsqr, cg, eigsh
import matplotlib.pyplot as plt

import os
this_path = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(this_path + '/ext/lib') and os.path.exists(this_path + '/ext/libigl/python'):
    sys.path.insert(0, this_path + '/ext/libigl/python/')
    sys.path.insert(0, this_path + '/ext/lib/')
    print "Importing IGL"
    import igl

#Quickly compute sparse Laplacian matrix with cotangent weights and Voronoi areas
#by doing many operations in parallel using NumPy
#VPos: N x 3 array of vertex positions
#ITris: M x 3 array of triangle indices
#anchorsIdx: List of anchor positions
def makeLaplacianMatrixCotWeights(VPos, ITris, anchorsIdx, anchorWeights = 1):
    N = VPos.shape[0]
    M = ITris.shape[0]
    #Allocate space for the sparse array storage, with 2 entries for every
    #edge for every triangle (6 entries per triangle); one entry for directed 
    #edge ij and ji.  Note that this means that edges with two incident triangles
    #will have two entries per directed edge, but sparse array will sum them 
    I = np.zeros(M*6)
    J = np.zeros(M*6)
    V = np.zeros(M*6)
    
    #Keep track of areas of incident triangles and the number of incident triangles
    IA = np.zeros(M*3)
    VA = np.zeros(M*3) #Incident areas
    VC = 1.0*np.ones(M*3) #Number of incident triangles
    
    #Step 1: Compute cotangent weights
    for shift in range(3): 
        #For all 3 shifts of the roles of triangle vertices
        #to compute different cotangent weights
        [i, j, k] = [shift, (shift+1)%3, (shift+2)%3]
        dV1 = VPos[ITris[:, i], :] - VPos[ITris[:, k], :]
        dV2 = VPos[ITris[:, j], :] - VPos[ITris[:, k], :]
        Normal = np.cross(dV1, dV2)
        #Cotangent is dot product / mag cross product
        NMag = np.sqrt(np.sum(Normal**2, 1))
        cotAlpha = np.sum(dV1*dV2, 1)/NMag
        I[shift*M*2:shift*M*2+M] = ITris[:, i]
        J[shift*M*2:shift*M*2+M] = ITris[:, j] 
        V[shift*M*2:shift*M*2+M] = cotAlpha
        I[shift*M*2+M:shift*M*2+2*M] = ITris[:, j]
        J[shift*M*2+M:shift*M*2+2*M] = ITris[:, i] 
        V[shift*M*2+M:shift*M*2+2*M] = cotAlpha
        if shift == 0:
            #Compute contribution of this triangle to each of the vertices
            for k in range(3):
                IA[k*M:(k+1)*M] = ITris[:, k]
                VA[k*M:(k+1)*M] = 0.5*NMag
    
    #Step 2: Create laplacian matrix
    L = sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()
    #Create the diagonal by summing the rows and subtracting off the nondiagonal entries
    L = sparse.dia_matrix((L.sum(1).flatten(), 0), L.shape) - L
    #Scale each row by the incident areas
    Areas = sparse.coo_matrix((VA, (IA, IA)), shape=(N, N)).tocsr()
    Areas = Areas.todia().data.flatten()
    Counts = sparse.coo_matrix((VC, (IA, IA)), shape=(N, N)).tocsr()
    Counts = Counts.todia().data.flatten()
    RowScale = sparse.dia_matrix((3*Counts/Areas, 0), L.shape)
    L = L.T.dot(RowScale).T
    
    #Step 3: Add anchors
    L = L.tocoo()
    I = L.row.tolist()
    J = L.col.tolist()
    V = L.data.tolist()
    I = I + range(N, N+len(anchorsIdx))
    J = J + anchorsIdx
    V = V + [anchorWeights]*len(anchorsIdx)
    L = sparse.coo_matrix((V, (I, J)), shape=(N+len(anchorsIdx), N)).tocsr()
    return L

#Use simple umbrella weights instead of cotangent weights
#VPos: N x 3 array of vertex positions
#ITris: M x 3 array of triangle indices
#anchorsIdx: List of anchor positions
def makeLaplacianMatrixUmbrellaWeights(VPos, ITris, anchorsIdx, anchorWeights = 1):
    N = VPos.shape[0]
    M = ITris.shape[0]
    I = np.zeros(M*6)
    J = np.zeros(M*6)
    V = np.ones(M*6)
    
    #Step 1: Set up umbrella entries
    for shift in range(3): 
        #For all 3 shifts of the roles of triangle vertices
        #to compute different cotangent weights
        [i, j, k] = [shift, (shift+1)%3, (shift+2)%3]
        I[shift*M*2:shift*M*2+M] = ITris[:, i]
        J[shift*M*2:shift*M*2+M] = ITris[:, j] 
        I[shift*M*2+M:shift*M*2+2*M] = ITris[:, j]
        J[shift*M*2+M:shift*M*2+2*M] = ITris[:, i] 
    
    #Step 2: Create laplacian matrix
    L = sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()
    L[L > 0] = 1
    #Create the diagonal by summing the rows and subtracting off the nondiagonal entries
    L = sparse.dia_matrix((L.sum(1).flatten(), 0), L.shape) - L
    
    #Step 3: Add anchors
    L = L.tocoo()
    I = L.row.tolist()
    J = L.col.tolist()
    V = L.data.tolist()
    I = I + range(N, N+len(anchorsIdx))
    J = J + anchorsIdx
    V = V + [anchorWeights]*len(anchorsIdx)
    L = sparse.coo_matrix((V, (I, J)), shape=(N+len(anchorsIdx), N)).tocsr()
    return L

def solveLaplacianMatrix(L, deltaCoords, anchors, anchorWeights = 1):
    y = np.concatenate((deltaCoords, anchorWeights*anchors), 0)
    y = np.array(y, np.float64)
    coo = L.tocoo()
    coo = np.vstack((coo.row, coo.col, coo.data)).T
    coo = igl.eigen.MatrixXd(np.array(coo, dtype=np.float64))
    LE = igl.eigen.SparseMatrixd()
    LE.fromCOO(coo)
    Q = LE.transpose()*LE
    start_time = time.time()
    #solver = igl.eigen.SimplicialLLTsparse(Q)
    solver = igl.eigen.CholmodSupernodalLLT(Q)
    ret = solver.solve(igl.eigen.MatrixXd(y))
    end_time = time.time()
    print 'factorization elapsed time:',end_time-start_time,'seconds'
    return np.array(ret)

#Make a QP solver with hard constraints
def makeLaplacianMatrixSolverIGLHard(VPos, ITris, anchorsIdx):
    VPosE = igl.eigen.MatrixXd(VPos)
    ITrisE = igl.eigen.MatrixXi(ITris)
    L = igl.eigen.SparseMatrixd()
    M = igl.eigen.SparseMatrixd()
    M_inv = igl.eigen.SparseMatrixd()
    igl.cotmatrix(VPosE,ITrisE,L)
    igl.massmatrix(VPosE,ITrisE,igl.MASSMATRIX_TYPE_VORONOI,M)
    igl.invert_diag(M,M_inv)
    L = M_inv*L
    deltaCoords = L*VPosE
    deltaCoords = np.array(deltaCoords)
    
    #Bi-laplacian
    Q = L.transpose()*L
    #Linear term with delta coordinates
    
    
    #TODO: Finish this
    #return (L, solver, deltaCoords)

def makeLaplacianMatrixSolverIGLSoft(VPos, ITris, anchorsIdx, anchorWeights, makeSolver = True):
    VPosE = igl.eigen.MatrixXd(VPos)
    ITrisE = igl.eigen.MatrixXi(ITris)
    '''
    #Doing this check slows things down by more than a factor of 2 (convert to numpy to make faster?)
    for f in range(ITrisE.rows()):
        v_list = ITrisE.row(f)
        v1 = VPosE.row(v_list[0])
        v2 = VPosE.row(v_list[1])
        v3 = VPosE.row(v_list[2])
        if (v1-v2).norm() < 1e-10 and (v1-v3).norm() < 1e-10 and (v2-v3).norm() < 1e-10:
            print 'zero area triangle!',f
    '''
    L = igl.eigen.SparseMatrixd()
    M = igl.eigen.SparseMatrixd()
    M_inv = igl.eigen.SparseMatrixd()
    igl.cotmatrix(VPosE,ITrisE,L)
    igl.massmatrix(VPosE,ITrisE,igl.MASSMATRIX_TYPE_VORONOI,M)
    #np.set_printoptions(threshold='nan')
    #print 'what is M?',M.diagonal()
    igl.invert_diag(M,M_inv)
    #L = M_inv*L
    deltaCoords = (M_inv*L)*VPosE

    #TODO: What to do with decaying_anchor_weights?
    '''
    anchor_dists = []
    for i in range(VPosE.rows()):
        anchor_dists.append(min([ (VPosE.row(i)-VPosE.row(j)).norm() for j in anchorsIdx ]))
    max_anchor_dist = max(anchor_dists)
    # assume linear weighting for anchor weights -> we are 0 at the anchors, anchorWeights at max_anchor_dist
    decaying_anchor_weights = []
    for anchor_dist in anchor_dists:
        decaying_anchor_weights.append(anchorWeights*(anchor_dist/max_anchor_dist))
    '''
    
    solver = None
    if makeSolver:
        Q = L*(M_inv*M_inv)*L
        #Now add in sparse constraints
        diagTerms = igl.eigen.SparseMatrixd(VPos.shape[0], VPos.shape[0])
        # anchor points
        for a in anchorsIdx:
            diagTerms.insert(a, a, anchorWeights)
        # off points
        '''
        for adx,decay_weight in enumerate(decaying_anchor_weights):
            if decay_weight == 0:
                diagTerms.insert(adx, adx, anchorWeights)
            else:
                diagTerms.insert(adx, adx, decay_weight)
        '''
        Q = Q + diagTerms
        Q.makeCompressed()
        start_time = time.time()
        solver = igl.eigen.SimplicialLLTsparse(Q)
        #solver = igl.eigen.CholmodSupernodalLLT(Q)
        end_time = time.time()
        print 'factorization elapsed time:',end_time-start_time,'seconds'
    

    return (L, M_inv, solver, np.array(deltaCoords))

#solver: Eigen simplicialLLT solver that has Laplace Beltrami + anchors
#deltaCoords: numpy array of delta coordinates
#anchors: numpy array of anchor positions
#anchorWeights: weight of anchors
def solveLaplacianMatrixIGLSoft(solver, L, M_inv, deltaCoords, anchorsIdx, anchors, anchorWeights):
    print "solveLaplacianMatrixIGLSoft: anchorWeights = %g"%anchorWeights
    y = np.array(L*M_inv*igl.eigen.MatrixXd(np.array(deltaCoords, dtype=np.float64)))
    y[anchorsIdx] += anchorWeights*anchors
    y = igl.eigen.MatrixXd(y)
    ret = solver.solve(y)
    return np.array(ret)

if __name__ == '__main__2':
    anchorWeights = 10000
    m = PolyMesh()
    m.loadOffFile("cow.off")
    m.performDisplayUpdate()
    X = sio.loadmat("anchors.mat")
    anchors = X['anchors']
    anchorsIdx = X['anchorsIdx'].flatten().tolist()
    deltaCoords = X['deltaCoords']
    L = makeLaplacianMatrixCotWeights(m.VPos, m.ITris, anchorsIdx, anchorWeights)
    m.VPos = solveLaplacianMatrix(L, deltaCoords, anchors, anchorWeights)
    m.saveOffFile("LapCow.off")

if __name__ == '__main__3':
    anchorWeights = 100
    m = getSphereMesh(1, 2)
    print "BBox Before: ", m.getBBox()
    m.performDisplayUpdate()
    anchorsIdx = np.random.randint(0, len(m.vertices), 30).tolist()
    L = makeLaplacianMatrixCotWeights(m.VPos, m.ITris, anchorsIdx, anchorWeights)
    sio.savemat("L.mat", {"L":L})
    deltaCoords = L.dot(m.VPos)[0:len(m.vertices), :]
    anchors = m.VPos[anchorsIdx, :]
    anchors = anchors*5
    m.VPos = solveLaplacianMatrix(L, deltaCoords, anchors, anchorWeights)
    print "BBox After:", m.getBBox()
    m.saveOffFile("LapSphere.off")
