#Programmer: Chris Tralie
#Purpose: To provide a sample script for performing manipulates of
#off file geometry
import sys
sys.path.append("S3DGLPy")
from PolyMesh import *
import numpy as np

if __name__ == '__main__':
    meshin = "meshes/candide.off"
    meshout = "meshes/candiderot.off"
    m = PolyMesh()
    (VPos, VColors, ITris) = loadOffFileExternal(meshin)
    #Note: Points come into the program as a matrix in rows, so we need
    #to transpose to get them into our column format
    VPos = VPos.T
    #Make a random rotation matrix
    [R, _, _] = np.linalg.svd(np.random.randn(3, 3))
    #Apply rotation
    VPos = R.dot(VPos)
    #Save the results
    VPos = VPos.T #Need to transpose back before saving
    saveOffFileExternal(meshout, VPos, VColors*255, ITris)
