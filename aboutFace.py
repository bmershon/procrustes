# Adapted from Chris Tralie's manipulateGeometry.py
import sys
sys.path.append("S3DGLPy")
from PolyMesh import *
import numpy as np

if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(1)
    meshin = sys.argv[1]
    meshout = sys.argv[2]
    m = PolyMesh()
    (VPos, VColors, ITris) = loadOffFileExternal(meshin)
    #Note: Points come into the program as a matrix in rows, so we need
    #to transpose to get them into our column format
    VPos = VPos.T
    #Make a random rotation matrix
    R = np.eye(3, 3)
    R[2,2] = -1
    print(R)
    #Apply rotation
    VPos = R.dot(VPos)
    #Save the results
    VPos = VPos.T #Need to transpose back before saving
    saveOffFileExternal(meshout, VPos, VColors*255, ITris)
