import numpy as np
from Primitives3D import *

if __name__ == '__main__':  
    np.random.seed(2)
    X = np.random.randn(100, 3)
    Y = np.random.randn(100, 3)
    xbbox = BBox3D()
    xbbox.fromPoints(X)
    print xbbox
    ybbox = BBox3D()
    ybbox.fromPoints(Y)
    print xbbox
