import numpy as np
from PolyMesh import *

if __name__ == '__main__':
    m = PolyMesh()
    m.loadFile("homer.off")
    m.performDisplayUpdate()

    m2 = PolyMesh()
    (m2.VPos, m2.VColors, m2.ITris) = loadOffFileExternal("homer.off")
    m2.performDisplayUpdate(True)
    
    print m.EdgeLines
    print "\n\n\n"
    print m2.EdgeLines
