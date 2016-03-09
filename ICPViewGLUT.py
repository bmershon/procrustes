from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.arrays import vbo

import sys
sys.path.append("S3DGLPy")
from Primitives3D import *
from PolyMesh import *
from Cameras3D import *
from struct import *
from sys import exit, argv
import numpy as np
import scipy.io as sio
import os
import math
import time
from time import sleep
import numpy as np
import matplotlib.pyplot as plt
from ICP import *

def saveImageGL(mvcanvas, filename, w, h):
    view = glGetIntegerv(GL_VIEWPORT)
    pixels = glReadPixels(0, 0, view[2], view[3], GL_RGB,
                     GL_UNSIGNED_BYTE)
    I = np.fromstring(pixels, dtype=np.dtype('<b'))
    I = np.reshape(I, (h, w, 3))
    for k in range(3):
        I[:, :, k] = np.flipud(I[:, :, k])
    plt.imshow(I/255.0)
    plt.axis('off')
    plt.savefig(filename, dpi = 150, bbox_inches='tight')
    plt.clf()

class ICPViewerCanvas(object):
    def __init__(self, xmesh, ymesh, MaxIters = 200, outputPrefix = ""):
        #GLUT Variables
        self.GLUTwindow_height = 800
        self.GLUTwindow_width = 800
        self.GLUTmouse = [0, 0]
        self.GLUTButton = [0, 0, 0, 0, 0]
        self.GLUTModifiers = 0
        self.keys = {}
        self.bbox = BBox3D()
        
        self.xmesh = xmesh
        self.ymesh = ymesh
        self.displayMeshEdges = False
        self.displayMeshFaces = True
        self.displayMeshPoints = True
        self.displayCorrespondences = True
        self.currCx = np.array([[0, 0, 0]]).T #Current X Centroid
        self.currCy = np.array([[0, 0, 0]]).T #Current Y Centroid
        self.currRx = np.eye(3) #Current rotation
        self.CxList = []
        self.CyList = []
        self.RxList = []
        self.corridx = np.zeros([]) #Current correspondences
        self.corridxbuff = None #Correspondence vertex buffer
        self.MaxIters = MaxIters
        self.outputPrefix = outputPrefix
        #Animation variables
        self.animating = False
        self.frameIdx = 0
        self.nearDist = 0.01
        self.farDist = 1000.0
    
    def GLUTResize(self, w, h):
        glViewport(0, 0, w, h)
        self.GLUTwindow_height = 800
        self.GLUTwindow_width = 800
        #Update camera parameters based on new size
        self.camera = MousePolarCamera(w, h)
        self.camera.centerOnBBox(self.bbox, math.pi/2, math.pi/2)
    
    def handleMouseStuff(self, x, y):
        y = self.GLUTwindow_height - y
        self.GLUTmouse[0] = x
        self.GLUTmouse[1] = y

    def GLUTMouse(self, button, state, x, y):
        buttonMap = {GLUT_LEFT_BUTTON:0, GLUT_MIDDLE_BUTTON:1, GLUT_RIGHT_BUTTON:2, 3:3, 4:4}
        if state == GLUT_DOWN:
            self.GLUTButton[buttonMap[button]] = 1
        else:
            self.GLUTButton[buttonMap[button]] = 0
        self.handleMouseStuff(x, y)
        glutPostRedisplay()

    def GLUTMotion(self, x, y):
        lastX = self.GLUTmouse[0]
        lastY = self.GLUTmouse[1]
        self.handleMouseStuff(x, y)
        dX = self.GLUTmouse[0] - lastX
        dY = self.GLUTmouse[1] - lastY
        if self.GLUTButton[1] == 1:
            self.camera.translate(dX, dY)
        else:
            zooming = False
            if 'z' in self.keys:
                #Want to zoom in as the mouse goes up
                if self.keys['z']:
                    self.camera.zoom(-dY)
                    zooming = True
            elif 'Z' in self.keys:
                #Want to zoom in as the mouse goes up
                if self.keys['Z']:
                    self.camera.zoom(-dY)
                    zooming = True
            if not zooming:
                self.camera.orbitLeftRight(dX)
                self.camera.orbitUpDown(dY)
        glutPostRedisplay()

    def GLUTKeyboard(self, key, x, y):
        self.handleMouseStuff(x, y)
        self.keys[key] = True
        glutPostRedisplay()
    
    def GLUTKeyboardUp(self, key, x, y):
        self.handleMouseStuff(x, y)
        self.keys[key] = False
        if key in ['x', 'X']:
            self.viewXMesh()
        elif key in ['y', 'Y']:
            self.viewYMesh()
        elif key in ['p', 'P']:
            self.displayMeshPoints = not self.displayMeshPoints
        elif key in ['e', 'E']:
            self.displayMeshEdges = not self.displayMeshEdges
        elif key in ['f', 'F']:
            self.displayMeshFaces = not self.displayMeshFaces
        elif key in ['c', 'C']:
            self.displayCorrespondences = not self.displayCorrespondences
        glutPostRedisplay()

    def displayCorrespondencesCheckbox(self, evt):
        self.displayCorrespondences = evt.Checked()
        glutPostRedisplay()
    
    def getBBoxs(self):
        #Make Y bounding box
        Vy = self.ymesh.getVerticesCols() - self.currCy
        ybbox = BBox3D()
        ybbox.fromPoints(Vy.T)
        
        #Make X bounding box
        Vx = self.xmesh.getVerticesCols() - self.currCx
        Vx = self.currRx.dot(Vx)
        xbbox = BBox3D()
        xbbox.fromPoints(Vx.T)
        
        bboxall = BBox3D()
        bboxall.fromPoints(np.concatenate((Vx, Vy), 1).T)
        self.farDist = bboxall.getDiagLength()*20
        self.nearDist = self.farDist/10000.0
        return (xbbox, ybbox)
    
    #Move the camera to look at the Y mesh (default)
    def viewYMesh(self):
        (xbbox, ybbox) = self.getBBoxs()
        self.camera.centerOnBBox(ybbox, theta = -math.pi/2, phi = math.pi/2)
        glutPostRedisplay()

    #Move the camera to look at the X mesh, taking into consideration
    #current transformation
    def viewXMesh(self):
        (xbbox, ybbox) = self.getBBoxs()
        self.camera.centerOnBBox(xbbox, theta = -math.pi/2, phi = math.pi/2)
        glutPostRedisplay()
    
    def GLUTSpecial(self, key, x, y):
        self.handleMouseStuff(x, y)
        self.keys[key] = True
        glutPostRedisplay()
    
    def GLUTSpecialUp(self, key, x, y):
        self.handleMouseStuff(x, y)
        self.keys[key] = False
        glutPostRedisplay()
    
    def updateCorrBuffer(self):
        X = self.xmesh.VPos.T - self.currCx
        X = self.currRx.dot(X)
        Y = self.ymesh.VPos.T - self.currCy
        idx = self.corridx
        N = idx.size
        C = np.zeros((N*2, 3))
        C[0::2, :] = X.T
        C[1::2, :] = Y.T[idx, :]
        self.corridxbuff = vbo.VBO(np.array(C, dtype=np.float32))
    
    #Call the students' centroid centering code and update the display
    def centerOnCentroids(self):
        self.currCx = getCentroid(self.xmesh.getVerticesCols())
        self.currCy = getCentroid(self.ymesh.getVerticesCols())
        if self.corridxbuff: #If correspondences have already been found
            self.updateCorrBuffer()
        self.viewYMesh()

    def findCorrespondences(self):
        X = self.xmesh.getVerticesCols()
        Y = self.ymesh.getVerticesCols()
        self.corridx = getCorrespondences(X, Y, self.currCx, self.currCy, self.currRx)
        self.updateCorrBuffer()
        glutPostRedisplay()
    
    def doProcrustes(self):
        if not self.corridxbuff:
            wx.MessageBox('Must compute correspondences before doing procrustes!', 'Error', wx.OK | wx.ICON_ERROR)
            return
        X = self.xmesh.getVerticesCols()
        Y = self.ymesh.getVerticesCols()
        (self.currCx, self.currCy, self.currRx) = getProcrustesAlignment(X, Y, self.corridx)
        self.updateCorrBuffer()
        glutPostRedisplay()

    def doICP(self):
        X = self.xmesh.getVerticesCols()
        Y = self.ymesh.getVerticesCols()
        (self.CxList, self.CyList, self.RxList) = doICP(X, Y, self.MaxIters)
        self.currRx = self.RxList[-1]
        self.corridxbuff = None
        self.viewYMesh()

    def doAnimation(self):
        if len(self.RxList) == 0:
            wx.MessageBox('Must compute ICP before playing animation!', 'Error', wx.OK | wx.ICON_ERROR)
            return
        self.currRx = self.RxList[0]
        self.animating = True
        self.frameIdx = 0
        glutPostRedisplay()

    def drawPoints(self, mesh):
        glEnableClientState(GL_VERTEX_ARRAY)
        mesh.VPosVBO.bind()
        glVertexPointerf(mesh.VPosVBO)
        glDisable(GL_LIGHTING)
        glPointSize(POINT_SIZE)
        glDrawArrays(GL_POINTS, 0, mesh.VPos.shape[0])
        mesh.VPosVBO.unbind()
        glDisableClientState(GL_VERTEX_ARRAY)

    def drawLines(self, buff, NLines):
        glEnableClientState(GL_VERTEX_ARRAY)
        buff.bind()
        glVertexPointerf(buff)
        glDisable(GL_LIGHTING)
        glPointSize(POINT_SIZE)
        glDrawArrays(GL_LINES, 0, NLines*2)
        buff.unbind()
        glDisableClientState(GL_VERTEX_ARRAY)        

    def setupPerspectiveMatrix(self, nearDist = -1, farDist = -1):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        if nearDist == -1:
            farDist = self.camera.eye - self.bbox.getCenter()
            farDist = np.sqrt(farDist.dot(farDist)) + self.bbox.getDiagLength()
            nearDist = farDist/50.0
        gluPerspective(180.0*self.camera.yfov/M_PI, float(self.GLUTwindow_width)/self.GLUTwindow_height, nearDist, farDist)

    def repaint(self):
        if np.isnan(self.camera.eye[0]):
            #TODO: Patch for a strange bug that I can't quite track down
            #where camera eye is initially NaNs (likely a race condition)
            self.viewYMesh()
        self.setupPerspectiveMatrix(self.nearDist, self.farDist)
        
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        glEnable(GL_LIGHTING)
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0.2, 0.2, 0.2, 1.0])
        glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, 64)
        
        #glLightfv(GL_LIGHT1, GL_POSITION, np.array([0, 0, 1, 1]))
        self.camera.gotoCameraFrame()     
        P = np.zeros(4)
        P[0:3] = self.camera.eye   
        glLightfv(GL_LIGHT0, GL_POSITION, P)
        
        #Draw the Y mesh
        TYC = np.eye(4)
        TYC[0:3, 3] = -self.currCy.flatten()
        glPushMatrix()
        glMultMatrixd((TYC.T).flatten())
        self.ymesh.renderGL(self.displayMeshEdges, False, self.displayMeshFaces, False, False, True, False)
        if self.displayMeshPoints:
            glColor3f(1.0, 0, 0)
            self.drawPoints(self.ymesh)
        glPopMatrix()
        
        #Draw the X mesh transformed        
        Rx = np.eye(4)
        Rx[0:3, 0:3] = self.currRx
        #Translation to move X to its centroid
        TXC = np.eye(4)
        TXC[0:3, 3] = -self.currCx.flatten()
        T = Rx.dot(TXC)
        glPushMatrix()
        #Note: OpenGL is column major
        glMultMatrixd((T.T).flatten())
        self.xmesh.renderGL(self.displayMeshEdges, False, self.displayMeshFaces, False, False, True, False)
        if self.displayMeshPoints:
            glColor3f(0, 0, 1.0)
            self.drawPoints(self.xmesh)
        glPopMatrix()
        
        if self.displayCorrespondences and self.corridxbuff:
            self.drawLines(self.corridxbuff, self.xmesh.VPos.shape[0])
        
        if self.animating:
            if not(self.outputPrefix == ""):
                #Ouptut screenshots
                saveImageGL(self, "%s%i.png"%(self.outputPrefix, self.frameIdx), self.GLUTwindow_width, self.GLUTwindow_height)
            self.frameIdx += 1
            if self.frameIdx == len(self.RxList):
                self.animating = False
            else:
                self.currCx = self.CxList[self.frameIdx]
                self.currCy = self.CyList[self.frameIdx]
                self.currRx = self.RxList[self.frameIdx]
                glutPostRedisplay()
        glutSwapBuffers()

    def dmenu(self, item):
        self.menudict[item]()
        return 0
    
    def initGL(self):
        glutInit('')
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
        glutInitWindowSize(self.GLUTwindow_width, self.GLUTwindow_height)
        glutInitWindowPosition(50, 50)
        glutCreateWindow('ICP Viewer')
        glutReshapeFunc(self.GLUTResize)
        glutDisplayFunc(self.repaint)
        glutKeyboardFunc(self.GLUTKeyboard)
        glutKeyboardUpFunc(self.GLUTKeyboardUp)
        glutSpecialFunc(self.GLUTSpecial)
        glutSpecialUpFunc(self.GLUTSpecialUp)
        glutMouseFunc(self.GLUTMouse)
        glutMotionFunc(self.GLUTMotion)
        
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
        glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_TRUE)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT1, GL_DIFFUSE, [0.5, 0.5, 0.5, 1.0])
        glEnable(GL_LIGHT1)
        glEnable(GL_NORMALIZE)
        glEnable(GL_LIGHTING)
        
        glEnable(GL_DEPTH_TEST)
        
        #Make menus
        (VOID, CENTER_ON_CENTROIDS, FIND_CORRESPONDENCES, DO_PROCRUSTES, DO_ICP, ANIMATE_ICP) = (0, 1, 2, 3, 4, 5)
        self.menudict = {CENTER_ON_CENTROIDS:self.centerOnCentroids, FIND_CORRESPONDENCES:self.findCorrespondences, DO_PROCRUSTES:self.doProcrustes, DO_ICP:self.doICP, ANIMATE_ICP:self.doAnimation}
        
        stepByStepMenu = glutCreateMenu(self.dmenu)
        glutAddMenuEntry("Center On Centroids", CENTER_ON_CENTROIDS)
        glutAddMenuEntry("Find Correspondences", FIND_CORRESPONDENCES)
        glutAddMenuEntry("Do Procrustes", DO_PROCRUSTES)
        
        
        icpMenu = glutCreateMenu(self.dmenu)
        glutAddMenuEntry("Compute ICP", DO_ICP)
        glutAddMenuEntry("Animate ICP", ANIMATE_ICP)
        
        globalMenu = glutCreateMenu(self.dmenu)
        glutAddSubMenu("ICP Step By Step", stepByStepMenu)
        glutAddSubMenu("ICP Algorithm Full", icpMenu)
        glutAttachMenu(GLUT_RIGHT_BUTTON)
        
        glutMainLoop()

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print "Usage: python ICPViewerGLUT.py <mesh to align file> <target mesh file> [Maximum Number of Iterations] [Output File Prefix]"
        sys.exit(0)
    (xmeshfile, ymeshfile) = (sys.argv[1], sys.argv[2])
    MaxIters = 200
    if len(argv) > 3:
        MaxIters = int(argv[3])
    outputPrefix = ""
    if len(argv) > 4:
        outputPrefix = argv[4]
    xmesh = PolyMesh()
    print "Loading %s..."%xmeshfile
    (xmesh.VPos, xmesh.VColors, xmesh.ITris) = loadOffFileExternal(xmeshfile)
    xmesh.performDisplayUpdate(True)
    
    ymesh = PolyMesh()
    print "Loading %s..."%ymeshfile
    (ymesh.VPos, ymesh.VColors, ymesh.ITris) = loadOffFileExternal(ymeshfile)
    ymesh.performDisplayUpdate(True)
    
    viewer = ICPViewerCanvas(xmesh, ymesh, MaxIters, outputPrefix)
    viewer.initGL()
