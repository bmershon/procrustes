from OpenGL.GL import *
from OpenGL.arrays import vbo
import wx
from wx import glcanvas

import sys
sys.path.append("S3DGLPy")
from Primitives3D import *
from PolyMesh import *
from Cameras3D import *
from MeshCanvas import *
from struct import *
from sys import exit, argv
import numpy as np
import scipy.io as sio
from pylab import cm
import os
import math
import time
from time import sleep
import matplotlib.pyplot as plt
from ICP import *

class ICPViewerCanvas(BasicMeshCanvas):
    def __init__(self, parent, xmesh, ymesh):
        super(ICPViewerCanvas, self).__init__(parent)
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
        self.maxItersTxt = None
        #Animation variables
        self.animating = False
        self.frameIdx = 0
        self.nearDist = 0.01
        self.farDist = 1000.0
        self.outputPrefixTxt = None
    
    def displayMeshFacesCheckbox(self, evt):
        self.displayMeshFaces = evt.Checked()
        self.Refresh()

    def displayMeshPointsCheckbox(self, evt):
        self.displayMeshPoints = evt.Checked()
        self.Refresh()

    def displayMeshEdgesCheckbox(self, evt):
        self.displayMeshEdges = evt.Checked()
        self.Refresh()
        
    def displayCorrespondencesCheckbox(self, evt):
        self.displayCorrespondences = evt.Checked()
        self.Refresh()
    
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
    def viewYMesh(self, evt):
        (xbbox, ybbox) = self.getBBoxs()
        self.camera.centerOnBBox(ybbox, theta = -math.pi/2, phi = math.pi/2)
        self.Refresh()

    #Move the camera to look at the X mesh, taking into consideration
    #current transformation
    def viewXMesh(self, evt):
        (xbbox, ybbox) = self.getBBoxs()
        self.camera.centerOnBBox(xbbox, theta = -math.pi/2, phi = math.pi/2)
        self.Refresh()
    
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
    def centerOnCentroids(self, evt):
        self.currCx = getCentroid(self.xmesh.getVerticesCols())
        self.currCy = getCentroid(self.ymesh.getVerticesCols())
        if self.corridxbuff: #If correspondences have already been found
            self.updateCorrBuffer()
        self.viewYMesh(None)

    def findCorrespondences(self, evt):
        X = self.xmesh.getVerticesCols()
        Y = self.ymesh.getVerticesCols()
        self.corridx = getCorrespondences(X, Y, self.currCx, self.currCy, self.currRx)
        self.updateCorrBuffer()
        self.Refresh()
    
    def doProcrustes(self, evt):
        if not self.corridxbuff:
            wx.MessageBox('Must compute correspondences before doing procrustes!', 'Error', wx.OK | wx.ICON_ERROR)
            return
        X = self.xmesh.getVerticesCols()
        Y = self.ymesh.getVerticesCols()
        (self.currCx, self.currCy, self.currRx) = getProcrustesAlignment(X, Y, self.corridx)
        self.updateCorrBuffer()
        self.Refresh()

    def doICP(self, evt):
        X = self.xmesh.getVerticesCols()
        Y = self.ymesh.getVerticesCols()
        MaxIters = 200
        if self.maxItersTxt:
            MaxIters = int(self.maxItersTxt.GetValue())
        (self.CxList, self.CyList, self.RxList) = doICP(X, Y, MaxIters)
        self.currRx = self.RxList[-1]
        self.corridxbuff = None
        self.viewYMesh(None)

    def doAnimation(self, evt):
        if len(self.RxList) == 0:
            wx.MessageBox('Must compute ICP before playing animation!', 'Error', wx.OK | wx.ICON_ERROR)
            return
        self.currRx = self.RxList[0]
        self.animating = True
        self.frameIdx = 0
        self.Refresh()

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

    def repaint(self):
        if np.isnan(self.camera.eye[0]):
            #TODO: Patch for a strange bug that I can't quite track down
            #where camera eye is initially NaNs (likely a race condition)
            self.viewYMesh(None)
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
            if self.outputPrefixTxt and not(self.outputPrefixTxt.GetValue() == ""):
                #Ouptut screenshots
                prefix = self.outputPrefixTxt.GetValue()
                saveImageGL(self, "%s%i.png"%(prefix, self.frameIdx))
            self.frameIdx += 1
            if self.frameIdx == len(self.RxList):
                self.animating = False
            else:
                self.currCx = self.CxList[self.frameIdx]
                self.currCy = self.CyList[self.frameIdx]
                self.currRx = self.RxList[self.frameIdx]
                self.Refresh()
        
        self.SwapBuffers()

class ICPViewerFrame(wx.Frame):
    (ID_SAVESCREENSHOT, ID_EXIT) = (1, 2)
    
    def __init__(self, parent, id, title, xmesh, ymesh, pos=DEFAULT_POS, size=DEFAULT_SIZE, style=wx.DEFAULT_FRAME_STYLE, name = 'GLWindow'):
        style = style | wx.NO_FULL_REPAINT_ON_RESIZE
        super(ICPViewerFrame, self).__init__(parent, id, title, pos, size, style, name)
        #Initialize the menu
        self.CreateStatusBar()
        
        self.size = size
        self.pos = pos
        self.glcanvas = ICPViewerCanvas(self, xmesh, ymesh)
        
        #####File menu
        filemenu = wx.Menu()
        menuSaveScreenshot = filemenu.Append(ICPViewerFrame.ID_SAVESCREENSHOT, "&Save Screenshot", "Save a screenshot of the GL Canvas")
        self.Bind(wx.EVT_MENU, self.OnSaveScreenshot, menuSaveScreenshot)
        menuExit = filemenu.Append(wx.ID_EXIT,"E&xit"," Terminate the program")
        self.Bind(wx.EVT_MENU, self.OnExit, menuExit)        
        #Creating the menubar
        menuBar = wx.MenuBar()
        menuBar.Append(filemenu,"&File") # Adding the "filemenu" to the MenuBar
        self.SetMenuBar(menuBar)  #Adding the MenuBar to the Frame content.
        
        self.rightPanel = wx.BoxSizer(wx.VERTICAL)
        
        #Buttons to go to a default view
        viewPanel = wx.BoxSizer(wx.HORIZONTAL)
        YmeshButton = wx.Button(self, -1, "Y Mesh")
        self.Bind(wx.EVT_BUTTON, self.glcanvas.viewYMesh, YmeshButton)
        viewPanel.Add(YmeshButton, 0, wx.EXPAND)
        XMeshButton = wx.Button(self, -1, "X Mesh")
        self.Bind(wx.EVT_BUTTON, self.glcanvas.viewXMesh, XMeshButton)
        viewPanel.Add(XMeshButton, 0, wx.EXPAND)
        self.rightPanel.Add(wx.StaticText(self, label="Views"), 0, wx.EXPAND)
        self.rightPanel.Add(viewPanel, 0, wx.EXPAND)
        
        #Checkboxes for displaying data
        self.rightPanel.Add(wx.StaticText(self, label=""))
        self.rightPanel.Add(wx.StaticText(self, label=""))
        self.rightPanel.Add(wx.StaticText(self, label="Display Options"), 0, wx.EXPAND)
        self.displayMeshFacesCheckbox = wx.CheckBox(self, label = "Display Mesh Faces")
        self.displayMeshFacesCheckbox.SetValue(True)
        self.Bind(wx.EVT_CHECKBOX, self.glcanvas.displayMeshFacesCheckbox, self.displayMeshFacesCheckbox)
        self.rightPanel.Add(self.displayMeshFacesCheckbox, 0, wx.EXPAND)
        
        self.displayMeshPointsCheckbox = wx.CheckBox(self, label = "Display Mesh Points")
        self.displayMeshPointsCheckbox.SetValue(True)
        self.Bind(wx.EVT_CHECKBOX, self.glcanvas.displayMeshPointsCheckbox, self.displayMeshPointsCheckbox)
        self.rightPanel.Add(self.displayMeshPointsCheckbox, 0, wx.EXPAND)

        self.displayMeshEdgesCheckbox = wx.CheckBox(self, label = "Display Mesh Edges")
        self.displayMeshEdgesCheckbox.SetValue(False)
        self.Bind(wx.EVT_CHECKBOX, self.glcanvas.displayMeshEdgesCheckbox, self.displayMeshEdgesCheckbox)
        self.rightPanel.Add(self.displayMeshEdgesCheckbox, 0, wx.EXPAND)

        self.displayCorrespondencesCheckbox = wx.CheckBox(self, label = "Display Correspondences")
        self.displayCorrespondencesCheckbox.SetValue(True)
        self.Bind(wx.EVT_CHECKBOX, self.glcanvas.displayCorrespondencesCheckbox, self.displayCorrespondencesCheckbox)
        self.rightPanel.Add(self.displayCorrespondencesCheckbox, 0, wx.EXPAND)

        #Buttons to test ICP algorithm step by step
        self.rightPanel.Add(wx.StaticText(self, label=""))
        self.rightPanel.Add(wx.StaticText(self, label=""))
        self.rightPanel.Add(wx.StaticText(self, label="ICP Algorithm Step By Step"))
        CentroidButton = wx.Button(self, -1, "Center Meshes on Centroids")
        self.Bind(wx.EVT_BUTTON, self.glcanvas.centerOnCentroids, CentroidButton)
        self.rightPanel.Add(CentroidButton, 0, wx.EXPAND)
        CorrespButton = wx.Button(self, -1, "Find Correspondences")
        self.Bind(wx.EVT_BUTTON, self.glcanvas.findCorrespondences, CorrespButton)
        self.rightPanel.Add(CorrespButton, 0, wx.EXPAND)
        ProcrustesButton = wx.Button(self, -1, "Do Procrustes Alignment")
        self.Bind(wx.EVT_BUTTON, self.glcanvas.doProcrustes, ProcrustesButton)
        self.rightPanel.Add(ProcrustesButton, 0, wx.EXPAND)        
        
        #Buttons to compute and test ICP in its entirety
        self.rightPanel.Add(wx.StaticText(self, label=""))
        self.rightPanel.Add(wx.StaticText(self, label=""))
        self.rightPanel.Add(wx.StaticText(self, label="ICP Algorithm Full"))
        ComputeICPButton = wx.Button(self, -1, "Compute ICP")
        self.Bind(wx.EVT_BUTTON, self.glcanvas.doICP, ComputeICPButton)
        self.rightPanel.Add(ComputeICPButton, 0, wx.EXPAND)
        AnimateICPButton = wx.Button(self, -1, "Animate ICP")
        self.Bind(wx.EVT_BUTTON, self.glcanvas.doAnimation, AnimateICPButton)
        self.rightPanel.Add(AnimateICPButton, 0, wx.EXPAND)
        self.rightPanel.Add(wx.StaticText(self, label="Maximum Iterations"))  
        self.glcanvas.maxItersTxt = wx.TextCtrl(self)
        self.glcanvas.maxItersTxt.SetValue("200")
        self.rightPanel.Add(self.glcanvas.maxItersTxt)
        self.rightPanel.Add(wx.StaticText(self, label="Output Prefix"))  
        self.glcanvas.outputPrefixTxt = wx.TextCtrl(self)
        self.rightPanel.Add(self.glcanvas.outputPrefixTxt)
        
        #Finally add the two main panels to the sizer        
        self.sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.sizer.Add(self.glcanvas, 2, wx.EXPAND)
        self.sizer.Add(self.rightPanel, 0, wx.EXPAND)
        
        self.SetSizer(self.sizer)
        self.Layout()
        self.glcanvas.Show()
        
    def OnSaveScreenshot(self, evt):
        dlg = wx.FileDialog(self, "Choose a file", ".", "", "*", wx.SAVE)
        if dlg.ShowModal() == wx.ID_OK:
            filename = dlg.GetFilename()
            dirname = dlg.GetDirectory()
            filepath = os.path.join(dirname, filename)
            saveImageGL(self.glcanvas, filepath)
        dlg.Destroy()
        return

    def OnExit(self, evt):
        self.Close(True)
        return

class ICPViewer(object):
    def __init__(self, xmeshfile, ymeshfile):
        #Do fast loading bypassing the mesh data structures since geometry
        #and topology are never changed
        xmesh = PolyMesh()
        print "Loading %s..."%xmeshfile
        (xmesh.VPos, xmesh.VColors, xmesh.ITris) = loadOffFileExternal(xmeshfile)
        xmesh.performDisplayUpdate(True)
        
        ymesh = PolyMesh()
        print "Loading %s..."%ymeshfile
        (ymesh.VPos, ymesh.VColors, ymesh.ITris) = loadOffFileExternal(ymeshfile)
        ymesh.performDisplayUpdate(True)
        
        app = wx.App()
        frame = ICPViewerFrame(None, -1, 'ICPViewer', xmesh, ymesh)
        frame.Show(True)
        app.MainLoop()
        app.Destroy()

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print "Usage: python ICPViewer.py <mesh to align file> <target mesh file>"
        sys.exit(0)
    viewer = ICPViewer(sys.argv[1], sys.argv[2])
