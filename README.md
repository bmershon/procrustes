# Procrustes Alignment

This assignment was completed as part of a course in 3D Digital Geometry (Math 290) taken at Duke University during Spring 2016. The course was taught by Chris Tralie.

# Files

*ICPView.py* and ICPViewGLUT.py are two GUI implementations designed for students to iteract with two meshes. The second implemention can be used when wxPython doesn't play nice on Mac operating systems.

*ICP.py* contains the code written by the student.

### Learning Process

The first major conceptual hurdle required figuring out how to use NumPy matrix operations, specifically *broadcasting*. In order to find correspondences between points in two point clouds, we can to create a matrix storing the (squared) Euclidean distance from a point in one cloud to each point in the second cloud. We want to do this without `for` loops, because we want this computation to happen very quickly.

This part of the implementation took me about 5 hours of reading about Python (from scratch), reading about NumPy, playing with various examples provided by Chris Tralie, and debugging in iPython to complete the *NearestNeighborBrute.py* example.

*from ICP.py*
```python
ab = np.dot(X.transpose(), Y) # each entry is Xi dot Yj
xx = np.sum(X*X, 0) # sum along squared coordinates, since points are column vectors
yy = np.sum(Y*Y, 0)
D = (xx[:, np.newaxis] + yy[np.newaxis, :]) - 2*ab
idx = np.argmin(D, 1)
```

### Observations

### Notes