# This file contains the code for each predicate used in the algorithms

import os
import numpy as np

#-----------------------------------------------------------------------
# Clockwise predicate for 2-dimensional points
#
# Input: Three 2-dimensional points type of np.array of list
# ex: [1.,1.] - [2.4,0.76] - [3.1 , 5.031]
# Output: Boolean
#-----------------------------------------------------------------------
def CW2(r, u, t):
    # Use the point to initialize a Numpy Array 
    # of shape (3, 3) where the first column is full of ones.
    # This array will be used to calculate the determinant
    
    n = 3
    tmp = np.array([r, u, t])
    arr = np.ones((n,n))
    arr[:, 1:] = tmp
    d = np.linalg.det(arr)
    if d < 0:
        # Clockwise turn - Point on right side
        return True
    elif d > 0:
        # Counter Clockwise turn - Point on left side
        return False
    else:
        # Det is 0 - Points are colinear
        # if u is between r and t, return true

        return is_between(r,t,u)


#-----------------------------------------------------------------------
# Counter Clockwise predicate for 3-dimensional points
#
# Input: One 3x3 array with the points of the face, and one 
# point out of the face
# Output: Boolean
#-----------------------------------------------------------------------
def CCW3(face, p):
    # Use the point to initialize a Numpy Array 
    # of shape (4, 4) where the first column is full of ones.
    # This array will be used to calculate the determinant
    
    n = 4
    arr = np.ones((n,n))
    arr[0, 1:] = p
    arr[1:, 1:] = face

    if np.linalg.det(arr) < 0:
        # Clockwise turn - Point on right side
        return False
    else:
        # Counter Clockwise turn - Point on left side
        return True

#-----------------------------------------------------------------------
# Helper function to deside if point c is between points a and b
# ALL POINTS MUST BE COLINEAR
#
# Input: Three 2-dimensional points type of np.array of list
# Output: Boolean
#-----------------------------------------------------------------------

def is_between(a, b, c):
    
    x0, x1, x2 = a[0], b[0], c[0]
    y0, y1, y2 = a[1], b[1], c[1]

    return ( (x2 >= min(x0, x1) and x2 <= max(x0, x1)) and (y2 >= min(y0, y1) and y2 <= max(y0, y1)) )
