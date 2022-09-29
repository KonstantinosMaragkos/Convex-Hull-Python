#------------------------------------------------------------------
# Gift Wrapping (Jarvis) Algorithm for 2-dimensional Convex Hull
# with finite number of points
#
# Input: Numpy Array with 2-dim points, a boolean for step-by-step vizualization
# Output: Returns vector with the vertices of the Convex Hull
# in counter clock wise order
# O(n^2) complexity
#------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

from algorithms.predicates import CW2, CCW3

class GiftWrapping:
    def __init__(self, S, plot=False):
        if type(S).__module__ != np.__name__:
            self.S = np.asarray(S)
        else:
            self.S = S
        self.n = self.S.shape[0] # number of points

        self.ch = []
        
        # initialize visited list for vertices
        self.visited = np.zeros(self.n, dtype=int)
        self.plot = plot
        self.edges = []
        self.pause = 0.01
    
    def find_range(self):
        mult = 1.1
        self.minx = mult*np.min(self.S[:,0])
        self.miny = mult*np.min(self.S[:,1])
        self.maxx = mult*np.max(self.S[:,0])
        self.maxy = mult*np.max(self.S[:,1])

    def find_min(self):
        # returns index , and min point

        # all min points based on x-axis
        minx = np.where(self.S[:,0] == np.amin(self.S[:,0]))

        # for each min point, find the one with the smallest y-axis value
        miny = np.argmin(self.S[minx], axis=0)[1]
        idx = minx[0][miny]
        return idx, self.S[idx]
    
    def is_visited(self, idx):
        return self.visited[idx]

    def pick_unvisited(self):
        idx = 0
        while self.visited[idx] and idx < self.n - 1:
            idx += 1
        return idx, self.S[idx]

    def all_visited(self):
        return sum(self.visited) == self.n
    
    def plot_edges(self):
        plt.clf()
        plt.xlim(self.minx, self.maxx)
        plt.ylim(self.miny, self.maxy)
        plt.scatter(self.S[:,0], self.S[:,1], color='b')
        for e in self.edges:
            x = [e[0][0], e[1][0]]
            y = [e[0][1], e[1][1]]
            plt.plot(x,y, color=e[2])

    
    def conv_hull(self):
        if self.plot:
            self.find_range()
        # step 1
        ridx, r = self.find_min()
        r0 = r
        #step 2
        self.ch = [r0]
        self.visited[ridx] = 1
        finished = False


        while not finished:
            # step 3
            uidx, u = self.pick_unvisited()
            self.visited[uidx] = 1

            if self.plot:
                self.edges.append((r, u, 'red'))
                self.plot_edges()
                plt.pause(self.pause)

            tidx = 0
            for t in self.S:
                if self.is_visited(tidx) or all(t == u):
                    tidx += 1
                    continue

                if self.plot:
                    self.edges.append((r, t, 'red'))
                    self.plot_edges()  
                    plt.pause(self.pause)

                if CW2(r, u, t):
                    if self.plot:
                        # pop rt edge
                        self.edges.pop(-1)
                        # pop ru edge
                        self.edges.pop(-1)
                        self.edges.append((r, t, 'red'))
                        self.plot_edges()
                        plt.pause(self.pause)
                    u = t
                    self.visited[uidx] = 0
                    uidx = tidx
                else:
                    if self.plot:
                        self.edges.pop(-1)
                tidx += 1
            # step 4
            if CW2(r, u, r0):
                finished = True
                if self.plot:
                    self.edges.pop(-1)
                    self.edges.append((r, r0, 'green'))
                    self.plot_edges()
                    plt.pause(self.pause)
            else:
                if self.plot:
                    self.edges.append((r, u, 'green'))
                    self.plot_edges()
                    plt.pause(self.pause)
                r = u
                self.visited[uidx] = 1
                self.ch.append(r)
                

        plt.show()
        self.ch = np.asarray(self.ch)
        return self.ch

    def plotCH(self):
        x = self.S[:,0].flatten()
        y = self.S[:, 1].flatten()
        plt.plot(x, y, 'ro')      # plot the starting points
        self.ch = np.append(self.ch, [self.ch[0]], axis=0) # add first point at the end of the list
            # plot convex hull
        plt.plot(self.ch[:,0], self.ch[:,1], linestyle='-', color='y')
        plt.show()


#------------------------------------------------------------------------------
# Incremental Algorithm for 2 and 3-dimensional Convex Hull
# with finite number of points
#
# Input: Numpy Array with 2 | 3-dim points
# Output: Returns array with the edges(2D) or the faces(3D) of the Convex Hull
# Ο(nlogn + n^2) Complexity
#------------------------------------------------------------------------------
class Incremental:
    def __init__(self, S):
        if type(S).__module__ != np.__name__:
            self.S = np.asarray(S)
        else:
            self.S = S
        self.n, self.dim = self.S.shape # number of points
        self.n , self.dim = self.S.shape
        self.v_color = {}               # color of edges/vertices in polytope
        self.f_color = {}               # color of faces/edges in polytope
        self.face = {}                  # holds points of faces/edges
        self.key = 0

    def sort_points(self):
        if self.dim == 2:
            self.S.view('f8,f8').sort(order=['f0'], axis=0)
        if self.dim == 3:
            self.S.view('f8,f8,f8').sort(order=['f0'], axis=0)
        self.S = self.S[::-1] # descending order

    def pick_inner_vertex(self, face, k):
        # find a point in the polytope
        # which is not a vertex of the face
        for point_idx in range(k) :
            if point_idx not in face:
                return point_idx
        return -1

    def find_color(self, face, k):
        # compare two CCW predicates
        # and return true if color of face/edge is red
        # otherwise false
        if self.dim == 2:
            i = self.pick_inner_vertex(face, k)
            return True if CW2(self.S[face[0]], self.S[face[1]], self.S[k]) != CW2(self.S[face[0]], self.S[face[1]], self.S[i]) else False
        else:
            i = self.pick_inner_vertex(face, k)
            tmp = [self.S[face[0]], self.S[face[1]], self.S[face[2]]]
            while i != -1:
                if CCW3(tmp, self.S[k]) != CCW3(tmp, self.S[i]):
                    return True
                face = np.append(face, [i], axis=0)
                i = self.pick_inner_vertex(face, k)
            return False

    def setup(self):
        self.key = 0
        for i in range(self.dim, -1, -1):
            if self.dim == 2:
                self.v_color[i] = 0
        
            for j in range(i-1, -1, -1):
                if self.dim == 2:
                    if i != j:
                        self.face[self.key] = [i,j]
                        self.key += 1
                else:
                    for k in range(j-1, -1, -1):
                        if i != j and j != k:
                            self.face[self.key] = [i,j,k]
                            self.color_edges(self.face[self.key], init=True)
                            self.key += 1
    
    def reset_colors(self):
        # resets vertices colors
        for key in self.v_color:
            self.v_color[key] = 0
        for key in self.f_color:
            self.f_color[key] = 0
    
    def add_faces(self, k):
        # find purple edges/points
        points = []
        for p in self.v_color:
            if self.v_color[p] == 1:
                points.append(p)
        
        # add faces/edges
        l = len(points)
        
        if self.dim == 2:
            for i in range(l):
                self.face[self.key] = [k, points[i]]
                self.key += 1
                self.v_color[k] = 0
        else:
            for i in range(l):
                edge = self.edge_to_points(points[i])
                self.face[self.key] = [k, edge[0], edge[1]]
                self.color_edges(self.face[self.key], init=True)
                self.key += 1
        
# ------------------------------------------------------------------ #
# -------------- HELPER FUNCTIONS FOR CONVEX HULL 3-D -------------- #

    def edge_to_points(self, edge):
        # extract point as ints
        # from edge as string
        tmp = edge.split('_')
        edge = []
        for i in tmp:
            edge.append(int(i))
        return edge
    
    def edge_to_string(self, p1, p2):
        return '{}_{}'.format(p1, p2)

    def color_edges(self, face, init=False):
        for i in combinations(face, 2):
            edge = self.edge_to_string(i[0], i[1])
            if init:
                self.v_color[edge] = 0
            else:
                self.v_color[edge] += 1

# ------------------------------------------------------------------ #

    def conv_hull(self):
        # sort point by descenting order on x-axis
        self.sort_points()
        # initialize convex hull, and color mapping structures
        self.setup()

        for k in range(self.dim+1, self.n):
            # find color of faces of polytope in contrast with k
            for key in self.face:
                if self.find_color(self.face[key], k) == True:
                    #print('Found red face/edge', self.face[key])
                    self.f_color[key] = 1       # red face(3D)/edge(2D)
                    if self.dim == 3:
                        # color edges
                        self.color_edges(self.face[key])
                    else:
                        # color vertices
                        for v in self.face[key]:
                            self.v_color[v] += 1

            # delete faces/edges
            for key in self.f_color:
                if self.f_color[key] == 1:
                    del self.face[key]
            # add new faces/edges (k, purple points)
            self.add_faces(k)
            self.reset_colors()

        self.ch = []
        if self.dim == 2:
            x = self.S[:,0]
            y = self.S[:,1]
            for v in self.face.values():
                    self.ch.append(v)
            self.ch = np.asarray(self.ch)
            return self.S[self.ch]
        if self.dim == 3:
            x = self.S[:,0]
            y = self.S[:,1]
            z = self.S[:,2]
            for v in self.face.values():
                tmp = []
                for p in v:
                    tmp.append([x[p], y[p], z[p]])
                self.ch.append(tmp)
            
            self.ch = np.asarray(self.ch)
            return self.ch
    
    def plotCH(self):
        if self.dim == 2:
            x = self.S[:,0].flatten()
            y = self.S[:, 1].flatten()
            ch = np.asarray([x[self.ch.T], y[self.ch.T]])
            plt.plot(x, y, 'ro')      # plot the starting points
            plt.plot(ch[0], ch[1], linestyle='-', color='y')
        else:
            ax = plt.axes(projection='3d')
            ax.scatter(self.S[:,0], self.S[:,1], self.S[:,2])
            for face in self.ch:
                # add first point of face at the end to close circle
                face = np.append(face, [face[0]], axis = 0)
                ax.plot(face[:,0], face[:,1], face[:,2], color='y')
        plt.show()

# ALGORITHM SHORTCUT KEYS
ALG_KEYS = {'gw': GiftWrapping, 'inc': Incremental}