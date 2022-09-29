#------------------------------------------------------------------
# Gift Wrapping (Jarvis) Algorithm for 2-dimensional Convex Hulls
# with finite number of points
#
# Input: Numpy Array with 2-dim points
# Output: 
#------------------------------------------------------------------

from algorithms.predicates import CW2, CCW3

import numpy as np
import matplotlib.pyplot as plt

from itertools import combinations

class GiftWrapping2:
    def __init__(self, S, plot=False):
        if type(S).__module__ != np.__name__:
            self.S = np.asarray(S)
        else:
            self.S = S
        self.n = S.shape[0] # number of points
        
        # initialize visited list for vertices
        self.visited = np.zeros(self.n, dtype=int)
        self.plot = plot
        self.edges = []
        self.pause = 0.01

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
        plt.xlim(-110,110)
        plt.ylim(-110,110)
        plt.scatter(self.S[:,0], self.S[:,1], color='b')
        for e in self.edges:
            x = [e[0][0], e[1][0]]
            y = [e[0][1], e[1][1]]
            plt.plot(x,y, color=e[2])

    
    def conv_hull(self):
        # step 1
        ridx, r = self.find_min()
        r0 = r
        #step 2
        ch = [r0]
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
                ch.append(r)
                

        plt.show()
        return np.asarray(ch)

class ColorMapping:
    def __init__(self, S):
        if type(S).__module__ != np.__name__:
            self.S = np.asarray(S)
        else:
            self.S = S
        self.n , self.dim = S.shape
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

        ch = []
        if self.dim == 2:
            x = self.S[:,0]
            y = self.S[:,1]
            for v in self.face.values():
                    ch.append(v)
            ch = np.asarray(ch)
            return np.asarray([x[ch.T], y[ch.T]])
        if self.dim == 3:
            x = self.S[:,0]
            y = self.S[:,1]
            z = self.S[:,2]
            for v in self.face.values():
                tmp = []
                for p in v:
                    tmp.append([x[p], y[p], z[p]])
                ch.append(tmp)
            return np.asarray(ch)

# ALGORITHM SHORTCUT KEYS
ALG_KEYS = {'gw2': GiftWrapping2, 'cm': ColorMapping}