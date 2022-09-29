#-----------------------------------
# Main execution of code
#-----------------------------------

import sys, getopt
import time
import numpy as np
import matplotlib.pyplot as plt

from algorithms.functions import ALG_KEYS

def main(argv):
    # get arguments
    try:
        opts, args = getopt.getopt(argv,"hn:d:a:o",["numer_of_points=","dimension=", "algorithm="])
    except getopt.GetoptError:
        print('python 3 main.py -n <number_of_points> -d <dimension> -a <algorithm>')
        sys.exit(2)
    
    for opt, arg in opts:
        if opt == '-h':
            print('python 3 main.py -n <number_of_points>  -d <dimension> -a <algorithm>')
            sys.exit()
        elif opt in ("-n", "--number_of_points"):
            n = int(arg)
        elif opt in ("-d", "--dimension"):
            dim = int(arg)
        elif opt in ("-a", "--algorithm"):
            algor = str(arg).lower()

    if algor not in ALG_KEYS:
        print('{} is not a supported algorithm. Try {}'.format(algor, ALG_KEYS.keys()))
        sys.exit(1)
    if algor == 'gw2' and dim != 2:
        print('Algorithm {} is expecting 2-dimensions, not {}'.format(algor, dim))
        sys.exit(1)
    elif algor == 'cm' and dim < 2 and dim > 3:
        print('Algorithm {} is expecting (2 or 3)-dimensions, not {}'.format(algor, dim))
        sys.exit(1)


    # Create Random Points
    S = np.random.uniform(-100, 100, size=(n,dim))
    #print(S)
    
    # Find Convex Hull of S
    alg = ALG_KEYS[algor](S)
    print('Calculating convex hull on {} points with {}...'.format(n, ALG_KEYS[algor]))
    start_time = time.time()
    ch = alg.conv_hull()
    end_time = time.time()
    print("Foung convex hull in {} seconds".format(end_time-start_time))

    #plot points and convex hull
    if dim == 2:
        for v in ch:
            print('({},{})'.format(v[0], v[1]), end=' ')
        print(end='\n')
        
        x = S[:,0].flatten()
        y = S[:, 1].flatten()
        plt.plot(x, y, 'ro')      # plot the starting points

        if algor == 'gw2':
            ch = np.append(ch, [ch[0]], axis=0) # add first point at the end of the list
            # plot convex hull
            plt.plot(ch[:,0], ch[:,1], linestyle='-', color='y')
        else:
            # plot convex hull
            plt.plot(ch[0], ch[1], linestyle='-', color='y')

        plt.show()
    if dim == 3:
        ax = plt.axes(projection='3d')
        ax.scatter(S[:,0], S[:,1], S[:,2])
        for i, face in enumerate(ch):
            # add first point of face at the end to close circle
            face = np.append(face, [face[0]], axis = 0)
            print('face = {} : p1 = {}, p2 = {}, p3 ={}'.format(i, face[0], face[1], face[2], face[3]))
            
            ax.plot(face[:,0], face[:,1], face[:,2], color='y')
            plt.pause(0.01)

        plt.show()

if __name__ == "__main__":
   main(sys.argv[1:])
