#-----------------------------------
# Main execution of code
#-----------------------------------

import sys, getopt
import time
import numpy as np
import matplotlib.pyplot as plt

from algorithms.functions import ALG_KEYS

def main(argv):
    # parse arguments
    try:
        opts, args = getopt.getopt(argv,"hn:d:a:",["numer_of_points=","dimension=", "algorithm="])
    except getopt.GetoptError:
        print('python 3 main.py -n <number_of_points> -d <dimension> -a <algorithm>')
        sys.exit(2)

    algor = ""
    n = 0
    dim = 0
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

    # Argument Check
    if algor not in ALG_KEYS:
        print('{} is not a supported algorithm. Try {}'.format(algor, ALG_KEYS.keys()))
        sys.exit(2)
    if algor == 'gw':
        dim = 2
    elif algor == 'inc' and  (dim < 2 or dim > 3):
        print('Algorithm {} is expecting (2 or 3)-dimensions, not {}'.format(algor, dim))
        sys.exit(2)
    if n == 0:
        print("Number of point must be an Integer Greater than zero.")
        sys.exit(2)


    # Create Random Points
    #S = np.random.uniform(-100, 100, size=(n,dim))
    S  = [[1,1], [2,1], [4,3], [1,5], [10,2], [0,8]]
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
        # print the returned convex hull
        for v in ch:
            print('({},{})'.format(v[0], v[1]), end=' ')
        print(end='\n')
    elif algor == 'inc' and dim == 3:
        for i, face in enumerate(ch):
            # add first point at the end to connect start with end of face
            print('face = {} : p1 = {}, p2 = {}, p3 ={}'.format(i, face[0], face[1], face[2]))

    alg.plotCH()

if __name__ == "__main__":
   main(sys.argv[1:])
