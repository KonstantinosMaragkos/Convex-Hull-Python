# ---------------------------------- #
# Executing Gift Wrapping for 2D
# with step by step vizualization
# ---------------------------------- #

import sys
import numpy as np
import matplotlib.pyplot as plt

from algorithms.functions import GiftWrapping2

def main():

    n = 15  # Number of points

    algor = 'gw2'
    dim = 2

    S = np.random.uniform(-100, 100, size=(n,dim))

    plt.clf()
    plt.scatter(S[:,0], S[:,1], color='b')
    
    gw2 = GiftWrapping2(S, plot=True)
    ch = gw2.conv_hull()
    print(ch)

if __name__ == '__main__':
    main()