# __Convex-Hull-Python__


## __Gift Wrapping (Jarvis March)__

Time Complexity: O(n<sup>2</sup>)

**Input**: Array with 2-dim points, a boolean for step-by-step vizualization

**Output**: Returns vector with the vertices of the Convex Hull in counter clock wise order

Example:
````python
    S  = [[1,1], [2,1], [4,3], [1,5], [10,2], [0,8]]
    gf = GiftWrapping(S)
    ch = gf.convex_hull()
````
To enable step-by-step vizualization with "plot" parameter:
````python
    gf = GiftWrapping(S, plot=True)
    ch = gf.conv_hull()
````

Use ```GiftWrapping.plotCh()``` to plot the convex hull.


## __Incremental via vertex coloring__

Time Complexity: Ο(nlogn + n<sup>2</sup>)

__Input__: Numpy Array with 2 | 3-dim points

__Output__: Returns array with the edges(2D) or the faces(3D) of the Convex Hull from rightmost to leftmost

Example:
````python
    S  = [[1,1], [2,1], [4,3], [1,5], [10,2], [0,8]]
    inc = Incremental(S)
    ch = inc.conv_hull()
````
Use ```Incremental.plotCh()``` to plot the convex hull.


### __General Information__

The classes of each algorith exist in "/algorithms/functions"

Use main function for an example use. Also gw2_vizualize.py runs GiftWrapping algorithm with vizualization

Example execution: 
````
python3 main.py -a <algorithm> -n <number_of_points> -d <dimension>
````