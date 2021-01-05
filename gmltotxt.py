from igraph import *

g = Graph.Read_GML("Layer42.gml")
g.write_pajek("layer42.txt")

