from tikz_network import plot
from tikz_network import TikzGraphDrawer

import networkx as nx
import igraph as ig
G = nx.fast_gnp_random_graph(11, 0.28)
G = nx.fast_gnp_random_graph(11, 0.28)
g = nx.planted_partition_graph(5, 5, 0.9, 0.1, seed=3)
g1 = ig.Graph(len(g), list(zip(*zip(*nx.to_edgelist(g))[:2])))
g1.get_adjacency()
plot(g1, 'bla')


