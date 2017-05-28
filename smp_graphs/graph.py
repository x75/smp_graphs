"""smp_graphs basic graph operations for networkx based smp graphs
"""

import networkx as nx
import re, copy, random, six

from numpy import array

# from matplotlib import colors as mcolors
from matplotlib import colors


colors_ = list(six.iteritems(colors.cnames))

# colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
# colors = dict(mcolors.BASE_COLORS)

def nxgraph_from_smp_graph(conf):
    """!@brief construct an nx.graph from smp_graph configuration dictionary"""
    # new empty graph
    G = nx.MultiDiGraph()
    # node count
    nc = 0
    
    # node order: fixed with numeric keys, alternative: key array
    # slightly different types: graph, subgraph, loopblock
    if conf['params'].has_key('graph') or conf['params'].has_key('subgraph'):
        for k, v in conf['params']['graph'].items():
            v['params']['id'] = k
            # print "v", v
            G.add_node(nc, v)
            nc += 1
            
    elif conf['params'].has_key('loopblock') and type(conf['params']['loop']) is list:
        for i, item in enumerate(conf['params']['loop']):
            lpconf = copy.deepcopy(conf['params']['loopblock'])
            lpconf['params']['id'] = "%s_%d" % (conf['params']['id'], i)
            # FIXME: loop over param items
            lpconf['params'][item[0]] = item[1].copy()
            G.add_node(nc, lpconf)
            nc += 1
            
    # attributes?
    # edges / bus?
    # visualization
    return G

def nxgraph_get_layout(G, layout_type):
    """!@brief get an nx.graph layout from a config string"""
        if layout_type == "spring":
            # spring
            layout = nx.spring_layout(G)
        elif layout_type == "shell":
            # shell, needs add. computation
            s1 = []
            s2 = []
            for node in G.nodes_iter():
                # shells =
                # print node
                if re.search("/", node) or re.search("_", node):
                    s2.append(node)
                else:
                    s1.append(node)
                
            print "s1", s1, "s2", s2
            layout = nx.shell_layout(G, [s1, s2])
        elif layout_type == "pygraphviz":
            # pygraphviz
            import pygraphviz
            A = nx.nx_agraph.to_agraph(G)
            layout = nx.nx_agraph.graphviz_layout(G)
        elif layout_type == "random":
            layout = nx.random_layout(G)
        return layout

def nxgraph_flatten(G):
    """!@brief flatten a nested nx.graph to a single level"""
    G_ = nx.MultiDiGraph()
    for node in G.nodes_iter():
        if hasattr(G.node[node]['block_'], 'nxgraph'):
            print "nxgraph_flatten: descending + 1"
            G_ = nx.compose(G_, nxgraph_flatten(G.node[node]['block_'].nxgraph))

    # final level
    ids_gen = (G.node[n]['params']['id'] for n in G)
    ids = list(ids_gen)
    mapping = dict(zip(G.nodes(), ids))
    rG = nx.relabel_nodes(G, mapping)
    qG = nx.compose(rG, G_)
    print "rG", rG.nodes(), "qG", qG.nodes()
    return qG

def nxgraph_node_by_id(G, nid):
    """!@brief get a node key from an nx.graph by searching via smp graphs id"""
    if nid is None: return
    gen = (n for n in G if G.node[n]['params']['id'] == nid)
    tmp = list(gen)
    # print "nid", nid, tmp
    return tmp
        
def nxgraph_node_by_id_recursive(G, nid):
    """!@brief get a node key from a nested nx.graph by searching via smp graphs id"""
    if nid is None: return
    gen = (n for n in G if G.node[n]['params']['id'] == nid)
    tmp = list(gen)
    # print "nid", nid, len(tmp)
    if len(tmp) == 0:
        gen2 = (n for n in G if hasattr(G.node[n]['block_'], 'graph'))
        tmp2 = list(gen2)
        # print "tmp2", tmp2
        
    return tmp

def nxgraph_add_edges(G):
    # add nxgraph edges
    edges = []
    for node in G.nodes_iter():
    # for node in G.nodes():
        # print "node", node
        cnode = G.node[node]['block_']
        gen = (n for n in G if G.node[n]['block_'].id.startswith(node))
        for loopchild in list(gen):
            print "loopchild", loopchild
            # if v['params'].has_key('loopblock') and len(v['params']['loopblock']) == 0:
            if loopchild != node: # cnode.id:
                # k_from = node.split("_")[0]
                print "k_from", node, loopchild
                edges.append((node, loopchild))
                # G.add_edge(k_from, node)
                
        for k, v in cnode.inputs.items():
            if not v.has_key('bus'): continue
            k_from_str, v_from_str = v['bus'].split('/')
            print "edge from %s to %s" % (k_from_str, cnode.id)
            # print nx.get_node_attributes(self.top.nxgraph, 'params')
            k_from = (n for n in G if G.node[n]['params']['id'] == k_from_str)
            k_to   = (n for n in G if G.node[n]['params']['id'] == cnode.id)
            k_from_l = list(k_from)
            k_to_l = list(k_to)
            if len(k_from_l) > 0 and len(k_to_l) > 0:
                print "fish from", k_from_l[0], "to", k_to_l[0]
                edges.append((k_from_l[0], k_to_l[0]))
                # G.add_edge(k_from_l[0], k_to_l[0])
                
    G.add_edges_from(edges)
    return G
    # pass
                
def nxgraph_plot(G, ax, pos = None, layout_type = "spring", node_color = None, node_size = 1):
    ax.grid(0)
    if pos is None:
        layout = nxgraph_get_layout(G, layout_type)
    else:
        layout = pos

    if node_color is None:
        node_color = random.choice(colors_)

    # print "G.nodes", G.nodes(data=True)
    print "layout", layout
    labels = {node[0]: '%s\n%s' % (node[1]['block_'].cname, node[1]['block_'].id) for node in G.nodes(data = True)}
    print "labels = %s" % labels
    nx.draw_networkx_nodes(G, ax = ax, pos = layout, node_color = node_color, node_shape = '8', node_size = node_size)
    # shift(layout, (0, -2 * node_size))
    nx.draw_networkx_labels(G, ax = ax, pos = layout, labels = labels, font_color = 'r', font_size = 8, )
    # edges
    e1 = [] # std edges
    e2 = [] # loop edges
    for edge in G.edges():
        print "edge", edge
        if re.search("[_/]", G.node[edge[1]]['params']['id']):
            e2.append(edge)
        else:
            e1.append(edge)

    nx.draw_networkx_edges(G, ax = ax, pos = layout, edgelist = e1, edge_color = "g", width = 2)
    nx.draw_networkx_edges(G, ax = ax, pos = layout, edgelist = e2, edge_color = "k")
    

def scale(pos = {}, sf = 1):
    for k, v in pos.items():
        print "k, v", k, v,
        v = v * sf
        print v
        pos[k] = v

def shift(pos = {}, cl = 0):
    for k, v in pos.items():
        print "k, v", k, v,
        v = v + array(cl)
        print v
        pos[k] = v
            
# from https://stackoverflow.com/questions/31457213/drawing-nested-networkx-graphs
def recursive_draw(G, currentscalefactor = 0.1, center_loc = (0, 0), node_size = 300, shrink = 0.1, ax = None, layout_type = "spring"):
    if ax is not None:
        ax.grid(0)

    node_color = random.choice(colors_)
    pos = nxgraph_get_layout(G, layout_type)
    poslabels = copy.deepcopy(pos)
    # pos = nx.spring_layout(G)
    scale(pos, currentscalefactor) # rescale distances to be smaller
    shift(pos, center_loc) #you'll have to write your own code to shift all positions to be centered at center_loc
    # shift(poslabels, (0, 0.1)) #you'll have to write your own code to shift all positions to be centered at center_loc
    # nx.draw(G, pos = pos, node_size = node_size)
    nxgraph_plot(G, pos = pos, ax = ax, node_color = node_color, node_size = node_size)
    # for node, nodedata in G.nodes_iter(data=True):
    for node in G.nodes_iter():
        # if type(node)==Graph: # or diGraph etc...
        if hasattr(G.node[node]['block_'], 'nxgraph'):
            recursive_draw(G.node[node]['block_'].nxgraph, currentscalefactor = shrink * currentscalefactor, center_loc = pos[node], node_size = node_size*shrink, shrink = shrink, ax = ax)
# def nxgraph_add_edge(conf):
#             # if not v['params'].has_key('inputs'): continue
#             # for inputkey, inputval in v['params']['inputs'].items():
#             #     print "ink", inputkey
#             #     print "inv", inputval
#             #     if not inputval.has_key('bus'): continue
#             #     # get the buskey for that input
#             #     if inputval['bus'] not in ['None']:
#             #         k_from, v_to = inputval['bus'].split('/')
#             #         G.add_edge(k_from, k)
#     pass
