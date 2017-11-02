"""Graph functions

.. moduleauthor:: Oswald Berthold, 2017

Basic graph operations for networkx based smp graphs: construction from
smp_graphs configuration dictionary, manipulation, plotting, searching
"""

import networkx as nx
import re, copy, random, six

from collections import OrderedDict

from numpy import array

from matplotlib import colors

from smp_graphs.utils import print_dict
from smp_graphs.common import loop_delim, dict_replace_idstr_recursive

# colors
colors_ = list(six.iteritems(colors.cnames))

from smp_base.plot import plot_colors

# colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
# colors = dict(mcolors.BASE_COLORS)

def nxgraph_nodes_iter(G, filt = None, data = False):
    if filter is None:
        return G.nodes(data = data)
    else:
        if data:
            node_filt_key = lambda G, n, filt: G.node[n[0]].has_key(filt)
            return filter(lambda n: not node_filt_key(G, n, filt) or (node_filt_key(G, n, filt) and G.node[n[0]][filt]), G.nodes(data = data))
        else:
            node_filt_key = lambda G, n, filt: G.node[n].has_key(filt)
            return filter(lambda n: not node_filt_key(G, n, filt) or (node_filt_key(G, n, filt) and G.node[n][filt]), G.nodes())
        # nodes_filtered = [nid for nid, ndata in G.nodes(data = True) if filter ndata.keys()]

def nxgraph_add_nodes_from_dict(conf, G, nc):
    """nxgraph_add_nodes_from_dict

    Take an smp_graphs dict with (id, block configuration) tuples and create a graph node for every item in the dict

    Args:
     - conf: configuration dict
     - G: the graph
     - nc: node count

    Returns:
     - tuple (G, nc), the new graph, the new node count after insertion
    """
    # assert conf.has_key('params'), "config needs params dict"
    
    for k, v in conf.items():
        assert type(v) is dict, "Block config not a dict, check configuration"
        assert v.has_key('block')
        (G, nc) = nxgraph_add_node_from_conf(k, v, G, nc)
    return (G, nc)

def nxgraph_add_node_from_conf(k, v, G, nc):
    assert type(v) is dict, "Expected type(v) = dict, got type %s, %s\n    node conf not a tuple?" % (type(v), v)
    if not v.has_key('params'): v['params'] = {}
    v['params']['id'] = k
    print "graphs.py: adding node = %s" % (v['params']['id'],)
    G.add_node(nc, **v)
    nc += 1
    return (G, nc)

def nxgraph_from_smp_graph(conf):
    """nxgraph_from_smp_graph

    Construct a networkx graph 'nxgraph' from an smp_graph configuration dictionary 'conf'.


    Args:
    - conf: smp_graphs configuration dict

    Returns:
    - G: the nx.graph object

    FIXME: graph G has no edges
    """
    # print "graph.py nxgraph_from_smp_graph kwargs[conf] = %s" % ( conf.keys(), )
    # new empty graph
    G = nx.MultiDiGraph()
    G.name = conf['params']['id']
    # node count
    nc = 0
    
    # assert conf['params'].has_key('loopmode'), "%s-%s loopmode?" % (conf['params']['id'], conf['block'])
    
    # node order: fixed with numeric keys, alternative: key array
    # slightly different types: graph, subgraph, loopblock
    if conf['params'].has_key('graph') or conf['params'].has_key('subgraph'):
        # two subordinate cases
        # 1 standard graph as dict
        # 2 existing node which we want to clone

        # print "type graph", type(conf['params']['graph'])
        if type(conf['params']['graph']) is OrderedDict:
            # print "nc", nc
            (G, nc) = nxgraph_add_nodes_from_dict(conf['params']['graph'], G, nc)
            # print "nc", nc
        elif type(conf['params']['graph']) is str and \
            conf['params']['graph'].startswith("id:"):
            # FIXME: obsolete, from cloning
            # look at the structure we want to copy and copy the input / output
            # spec?
            # no, input / output spec needs to be given, do the inference in a
            # later version
            
            # 1 get the block by id
            # 2 store the block in the graph
            # 3 check in step if things need initialization
            pass
            
    # do loopblock specific preprocessing for list based loop
    # FIXME: only do this for Loop, not SeqLoop
    elif conf['params'].has_key('loopblock') \
      and conf['params'].has_key('loopmode') \
      and conf['params']['loopmode'] is not 'sequential' \
      and type(conf['params']['loop']) is list:
        # construction loop
        for i, item in enumerate(conf['params']['loop']):
            # at least a one element list of key:val tuples
            if type(item) is tuple:
                item = [item]
            # get template and copy
            lpconf = copy.deepcopy(conf['params']['loopblock'])

            # rewrite block ids with loop count
            lpconf = dict_replace_idstr_recursive(d = lpconf, cid = conf['params']['id'], xid = "%d" % (i, ))
            
            """Examples for loop specification

            [('inputs', {'x': {'val': 1}}), ('inputs', {'x': {'val': 2}})]
            or
            [
                [
                    ('inputs',  {'x': {'val': 1}}),
                    ('gain',    0.5),
                    ],
                [
                    ('inputs', {'x': {'val': 2}})
                    ('gain',    0.75),
                    ]
                ]
            """
            # copy loop items into full conf
            for (paramk, paramv) in item:
                lpconf['params'][paramk] = paramv # .copy()

            # print "nxgraph_from_smp_graph", lpconf['params']['id']
            # print "nxgraph_from_smp_graph", print_dict(lpconf['params'])
            # print "|G|", G.name, G.number_of_nodes()
            G.add_node(nc, **lpconf)
            # print "|G|", G.name, G.number_of_nodes()
            nc += 1

    # FIXME: function based loops, only used with SeqLoop yet
    # print "G.name", G.name, G.number_of_nodes()
    return G

def nxgraph_get_layout(G, layout_type):
    """get an nx.graph layout from a config string"""
    if layout_type == "spring":
        # spring
        layout = nx.spring_layout(G)
    elif layout_type == "shell":
        # shell, needs add. computation
        s1 = []
        s2 = []
        for node in G.nodes():
            # shells =
            # print node
            # if re.search("/", node) or re.search("_", node):
            if re.search('%s' % (loop_delim, ), node): #  or re.search("_", node):
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
    # FIXME: include custom nested layout
    elif layout_type == "linear_hierarchical":
        # print 'G.nodes', G.nodes()
        pos = {}
        for node in G.nodes():
            pos[node] = array([G.node[node]['layout']['x'], G.node[node]['layout']['y']])

        return pos
        
    elif layout_type == "linear_hierarchical2":
        # print 'G.nodes', G.nodes()
        pos = {}
        lvly = {-1: 0}
        lvln = {}
        y_ = -1
        snodes = G.nodes()
        snodes.sort()
        print "snodes", snodes
        for node in snodes:
            lvl = int(node[1:2])
            lvln[node] = lvl
            # print "lvl", lvl # , lvly.keys()

            # # start each level from 0
            # if lvly.has_key(lvl):
            #     lvly[lvl] += 1
            # else:
            #     if lvl == 0: lvly[lvl] = 0
            #     else:
            #         lvly[lvl] = lvly[lvl-1]-1
            # y_ = lvly[lvl]

            # # shared y for all levels
            # y_ += 1
            
            # start each level's y at parent's y
            if not lvly.has_key(lvl): # new level, search parent
                print "lvl %d first = %s" % (lvl, node)
                
                # parentnode = None
                # # # assume edge
                # # for e in G.edges():
                # #     # print "e", e, node, e[1] == node
                # #     if e[1] == node and parentnode is None:
                # #         parentnode = e[0]
                        
                # # parentnode = G.edge[node]
                # # if parentnode is not None and len(parentnode) > 0:
                # print "node %s's parentnode is %s" % (node, parentnode) #, G.node[parentnode] #['graph_level'], lvl
                # if parentnode is None:
                #     lvly[lvl] = 0
                # else:
                #     lvly[lvl] = lvln[parentnode]
                #     # shift all lvl-1 nodes after parentnode by numthislvl
                lvly[lvl] = lvly[lvl-1]
                y_ = lvly[lvl]
            else:
                # print "has outgoing edge?", node, G.edge[node].keys()
                lvly[lvl] += 1
                
                numedges = len(G.edge[node].keys())
                y_ = lvly[lvl]
                if numedges > 0:
                    lvly[lvl+1] = lvly[lvl] - 1
                    lvly[lvl] += numedges - 1
                
            pos[node] = array([lvl, y_])
        # layout = nx.random_layout(G)
        # print "layout1", layout
        # print "layout2", pos
        layout = pos
    return layout
            
def nxgraph_flatten(G):
    """nxgraph_flatten

    Flatten a nested nxgraph 'G' to a single level
    """
    # print "nxgraph_flatten: G = %s with edges = %s" % (G.name, G.edges())
    
    # new graph
    G_ = nx.MultiDiGraph()

    # loop nodes and descend nesting hierarchy
    # for node in G.nodes():
    for node in nxgraph_nodes_iter(G, 'enable'):
        if hasattr(G.node[node]['block_'], 'nxgraph'):
            # print "nxgraph_flatten: descending + 1"
            G_ = nx.compose(G_, nxgraph_flatten(G.node[node]['block_'].nxgraph))

    # bottom level
    # generator for graph node ids
    ids_gen = (G.node[n]['params']['id'] for n in G)
    # generate ids
    ids = list(ids_gen)
    # map nx ids to smp_graph ids
    mapping = dict(zip(G.nodes(), ids))
    # relabel nodes
    rG = nx.relabel_nodes(G, mapping)
    # merge this graph into flat graph
    qG = nx.compose(rG, G_)
    # print "nxgraph_flatten: relabeled graph", rG.nodes(), "composed graph", qG.nodes()
    # FIXME: currently dropping edges?
    return qG

def nxgraph_to_smp_graph(G, level = 0):
    """Walk the hierarchical graph depth-first and dump a dict"""
    gstr = str()
    # for n in G.nodes():
    for n in nxgraph_nodes_iter(G, 'enable'):
        gn = G.node[n]
        # print "nxgraph_to_smp_graph: ", type(gn)
        # gstr += str(gn)
        gstr += '(\''
        gstr += gn['block_'].id
        gstr += '\', ' # {'
        gstr += print_dict(gn)
        gstr += '),\n'
        if hasattr(gn, 'nxgraph'):
            gstr += nxgraph_to_smp_graph(gn.nxgraph, level = level + 1)
    return gstr

def nxgraph_node_by_id(G, nid):
    """get a node key from an nx.graph by searching for an smp_graphs id"""
    if nid is None: return
    gen = (n for n in G if G.node[n]['params']['id'] == nid)
    tmp = list(gen)
    # print "nid", nid, tmp
    return tmp
        
def nxgraph_node_by_id_recursive(G, nid):
    """get a node key from a nested nxgraph by searching for an smp_graphs id

    Args:
        G: nxgraph
        nid: node params['id'] to search for
"""
    if nid is None: return []
    gen = ((n, G) for n in G if G.node[n]['params']['id'] == nid)
    tmp = list(gen)
    if len(tmp) > 0:
        return tmp
    else:
        # for n in G.nodes():
        for n in nxgraph_nodes_iter(G, 'enable'):
            # print "nxgraph_node_by_id_recursive: n = %s, node['block_'] = %s" % (n, G.node[n]['block_'].id)
            if hasattr(G.node[n]['block_'], 'nxgraph'):
                # print "nxgraph_node_by_id_recursive: node[%s]['block_'].nxgraph = %s" % (n, G.node[n]['block_'].nxgraph.nodes())
                tmp = nxgraph_node_by_id_recursive(G.node[n]['block_'].nxgraph, nid)
                if len(tmp) > 0:
                    return tmp
    # # print "nid", nid, len(tmp)
    # if len(tmp) < 1:
    #     gen2 = (n for n in G if hasattr(G.node[n]['block_'], 'graph'))
    #     tmp2 = list(gen2)
    #     # print "tmp2", tmp2
    
    return []

def nxgraph_add_edges(G):
    """nxgraph_add_edges

    Add edges to an nxgraph indicating block interactions via the bus or via looping.
    """
    # print "nxgraph_add_edges: G = %s with nodes = %s" % (G.name, G.nodes())
    # init edges
    edges = []
    # loop nodes
    # for node in G.nodes():
    for node in nxgraph_nodes_iter(G, 'enable'):
        # print "node", node
        # get the current node's block instance
        cnode = G.node[node]['block_']

        # get child nodes of the current node by matching block ids indicating loop interaction
        childgen = None
        # if G.node[node].has_key('block_'):
        childgen = (n for n in G if G.node[n].has_key('block_') and G.node[n]['block_'].id.startswith(node))
        # loop child nodes
        for childnode in list(childgen):
            # print "graph.nxgraph_add_edges: node = %s, childnode = %s" %(node, childnode,)
            # if v['params'].has_key('loopblock') and len(v['params']['loopblock']) == 0:
            if childnode != node: # cnode.id:
                # k_from = node.split("_")[0]
                # print "nxgraph_add_edges: loop edge %s -> %s" % (node, childnode)
                # edges.append((node, childnode))
                # edges.append((node, childnode, {'type': 'loop'}))
                G.add_edge(node, childnode, type = 'loop')

        # get child nodes of the current nodes by matching input bus ids indicating signal-based interaction
        for k, v in cnode.inputs.items():
            # ignore constant inputs
            if not v.has_key('bus'): continue

            # get node id and output variable of current input source
            k_from_str, v_from_str = v['bus'].split('/')
            # print "nxgraph_add_edges: cnode = %s, input = %s, bus edge %s -> %s" % (cnode.id, k, k_from_str, cnode.id)
            # print nx.get_node_attributes(self.top.nxgraph, 'params')

            # check from nodes exist
            # k_from = (n for n in G if G.node[n]['params']['id'] == k_from_str)
            k_from = (n for n in G if G.node[n].has_key('block_') and G.node[n]['block_'].id == k_from_str)
            k_from_l = list(k_from)
            # check to nodes exist
            k_to   = (n for n in G if G.node[n]['params']['id'] == cnode.id)
            k_to_l = list(k_to)

            # print "   k_from_l = %s\n     k_to_l = %s\n" % (k_from_l, k_to_l)
            
            # append edge if both exist
            if len(k_from_l) > 0 and len(k_to_l) > 0:
                # print "fish from", k_from_l[0], "to", k_to_l[0]
                edges.append((k_from_l[0], k_to_l[0]))
                # edges.append((k_from_l[0], k_to_l[0], {'type': 'data'}))
                G.add_edge(k_from_l[0], k_to_l[0], type = 'data')

    # update the graph with edges
    # G.add_edges_from(edges)
    return G

################################################################################
# plotting funcs
def nxgraph_plot(G, ax = None, pos = None, layout_type = "spring", node_color = None, node_size = 1):
    """nxgraph_plot

    Graph plotting func for flat graphs
    """
    assert ax is not None, "Need to pass 'ax' argument not None"
    # set grid off
    ax.grid(0)

    # compute layout if not supplied as argument
    if pos is None:
        layout = nxgraph_get_layout(G, layout_type)
    # or use precomputed layout from argument
    else:
        layout = pos

    # set node color default
    if node_color is None:
        node_color = random.choice(colors_)

    # # debug
    # print "G.nodes", G.nodes(data=True)
    # print "layout", layout

    # label all nodes with id and blocktype
    # labels = {node[0]: '%s\n%s' % (node[1]['block_'].id, node[1]['block_'].cname[:-6]) for node in G.nodes(data = True)}
    labels = {node[0]: '%s\n%s' % (node[1]['block_'].id, node[1]['block_'].cname[:-6]) for node in nxgraph_nodes_iter(G, 'enable', data = True)}

    # print "labels = %s" % labels

    # draw the nodes of 'G' into axis 'ax' using positions 'layout' etc
    nx.draw_networkx_nodes(G, ax = ax, pos = layout, node_color = node_color, node_shape = '8', node_size = node_size, alpha = 0.5)

    # # global shift?
    # shift(layout, (0, -2 * node_size))

    # draw the node labels
    nx.draw_networkx_labels(G, ax = ax, pos = layout, labels = labels, font_color = 'k', font_size = 8, fontsize = 6, alpha = 0.75)
    
    # edges
    typededges = {'data': [], 'loop': [], 'hier': []}
    # e1 = [] # bus edges
    # e2 = [] # loop edges

    # loop over edges
    for edge in G.edges(data = True):
        # edgetype = re.search("[_/]", G.node[edge[1]]['params']['id'])
        
        # this works for '|' style loop-id delimiter
        # nodetype_0 = re.search(r'[%s]' % (loop_delim, ), G.node[edge[0]]['params']['id'])
        # nodetype_1 = re.search(r'[%s]' % (loop_delim, ), G.node[edge[1]]['params']['id'])

        if type(edge[2]) is dict and edge[2].has_key('type'):
            if edge[2]['type'] == 'hier':
                if edge[2].has_key('main') and edge[2]['main']:
                    typededges[edge[2]['type']].append(edge)
            else:
                typededges[edge[2]['type']].append(edge)
                
        else: # infer type
            # this works for '_ll' style delimiter
            nodetype_0 = re.search(r'%s' % (loop_delim, ), G.node[edge[0]]['params']['id'])
            nodetype_1 = re.search(r'%s' % (loop_delim, ), G.node[edge[1]]['params']['id'])
            # print "node types = %s, %s based on loop_delim %s" % (nodetype_0, nodetype_1, loop_delim)
            if nodetype_1 and not nodetype_0: # edgetype: # loop
                typededges['loop'].append(edge)
                edgetype = "loop"
            else:
                typededges['data'].append(edge)
                edgetype = "data"
            
            # print "edge type = %s, %s" % (edgetype, edge)

    nx.draw_networkx_edges(G, ax = ax, pos = layout, edgelist = typededges['loop'], edge_color = "g", width = 1.0, alpha = 0.2)
    nx.draw_networkx_edges(G, ax = ax, pos = layout, edgelist = typededges['data'], edge_color = "k", width = 1.0, alpha = 0.2)
    nx.draw_networkx_edges(G, ax = ax, pos = layout, edgelist = typededges['hier'], edge_color = "b", width = 0.8, alpha = 0.2)

    # set title to config filename removing timestamp and hash
    # title = re.sub(r'_[0-9]+_[0-9]+', r'', G.name.split("-")[0])
    title = ''
    ax.set_title(title + 'nxgraph G, |G| = %d' % (G.number_of_nodes(), ), fontsize = 8)

    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([]) 

def scale(pos = {}, sf = 1):
    for k, v in pos.items():
        # print "nxgraph.scale k, v", k, v,
        v = v * sf
        # print v
        pos[k] = v

def shift(pos = {}, cl = 0):
    for k, v in pos.items():
        # print "nxgraph.shift k, v", k, v,
        v = v + array(cl)
        # print v
        pos[k] = v

def recursive_draw(G, currentscalefactor = 0.1, center_loc = (0, 0), node_size = 300, shrink = 0.1, ax = None, layout_type = "spring"):
    """recursively draw nested graph

    from https://stackoverflow.com/questions/31457213/drawing-nested-networkx-graphs

    FIXME: draw edges across hierarchy boundaries
    """
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
    # for node, nodedata in G.nodes(data=True):
    for node in G.nodes():
        # if type(node)==Graph: # or diGraph etc...
        if hasattr(G.node[node]['block_'], 'nxgraph'):
            recursive_draw(
                G.node[node]['block_'].nxgraph,
                currentscalefactor = shrink * currentscalefactor,
                center_loc = pos[node],
                node_size = node_size*shrink,
                shrink = shrink,
                ax = ax)

def nxgraph_plot2(G):
    import matplotlib.pyplot as plt
    G_ = G
    # nxgraph_plot(G_)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    layout_type = 'linear_hierarchical' # shell, pygraphviz, random
    pos = nxgraph_get_layout(G_, layout_type)

    nxgraph_plot(G = G_, pos = pos, ax = ax, node_size = 300)
    # print "pos", pos
    # labels = {node[0]: '%s\n%s' % (node[1]['block_'].id, node[1]['block_'].cname[:-6]) for node in G_.nodes(data = True)}
    # nx.draw_networkx_nodes(G_, pos = pos, ax = ax)
    # nx.draw_networkx_labels(G_, ax = ax, pos = pos, labels = labels, font_color = 'k', font_size = 8, fontsize = 6)
    # plt.draw()
    # plt.pause(0)
            
def recursive_hierarchical(G, lvlx = 0, lvly = 0):
    """another recursive graph drawing func

    1) draw the hierarchy without scaling of subgraphs (like printgraph)
    2) construct flattened graph during draw recursion
    3) draw edges across hierarchy boundaries
    """

    G_ = nx.MultiDiGraph()

    xincr = 0.2/float(G.number_of_nodes() + 1)
    
    # for i, node in enumerate(G.nodes()):
    for i, node in enumerate(nxgraph_nodes_iter(G, 'enable')):
        G.node[node]['layout'] = {}
        G.node[node]['layout']['level'] = lvlx
        G.node[node]['layout']['x'] = lvlx + (i * xincr)
        G.node[node]['layout']['y'] = lvly
        # G_.add_node('l%d_%s_%s' % (lvl, node, G.node[node]['block_'].id), G.node[node])
        nodeid_ = 'l%d_%s' % (lvlx, G.node[node]['block_'].id)
        # print "node", nodeid_ # G.node[node]['block_'].id # .keys()
        G_.add_node(nodeid_, **G.node[node])
        # descend into subgraph
        if hasattr(G.node[node]['block_'], 'nxgraph'):
            # print "node.nxgraph:", G.node[node]['block_'].nxgraph
            # lvlx += 1
            G2, G2_number_of_nodes_total = recursive_hierarchical(G.node[node]['block_'].nxgraph, lvlx = lvlx + 1, lvly = lvly)
            lvly += G2_number_of_nodes_total # G2.number_of_nodes()
            # print "G2", G2.nodes()
            G_ = nx.compose(G2, G_)
            mainedge = True
            for g2node in G2:
                g2nodelyt = G2.node[g2node]['layout']
                g2x = g2nodelyt['x']
                g2y = g2nodelyt['y']
                # print "node %s - g2node %s[%d/%d]" % (nodeid_, g2node, g2x, g2y)
                if lvlx == g2x-1 and G_.node[nodeid_]['layout']['y'] == g2y:
                    G_.add_edge(nodeid_, g2node, type = 'hier', main = mainedge)
                    mainedge = False
            # print "G_", G_.nodes(), G_.edges()
        else:
            lvly += 1

    if lvlx == 0:
        G_.name = G.name
        G_ = nxgraph_add_edges(G_)
        # nxgraph_plot2(G_)
    return G_, G_.number_of_nodes()


def nxgraph_get_node_colors(G):
    """nxgraph_get_node_colors

    Get the block's color from 'block_color' attribute and return it
    """
    G_cols = []
    pcks = plot_colors.keys()
    pcks.sort()
    for i, n in enumerate(G.nodes()):
        if hasattr(G.node[n]['block_'], 'block_color'):
            block_color = G.node[n]['block_'].block_color
        else:
            # block_color = "k"
            # block_color = random.choice(colors_)[1]
            block_color = plot_colors[pcks[i]]
        # print "nxgraph_get_node_colors node", n, "block_color", block_color
        G_cols.append(block_color)
    return G_cols
