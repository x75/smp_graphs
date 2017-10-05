"""smp_graphs graph.py

2017 Oswald Berthold

basic graph operations for networkx based smp graphs: construction from
smp_graphs config, manipulation, plotting, searching
"""

import networkx as nx
import re, copy, random, six

from collections import OrderedDict

from numpy import array

from matplotlib import colors

from smp_graphs.utils import print_dict
from smp_graphs.common import loop_delim, dict_replace_idstr_recursive

colors_ = list(six.iteritems(colors.cnames))

# colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
# colors = dict(mcolors.BASE_COLORS)

def nxgraph_add_nodes_from_dict(conf, G, nc):
    """nxgraph_add_nodes_from_dict

    take an smp_graphs dict with id, {block config} and create
    a graph node for every item in the dict
    Args:
     - conf: configuration dict
     - G: the graph
     - nc: node count

    Returns:
     - tuple (G, nc), the new graph, the new node count after insertion
    """
    # assert conf.has_key('params'), "config needs params dict"
    
    for k, v in conf.items():
        (G, nc) = nxgraph_add_node_from_conf(k, v, G, nc)
    return (G, nc)

def nxgraph_add_node_from_conf(k, v, G, nc):
    if not v.has_key('params'): v['params'] = {}
    v['params']['id'] = k
    # print "graphs.py: adding node = %s" % (v,)
    G.add_node(nc, v)
    nc += 1
    return (G, nc)

def nxgraph_from_smp_graph(conf):
    """nxgraph_from_smp_graph

    Construct a networkx graph 'nxgraph' from an smp_graph configuration dictionary 'conf'

    Args:
    - conf: smp_graphs configuration dict

    Returns:
    - G: the nx.graph object
    """
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
            G.add_node(nc, lpconf)
            # print "|G|", G.name, G.number_of_nodes()
            nc += 1

    # FIXME: function based loops, only used with SeqLoop yet
    # print "G.name", G.name, G.number_of_nodes()
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
    return layout
            
def nxgraph_flatten(G):
    """!@brief flatten a nested nx.graph to a single level"""
    G_ = nx.MultiDiGraph()
    for node in G.nodes_iter():
        if hasattr(G.node[node]['block_'], 'nxgraph'):
            # print "nxgraph_flatten: descending + 1"
            G_ = nx.compose(G_, nxgraph_flatten(G.node[node]['block_'].nxgraph))

    # final level
    ids_gen = (G.node[n]['params']['id'] for n in G)
    ids = list(ids_gen)
    mapping = dict(zip(G.nodes(), ids))
    rG = nx.relabel_nodes(G, mapping)
    qG = nx.compose(rG, G_)
    # print "nxgraph_flatten: relabeled graph", rG.nodes(), "composed graph", qG.nodes()
    # FIXME: consider edges
    return qG

def nxgraph_to_smp_graph(G, level = 0):
    """Walk the hierarchical graph depth-first and dump a dict"""
    gstr = str()
    for n in G.nodes():
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
    """!@brief get a node key from an nx.graph by searching for an smp_graphs id"""
    if nid is None: return
    gen = (n for n in G if G.node[n]['params']['id'] == nid)
    tmp = list(gen)
    # print "nid", nid, tmp
    return tmp
        
def nxgraph_node_by_id_recursive(G, nid):
    """!@brief get a node key from a nested nxgraph by searching for an smp_graphs id

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
        for n in G.nodes():
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
    # add nxgraph edges
    edges = []
    for node in G.nodes_iter():
    # for node in G.nodes():
        # print "node", node
        cnode = G.node[node]['block_']
        gen = (n for n in G if G.node[n]['block_'].id.startswith(node))
        for loopchild in list(gen):
            # print "graph.nxgraph_add_edges: loopchild = %s" %( loopchild,)
            # if v['params'].has_key('loopblock') and len(v['params']['loopblock']) == 0:
            if loopchild != node: # cnode.id:
                # k_from = node.split("_")[0]
                # print "k_from", node, loopchild
                edges.append((node, loopchild))
                # G.add_edge(k_from, node)
                
        for k, v in cnode.inputs.items():
            if not v.has_key('bus'): continue
            k_from_str, v_from_str = v['bus'].split('/')
            # print "edge from %s to %s" % (k_from_str, cnode.id)
            # print nx.get_node_attributes(self.top.nxgraph, 'params')
            k_from = (n for n in G if G.node[n]['params']['id'] == k_from_str)
            k_to   = (n for n in G if G.node[n]['params']['id'] == cnode.id)
            k_from_l = list(k_from)
            k_to_l = list(k_to)
            if len(k_from_l) > 0 and len(k_to_l) > 0:
                # print "fish from", k_from_l[0], "to", k_to_l[0]
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
    # print "layout", layout
    labels = {node[0]: '%s\n%s' % (node[1]['block_'].id, node[1]['block_'].cname[:-6]) for node in G.nodes(data = True)}
    # print "labels = %s" % labels
    nx.draw_networkx_nodes(G, ax = ax, pos = layout, node_color = node_color, node_shape = '8', node_size = node_size)
    # shift(layout, (0, -2 * node_size))
    nx.draw_networkx_labels(G, ax = ax, pos = layout, labels = labels, font_color = 'r', font_size = 8, fontsize = 6)
    # edges
    e1 = [] # std edges
    e2 = [] # loop edges
    for edge in G.edges():
        # edgetype = re.search("[_/]", G.node[edge[1]]['params']['id'])
        nodetype_0 = re.search('[%s]' % (loop_delim, ), G.node[edge[0]]['params']['id'])
        nodetype_1 = re.search('[%s]' % (loop_delim, ), G.node[edge[1]]['params']['id'])
        if nodetype_1 and not nodetype_0: # edgetype: # loop
            e2.append(edge)
            edgetype = "loop"
        else:
            e1.append(edge)
            edgetype = "data"
            
        # print "edge type = %s, %s" % (edgetype, edge)

    nx.draw_networkx_edges(G, ax = ax, pos = layout, edgelist = e1, edge_color = "g", width = 2)
    nx.draw_networkx_edges(G, ax = ax, pos = layout, edgelist = e2, edge_color = "k")

    # set title
    ax.set_title(G.name + " nxgraph")

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
