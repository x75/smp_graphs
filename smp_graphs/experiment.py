"""smp_graphs - smp sensorimotor experiments as computation graphs

2017 Oswald Berthold

experiment: basic experiment shell for
 - running a graph
 - loading and drawing a graph (networkx)
"""

import argparse, os
import time

from collections import OrderedDict
from functools import partial

import matplotlib.pyplot as plt

import numpy as np

# for config reading
from numpy import array

from smp_base.plot import set_interactive, makefig

from smp_graphs.block import Block2
from smp_graphs.utils import print_dict
from smp_graphs.common import conf_header, conf_footer
from smp_graphs.common import get_config_raw
from smp_graphs.graph import nxgraph_plot, recursive_draw, nxgraph_flatten, nxgraph_add_edges

################################################################################
# utils, TODO: move to utils.py
def get_args():
    # define defaults
    default_conf     = "conf/default.py"
    default_numsteps = None # 10
    # create parser
    parser = argparse.ArgumentParser()
    # add required arguments
    parser.add_argument("-c", "--conf",     type=str, default=default_conf,     help="Configuration file [%s]" % default_conf)
    parser.add_argument("-m", "--mode",     type=str, default="run",            help="Which subprogram to run [run], one of [run, graphviz]")
    parser.add_argument("-n", "--numsteps", type=int, default=default_numsteps, help="Number of outer loop steps [%s]" % default_numsteps)
    parser.add_argument("-s", "--randseed",     type=int, default=None,             help="Random seed [None], seed is taken from config file")
    # parser.add_argument("-sp", "--saveplot", type=int, default=None,             help="Random seed [None], seed is taken from config file")
    # 
    # parse arguments
    args = parser.parse_args()
    # return arguments
    return args

def set_config_defaults(conf):
    if not conf['params'].has_key("numsteps"):
        conf['params']['numsteps'] = 100
    return conf

def set_config_commandline_args(conf, args):
    # for commandline_arg in conf['params'].has_key("numsteps"):
    #     conf['params']['numsteps'] = 100
    gparams = ['numsteps', 'randseed']
    for clarg in gparams:
        if getattr(args, clarg) is not None:
            conf['params'][clarg] = getattr(args, clarg)
    return conf

def make_expr_id_configfile(name = "experiment", configfile = "conf/default2.py"):
    """return experiment signature as name and timestamp"""
    confs = configfile.split("/")
    confs = confs[-1].split(".")[0]
    # print "configfile", confs
    return "%s_%s_%s" % (name, make_expr_sig(), confs)

def make_expr_id(name = "experiment"):
    pass

def make_expr_sig(args =  None):
    """return experiment timestamp"""
    return time.strftime("%Y%m%d_%H%M%S")

class Experiment(object):
    """!@brief Main experiment class

Arguments:
   args: argparse configuration namespace (key, value)


Load a config from the file in args.conf

    """

    gparams = ['ros', 'numsteps', 'recurrent', 'debug', 'dim', 'dt', 'showplot', 'saveplot', 'randseed']
    
    def __init__(self, args):
        global make_expr_id
        make_expr_id = partial(make_expr_id_configfile, configfile = args.conf)
        self.conf = get_config_raw(args.conf)
        assert self.conf is not None, "%s.init: Couldn't read config file %s" % (self.__class__.__name__, args.conf)
        self.conf = set_config_defaults(self.conf)

        if self.conf['params']['ros']:
            import rospy
            rospy.init_node("smp_graph")

        # update conf with commandline arguments
        self.conf = set_config_commandline_args(self.conf, args)
        
        # print "%s.init: conf keys = %s\n\n\n\n" % (self.__class__.__name__, self.conf.keys())
        
        for k in self.conf.keys():
            setattr(self, k, self.conf[k])
            # print "%s.init: self.%s = %s\n" % (self.__class__.__name__, k, getattr(self, k))
        # self.numsteps = self.conf['params']['numsteps']

        # print "Experiment.init\n", print_dict(self.conf)

        # self.topblock = Block(
        #     block = self.conf['block'],
        #     conf = self.conf['params'],
        # )
        
        self.topblock = Block2(conf = self.conf)

        # plot the computation graph and the bus
        set_interactive(True)
        
        # graph_fig = makefig(rows = 1, cols = 3, wspace = 0.1, hspace = 0.0,
        #                     axesspec = [(0, 0), (0, slice(1, None))], title = "Nxgraph and Bus")
        # # nxgraph_plot(self.topblock.nxgraph, ax = graph_fig.axes[0])
        # # flatten for drawing, quick hack
        # G = nxgraph_flatten(self.topblock.nxgraph)
        # # for node,noded in G.nodes_iter(data=True):
        # #     print "node", node, G.node[node], noded
        # G = nxgraph_add_edges(G)
        # # for edge in G.edges_iter():
        # #     print "edge", edge
        # nxgraph_plot(G, ax = graph_fig.axes[0], layout_type = "spring", node_size = 300)
        # # recursive_draw(self.topblock.nxgraph, ax = graph_fig.axes[0], node_size = 300, currentscalefactor = 0.1)
        # self.topblock.bus.plot(graph_fig.axes[1])
        # if self.conf['params']['saveplot']:
        #     filename = "data/%s_%s.%s" % (self.topblock.id, "graph_bus", 'jpg')
        #     graph_fig.savefig(filename, dpi=300, bbox_inches="tight")
        # # print self.conf['params']
        
        # print "print_dict\n", print_dict(self.conf)
    
    def run(self):
        print '#' * 80
        print "Init done, running the graph"
        print "{0}.run: numsteps = {1}".format(self.__class__.__name__, self.params['numsteps'])

        # TODO: try run
        #       except go interactive
        # import pdb
        # topblock_x = self.topblock.step(x = None)
        for i in xrange(self.params['numsteps']):
            # try:
            topblock_x = self.topblock.step(x = None)
            # except:
            # pdb.set_trace()
            # FIXME: progress bar / display        
            
        print "final return value topblock.x = %s" % (topblock_x)

        if self.conf['params']['showplot']:
            set_interactive(False)
            plt.show()

import networkx as nx
import re

class Graphviz(object):
    """!@brief Special experiment: Load runtime config into a networkx graph and plot it"""
    def __init__(self, args):
        # load graph config
        self.conf = get_config_raw(args.conf)
        assert self.conf is not None, "%s.init: Couldn't read config file %s" % (self.__class__.__name__, args.conf)

        # set the layout
        self.layouts = ["spring", "shell", "pygraphviz", "random"]
        self.layout  = self.layouts[2]

    def run(self):
        # create nx graph
        G = nx.MultiDiGraph()

        # FIXME: make the node and edge finding stuff into recursive functions
        #        to accomodate nesting and loops at arbitrary levels
        
        # pass 1: add the nodes
        for k, v in self.conf['params']['graph'].items():
            print "k", k, "v", v
            blockname = re.sub(r"<smp_graphs.block.*\.(.*) object.*", "\\1", v['block'])
            G.add_node(k, block = blockname)
            if v['params'].has_key('graph'): # hierarchical block containing subgraph
                for subk, subv in v['params']['graph'].items():
                    # print "sub", subk, subv
                    blockname = re.sub(r"<smp_graphs.block.*\.(.*) object.*", "\\1", subv['block'])
                    G.add_node(subk, block = blockname)
            elif v['params'].has_key('loopblock') and v['params'].has_key('blocksize'):
                if len(v['params']['loopblock']) < 1: continue
                # for subk, subv in v['params']['loopblock'].items():
                # print "sub", subk, subv
                # print k, print_dict(v['params'])
                lblock = v['params']['loopblock']
                # print "lblock", lblock, v['params']['blocksize']
                blockname = re.sub(r"<class 'smp_graphs.block.*\.(.*)'>", "\\1", lblock['block'])
                # print "block.id", lblock['params']['id']
                for i in range(v['params']['blocksize']):
                    k_from = lblock['params']['id'] + "/%d" % (i,)
                    G.add_node(k_from, block = blockname)
                    G.add_edge(k, k_from)
                    
            # print "k", k
            # print "v", v
            
        # pass 2: add the edges
        for k, v in self.conf['params']['graph'].items():
            # print "v['params']", v['params']
            # loop edges
            if v['params'].has_key('loopblock') and len(v['params']['loopblock']) == 0:

                # print "G", G[k]
                k_from = k.split("_")[0]
                G.add_edge(k_from, k)
            
            # input edges
            if not v['params'].has_key('inputs'): continue
            for inputkey, inputval in v['params']['inputs'].items():
                print "ink", inputkey
                print "inv", inputval
                if not inputval.has_key('bus'): continue
                # get the buskey for that input
                if inputval['bus'] not in ['None']:
                    k_from, v_to = inputval['bus'].split('/')
                    G.add_edge(k_from, k)

        # FIXME: add _loop_ and _containment_ edges with different color
        # print print_dict(pdict = self.conf[7:])

        # pass 3: create the layout

        nxgraph_plot(G, layout_type = self.layout)
        
        # layout = nxgraph_get_layout(G, self.layout)
                    
        # print G.nodes(data = True)
        # labels = {'%s' % node[0]: '%s' % node[1]['block'] for node in G.nodes(data = True)}
        # print "labels = %s" % labels
        # # nx.draw(G)
        # # nx.draw_networkx_labels(G)
        # # nx.draw_networkx(G, pos = layout, node_color = 'g', node_shape = '8')
        # nx.draw_networkx_nodes(G, pos = layout, node_color = 'g', node_shape = '8')
        # nx.draw_networkx_labels(G, pos = layout, labels = labels, font_color = 'r', font_size = 8, )
        # # print G.nodes()
        # e1 = [] # std edges
        # e2 = [] # loop edges
        # for edge in G.edges():
        #     # print edge
        #     if re.search("[_/]", edge[1]):
        #         e2.append(edge)
        #     else:
        #         e1.append(edge)

        # nx.draw_networkx_edges(G, pos = layout, edgelist = e1, edge_color = "g", width = 2)
        # nx.draw_networkx_edges(G, pos = layout, edgelist = e2, edge_color = "k")
        plt.show()
