"""Define, load and run the experiment defined in the configuration

.. moduleauthor:: Oswald Berthold, 2017

Experiment class provides the basic shell for running an experiment with methods for
 - running a graph
 - loading and drawing a graph (networkx)
"""

import argparse, os, re, sys
import time, datetime

from collections import OrderedDict
from functools import partial
# from types import FunctionType

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

# for config reading
from numpy import array

from smp_base.plot import set_interactive, makefig

from smp_graphs.block import Block2
from smp_graphs.utils import print_dict
from smp_graphs.common import conf_header, conf_footer, conf_strip_variables
from smp_graphs.common import md5, get_config_raw
from smp_graphs.graph import nxgraph_plot, recursive_draw, recursive_hierarchical
from smp_graphs.graph import nxgraph_flatten, nxgraph_add_edges, nxgraph_get_node_colors
from smp_graphs.graph import nxgraph_nodes_iter

################################################################################
# utils, TODO: move to utils.py
def get_args():
    """Get commandline arguments for an :class:`Experiment`

    Set up an :class:`argparse.ArgumentParser` and add common set of
    commandline arguments for controlling global experiment parameters.
    """
    # define defaults
    default_conf     = "conf/default.py"
    default_numsteps = None
    
    # create parser
    parser = argparse.ArgumentParser()
    
    # add commandline arguments
    parser.add_argument("-c", "--conf",       type=str, default=default_conf,     help="Configuration file [%s]" % default_conf)
    parser.add_argument("-dc", "--do-cache",  dest='docache', action='store_true', help="Enable experiment and block caching mechanisms [True].", default = True)
    parser.add_argument("-nc", "--no-cache",  dest='docache', action='store_false', help="Enable experiment and block caching mechanisms [True].")
    parser.add_argument("-dr", "--do-ros",    dest="ros", action="store_true",    default = None, help = "Do / enable ROS?")
    parser.add_argument("-nr", "--no-ros",    dest="ros", action="store_false",   default = None, help = "No / disable ROS?")
    parser.add_argument("-m", "--mode",       type=str, default="run",            help="Which subprogram to run [run], one of [run, graphviz]")
    parser.add_argument("-n", "--numsteps",   type=int, default=default_numsteps, help="Number of outer loop steps [%s]" % default_numsteps)
    parser.add_argument("-s", "--randseed",   type=int, default=None,             help="Random seed [None], if None, seed is taken from config file")
    parser.add_argument("-pg", "--plotgraph", dest="plotgraph", action="store_true", default = False, help = "Enable plot of smp graph [False]")
    parser.add_argument("-sp",  "--saveplot",     dest="saveplot",  action="store_true", default = None, help = "Enable saving the plots of this experiment [None]")
    parser.add_argument("-nsp", "--no-saveplot",  dest="saveplot",  action="store_false", default = None, help = "Disable saving the plots of this experiment [None]")

    # parse arguments
    args = parser.parse_args()
    
    # return arguments
    return args

def set_config_defaults(conf):
    """Experiment.py.set_config_defaults

    Set configuration defaults if they are missing
    """
    if not conf['params'].has_key("numsteps"):
        conf['params']['numsteps'] = 100
    return conf

def set_config_commandline_args(conf, args):
    """Experiment.py.set_config_commandline_args

    Set configuration params from commandline, used to override config file setting for quick tests
    """
    # for commandline_arg in conf['params'].has_key("numsteps"):
    #     conf['params']['numsteps'] = 100
    gparams = ['numsteps', 'randseed', 'ros', 'docache', 'saveplot']
    for clarg in gparams:
        if getattr(args, clarg) is not None:
            conf['params'][clarg] = getattr(args, clarg)
    return conf

def make_expr_id_configfile(name = "experiment", configfile = "conf/default2.py"):
    """Experiment.py.make_expr_id_configfile

    Make experiment signature from name and timestamp
    """
    # split configuration path
    confs = configfile.split("/")
    # get last element config filename
    confs = confs[-1].split(".")[0]
    # format and return
    return "%s_%s_%s" % (name, confs, make_expr_sig())

def make_expr_id(name = "experiment"):
    """Experiment.py.make_expr_id

    Dummy callback
    """
    pass

def make_expr_sig(args =  None):
    """Experiment.py.make_expr_sig

    Return formatted timestamp
    """
    return time.strftime("%Y%m%d_%H%M%S")

def make_expr_md5(obj):
    return md5(str(obj))

def set_random_seed(args):
    """set_random_seed
    
    Extract randseed parameter from args.conf, override with args.randseed if set and seed the numpy prng.

    Arguments:
    - args: argparse Namespace

    Returns:
    - randseed: the seed
    """
    assert hasattr(args, 'conf')
    randseed = 0
    
    conf = get_config_raw(args.conf, confvar = 'conf', fexec = False)

    pattern = re.compile('(randseed *= *[0-9]*)')
    # pattern = re.compile('.*(randseed).*')
    # print "pattern", pattern
    m = pattern.search(conf)
    # print "m[:] = %s" % (m.groups(), )
    # print "m[0] = %s" % (m.group(0), )
    # print "lv = %s" % (lv, )
    # m = re.search(r'(.*)(randseed *= *[0-9]*)', conf)
    # conf_ = re.sub(r'\n', r' ', conf)
    # conf_ = re.sub(r' +', r' ', conf_)
    # print "m", conf_
    # m = re.search(r'.*(randseed).*', conf)
    # print "m", m.group(1)
    if m is not None:
        code = compile(m.group(0), "<string>", "exec")
        gv = {}
        lv = {}
        exec(code, gv, lv)
        randseed = lv['randseed']
        # print "args.conf randseed match %s" % (randseed, )

    if hasattr(args, 'randseed') and args.randseed is not None:
        randseed = args.randseed

    # print "m", m
    # print "conf", conf
    # print "randseed", randseed
        
    np.random.seed(randseed)
    return randseed

class Experiment(object):
    """Experiment class

    Arguments:
    - args: argparse configuration namespace (key, value)

    Load a config from the file given in args.conf, initialize nxgraph from conf, run the graph
    """

    # global config file parameters
    gparams = ['ros', 'numsteps', 'recurrent', 'debug', 'dim', 'dt', 'showplot', 'saveplot', 'randseed']
    
    def __init__(self, args):
        """Experiment.__init__

        Experiment init

        Arguments:
        - args: argparse configuration namespace (key, value) containing args.conf
        """
        # get global func pointer
        global make_expr_id
        # point at other func, global make_expr_id is used in common (FIXME please)
        make_expr_id = partial(make_expr_id_configfile, name = 'smp', configfile = args.conf)

        # set random seed _before_ compiling conf
        set_random_seed(args)

        # get configuration from file # , this implicitly sets the id via global make_expr_id which is crap
        self.conf_localvars = get_config_raw(args.conf, confvar = None)
        # print "experiment.py conf_localvars", self.conf_localvars.keys()
        self.conf = self.conf_localvars['conf']
        # print "conf.params.id", self.conf['params']['id']
        assert self.conf is not None, "%s.init: Couldn't read config file %s" % (self.__class__.__name__, args.conf)
        # fill in missing defaults
        self.conf = set_config_defaults(self.conf)
        # update conf from commandline arguments
        self.conf = set_config_commandline_args(self.conf, args)

        # initialize ROS if needed
        if self.conf['params']['ros']:
            import rospy
            rospy.init_node("smp_graph")

        # store all conf entries in self
        # print "%s-%s.init\n" % (self.__class__.__name__, None),
        for k in self.conf.keys():
            setattr(self, k, self.conf[k])
            # selfattr = getattr(self, k)
            # if type(selfattr) is dict:
            #     print "        self.%s = %s\n" % (k, print_dict(selfattr))
            # else:
            #     print "        self.%s = %s\n" % (k, selfattr)
        """

        Hash functions
        - hashlib md5/sha
        - locally sensitive hashes, lshash. this is not independent of input size, would need maxsize kludge
        """
        # for hashing the conf, strip all entries with uncontrollably variable values (function pointers, ...)
        self.conf_ = conf_strip_variables(self.conf)
        print "numsteps", self.conf['params']['numsteps']
        print "numsteps_", self.conf_['params']['numsteps']
        # compute id hash of the experiment from the configuration dict string
        # print "experiment self.conf stripped", print_dict(self.conf_)
        m = make_expr_md5(self.conf_)
        # FIXME: 1) block class knows which params to hash, 2) append localvars to retain environment
        
        # update experiments database with the current expr
        xid = self.update_experiments_store(xid = m.hexdigest())
            
        # store md5 in params _after_ we computed the md5 hash
        self.conf['params']['md5'] = xid
        self.conf['params']['cached'] = self.params['docache'] and self.cache is not None and self.cache.shape[0]
        print "experiment %s cached with %s" % (self.conf['params']['md5'], self.conf['params']['cached'], )
        
        # instantiate topblock
        self.topblock = Block2(conf = self.conf, conf_localvars = self.conf_localvars)

        # plotting
        self.plotgraph_flag = args.plotgraph

    def update_experiments_store(self, xid = None):
        """Experiment.update_experiments_store

        Update the global store of experiments with the current one.

        The idea is to take a hash of the configuration and store the
        experiment's results with its hash as a key. If the experiment
        is rerun with the same config, only the logfile is loaded
        instead of recomputing everything.

        Storage options:
         1. a dict with pickle, fail
         2. tinydb, fail
         3. storage: hdf5 via pandas dataframe, works, current
         4. maybe upgrade to nosql / distributed a la mongodb, couchdb, or current favorite elasticsearch
        """

        # # debug configuration invariance vs. randseed and function pointers
        # f = open('conf-%s.txt' % (time.strftime('%H%M%S')), 'wa')
        # f.write(str(self.conf))
        # f.flush()

        # compute hash if it hasn't been supplied
        if xid is None:
            m = make_expr_md5(self.conf_)
            xid = m.hexdigest()

        # set the experiment's id
        self.conf['params']['id'] = make_expr_id() + "-" + xid

        # check no-cache argument
        if not self.params['docache']: return xid
        
        # prepare experiment database
        experiments_store = 'data/experiments_store.h5'
        columns = ['md5', 'timestamp', 'block', 'params']
        values = [[xid, pd.to_datetime(datetime.datetime.now()), str(self.conf['block']), str(self.conf['params'])]]
        # print "%s.update_experiments_store values = %s" % (self.__class__.__name__, values)
        # values = [[xid, self.conf['block'], self.conf['params']]]

        # load experiment database if one exists
        if os.path.exists(experiments_store):
            try:
                self.experiments = pd.read_hdf(experiments_store, key = 'experiments')
                print "Experiment.update_experiments_store loaded experiments_store = %s with shape = %s" % (experiments_store, self.experiments.shape)
                # search for hash
            except Exception, e:
                print "Loading store %s failed with %s" % (experiments_store, e)
                sys.exit(1)
        # create a new experiment database if it does not exist
        else:
            self.experiments = pd.DataFrame(columns = columns)

        # query database about current experiment
        self.cache = self.experiments[:][self.experiments['md5'] == xid]

        # load the cached experiment if it exists
        if self.cache is not None and self.cache.shape[0] != 0:
            print "Experiment.update_experiments_store found cached results = %s\n%s" % (self.cache.shape, self.cache)
        # store the experiment in the cache if it doesn't exist
        else:
            print "Experiment.update_experiments_store no cached results found, creating new entry"
            # temp dataframe
            df = pd.DataFrame(values, columns = columns, index = [self.experiments.shape[0]])

            dfs = [self.experiments, df]

            print "dfs", dfs
        
            # concatenated onto main df
            self.experiments = pd.concat(dfs)

        # write store 
        self.experiments.to_hdf(experiments_store, key = 'experiments')

        # return the hash
        return xid
        
    def plotgraph(self):
        """Experiment.plotgraph

        Show a visualization of the initialized top graph defining the
        experiment in 'self.topblock.nxgraph'.
        """
        # axesspec = [(0, 0), (0, 1), (0, 2), (0, slice(3, None))]
        # axesspec = [(0, 0), (0,1), (1, 0), (1,1)]
        axesspec = None
        fig_nxgr = makefig(
            rows = 1, cols = 1, wspace = 0.1, hspace = 0.1,
            axesspec = axesspec, title = "%s"  % (self.topblock.id.split('-')[0], )) # "nxgraph")
        fig_bus = makefig(
            rows = 1, cols = 1, wspace = 0.1, hspace = 0.1,
            axesspec = axesspec, title = "%s"  % (self.topblock.id.split('-')[0], ))
        axi = 0
        # nxgraph_plot(self.topblock.nxgraph, ax = fig_nxgr.axes[0])

        # # flatten for drawing, quick hack
        # G = nxgraph_flatten(self.topblock.nxgraph)
        # # # debug flattened graph
        # # for node,noded in G.nodes(data=True):
        # #     print "node.id = %s\n    .data = %s\n    .graphnode = %s\n" % (node, noded, G.node[node])

        # # add edges to flattened graph
        # G = nxgraph_add_edges(G)
        # # for edge in G.edges_iter():
        # #     print "experiment.plotgraph: edge after add_edges = %s" %( edge,)

        # # plot the flattened graph
        # nxgraph_plot(G, ax = fig_nxgr.axes[0], layout_type = "spring", node_size = 300)

        G_ = recursive_hierarchical(self.topblock.nxgraph)
        G_cols = nxgraph_get_node_colors(G_)
        print "G_cols", G_cols
        nxgraph_plot(G_, ax = fig_nxgr.axes[axi], layout_type = "linear_hierarchical", node_color = G_cols, node_size = 300)
        # fig_nxgr.axes[axi].set_aspect(1)
        axi += 1
        
        # # plot the nested graph
        # recursive_draw(
        #     self.topblock.nxgraph,
        #     ax = fig_nxgr.axes[2],
        #     node_size = 100,
        #     currentscalefactor = 1.0,
        #     shrink = 0.8)

        # plot the bus with its builtin plot method
        axi = 0
        (xmax, ymax) = self.topblock.bus.plot(fig_bus.axes[axi])
        print "experiment plotting bus xmax = %s, ymax = %s"  % (xmax, ymax)
        # fig_bus.axes[axi].set_aspect(1)
        fig_bus.set_size_inches((5, ymax / 25.0))
        axi += 1

        for fi, fig_ in enumerate([fig_nxgr, fig_bus]): # 
            # for ax in fig_.axes:
            #     xlim = ax.get_xlim()
            #     ylim = ax.get_ylim()
            #     print "fig", fig_, "ax xlim", xlim, "ylim", ylim
            #     width = (xlim[1] - xlim[0])/1.0
            #     height = (ylim[1] - ylim[0])/1.0
            #     fig_.set_size_inches((width, height))
            savetype = 'pdf'
            # save the plot if saveplot is set
            if self.params['saveplot']:
                filename = "data/%s_%s_%d.%s" % (self.topblock.id, 'graph_bus', fi, savetype)
                try:
                    print "Saving experiment graph plot to %s" % (filename, )
                    fig_.savefig(filename, dpi=300, bbox_inches="tight")
                except Exception, e:
                    print "Saving experiment graph plot to %s failed with %s" % (filename, e,)

    def printgraph_recursive(self, G, lvl = 0):
        indent = " " * 4 * lvl
        # iterate enabled nodes
        for node in nxgraph_nodes_iter(G, 'enable'):
            print "%snode = %s" % (indent, G.node[node]['block_'].id, )
            if hasattr(G.node[node]['block_'], 'nxgraph'):
                G_ = G.node[node]['block_'].nxgraph
                lvl += 1
                print "%sG%d.name = %s" % (indent, lvl, G_.name)
                print "%s  .nodes = %s" % (indent, ", ".join([G_.node[n]['params']['id'] for n in G_.nodes()]))
                self.printgraph_recursive(G = G_, lvl = lvl)
                
    def printgraph(self):
        print "\nPrinting graph\n", 
        G = self.topblock.nxgraph
        print "G.name  = %s" % (G.name,)
        # print "G.nodes = %s" % ([(G.node[n]['params']['id'], G.node[n].keys()) for n in G.nodes()])
        print " .nodes = %s" % (", ".join([G.node[n]['params']['id'] for n in G.nodes()]))

        self.printgraph_recursive(G, lvl = 1)
            
    def run(self):
        """Experiment.run

        Run the experiment by running the graph.
        """
        print '#' * 80
        print "Init done, running %s" % (self.topblock.nxgraph.name, )
        print "    Graph: %s" % (self.topblock.nxgraph.nodes(), )
        print "      Bus: %s" % (self.topblock.bus.keys(),)
        print " numsteps: {0}/{1}".format(self.params['numsteps'], self.topblock.numsteps)

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
            
        # print "final return value topblock.x = %s" % (topblock_x)

        # final writes: log store, experiment/block store?, graphics, models
        # self.topblock.bus.plot(fig_nxgr.axes[3])
        
        self.printgraph()
        
        # plot the computation graph and the bus
        set_interactive(True)
        if self.plotgraph_flag:
            self.plotgraph()

        if self.conf['params']['showplot']:
            set_interactive(False)
            plt.show()

import networkx as nx
import re

class Graphviz(object):
    """Graphviz class

    Load a runtime config into a networkx graph and plot it
    """
    def __init__(self, args):
        """Graphviz.__init__

        Initialize a Graphviz instance

        Arguments:
        - args: argparse configuration namespace (key, value)
        """
        # load graph config
        self.conf = get_config_raw(args.conf)
        assert self.conf is not None, "%s.init: Couldn't read config file %s" % (self.__class__.__name__, args.conf)

        # set the layout
        self.layouts = ["spring", "shell", "pygraphviz", "random"]
        self.layout  = self.layouts[2]

    def run(self):
        """Graphviz.run

        Run method
        """
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
