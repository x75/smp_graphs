"""smp_graphs - smp sensorimotor experiments as computation graphs

2017 Oswald Berthold

experiment: basic experiment shell for
 - running a graph
 - loading and drawing a graph (networkx)
"""

import argparse
import time

from collections import OrderedDict

import matplotlib.pyplot as plt

import numpy as np

# for config reading
from numpy import array

from smp_graphs.block import Block2
from smp_graphs.utils import print_dict

################################################################################
# static templates
conf_header = """
from smp_graphs.experiment import make_expr_id

from collections import OrderedDict

from smp_graphs.block import Block2, ConstBlock2, UniformRandomBlock2
from smp_graphs.block import LoopBlock2
from smp_graphs.block import FileBlock2
from smp_graphs.block_plot import TimeseriesPlotBlock2

from smp_base.plot import timeseries, histogram, rp_timeseries_embedding

import numpy as np

debug = False
"""

conf_footer = """
# top block
# FIXME: would like to get rid of this, common for all conf
conf = {
    'block': Block2,
    'params': {
        'id': make_expr_id(),
        'debug': debug,
        'topblock': True,
        'numsteps': numsteps,
        'graph': graph,
    }
}
"""

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
    # 
    # parse arguments
    args = parser.parse_args()
    # return arguments
    return args

def get_config_raw(conf):
    # open and read config file containing a dictionary spec of the graph
    s_ = open(conf, "r").read()

    # prepend / append header and footer
    s   = "%s\n%s\n%s" % (conf_header, s_, conf_footer)

    # load config by running the code string
    code = compile(s, "<string>", "exec")
    global_vars = {}
    local_vars  = {}
    exec(code, global_vars, local_vars)

    # conf = local_vars["conf"]
    # print "conf", conf
    # return conf
    return local_vars["conf"]

def set_config_defaults(conf):
    if not conf['params'].has_key("numsteps"):
        conf['params']['numsteps'] = 100
    return conf

def make_expr_id(name = "experiment"):
    """return experiment signature as name and timestamp"""
    return "%s_%s" % (name, make_expr_sig())

def make_expr_sig(args =  None):
    """return experiment timestamp"""
    return time.strftime("%Y%m%d_%H%M%S")

class Experiment(object):
    """smp_graphs Experiment

Arguments:
   args: argparse configuration namespace (key, value)


Load a config from the file in args.conf

    """
    def __init__(self, args):
        self.conf = get_config_raw(args.conf)
        self.conf = set_config_defaults(self.conf)
        
        # print "%s.init: conf keys = %s\n" % (self.__class__.__name__, self.conf.keys())
        
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

        # print self.conf['params']
        
        # print "print_dict\n", print_dict(self.conf)
    
    def run(self):
        print "%s.run: conf['numsteps'] = %d" % (self.__class__.__name__, self.params['numsteps'])
        
        # topblock_x = self.topblock.step(x = None)
        for i in xrange(self.params['numsteps']):
            topblock_x = self.topblock.step(x = None)
            # FIXME: progress bar / display

        print "final return value topblock.x = %s" % (topblock_x)

        plt.show()

import networkx as nx
import re
class Graphviz(object):
    def __init__(self, args):
        self.conf = get_config_raw(args.conf)
        # f = open(args.conf, "r")
        # self.conf = f.read()

    def run(self):
        G = nx.MultiDiGraph()
        # pass one add nodes
        for k, v in self.conf['params']['graph'].items():
            blockname = re.sub(r"<smp_graphs.block.*\.(.*) object.*", "\\1", v['block'])
            G.add_node(k, block = blockname)
            # print "k", k
            # print "v", v
            
        # pass two add edges
        for k, v in self.conf['params']['graph'].items():
            # print "v['params']", v['params']
            if not v['params'].has_key('inputs'): continue
            for inputkey, inputval in v['params']['inputs'].items():
                # print inputkey
                # print inputval
                if inputval[2] not in ['None']:
                    k_from, v_to = inputval[2].split('/')
                    G.add_edge(k_from, k)
        # print print_dict(pdict = self.conf[7:])
        layout = nx.spring_layout(G)
        print G.nodes(data = True)
        labels = {'%s' % node[0]: '%s' % node[1]['block'] for node in G.nodes(data = True)}
        print "labels = %s" % labels
        # nx.draw(G)
        # nx.draw_networkx_labels(G)
        nx.draw_networkx(G, pos = layout, node_color = 'g', node_shape = '8')
        nx.draw_networkx_labels(G, pos = layout, labels = labels, font_color = 'r')
        plt.show()
