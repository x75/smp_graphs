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
import copy

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import PIL
from PIL import Image, ImageDraw, ImageFont

import numpy as np
import pandas as pd

# for config reading
from numpy import array

from smp_base.common import get_module_logger
from smp_base.plot import set_interactive, makefig

from smp_graphs.block import Block2
from smp_graphs.utils import print_dict
from smp_graphs.common import conf_header, conf_footer, conf_strip_variables
from smp_graphs.common import md5, get_config_raw, check_datadir
from smp_graphs.graph import nxgraph_plot, recursive_draw, recursive_hierarchical
from smp_graphs.graph import nxgraph_flatten, nxgraph_add_edges, nxgraph_get_node_colors
from smp_graphs.graph import nxgraph_nodes_iter, nxgraph_to_smp_graph
from smp_graphs.graph import nxgraph_load, nxgraph_store

# filter warnings
import warnings
warnings.filterwarnings('ignore',category=pd.io.pytables.PerformanceWarning)

from logging import INFO as logging_INFO
from logging import DEBUG as logging_DEBUG
# import logging
loglevel_DEFAULT = logging_INFO
logger = get_module_logger(modulename = 'experiment', loglevel = loglevel_DEFAULT)

# # 'application' code
# logger.debug('debug message')
# logger.info('info message')
# logger.warn('warn message')
# logger.error('error message')
# logger.critical('critical message')

# logging.basicConfig(
# logger = logging.getLogger(__name__)    # filename = 'example.log',
#     level = logging.DEBUG,
#     format = '%(levelname)s:%(message)s',
# )

# logger = logging.getLogger()
# print "logger handlers", logger.handlers

# logging.basicConfig(
#     format='%(levelname)s:%(message)s',
#     level=logging.DEBUG
# )

# logging.basicConfig(format='%(asctime)s %(message)s')

# logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

################################################################################
# utils, TODO: move to utils.py
def get_args():
    """Get commandline arguments for an :class:`Experiment`

    Set up an :class:`argparse.ArgumentParser` and add common set of
    commandline arguments for controlling global experiment parameters.
    """
    # define defaults
    default_conf     = "conf/default.py"
    default_datadir = 'data'
    default_numsteps = None
    
    # create parser
    parser = argparse.ArgumentParser()
    
    # add commandline arguments
    parser.add_argument("-c", "--conf",       type=str, default=default_conf,     help="Configuration file [%s]" % default_conf)
    parser.add_argument("-d", "--datadir",    type=str, default=default_datadir,  help="Data directory [%s]" % default_datadir)
    parser.add_argument("-dc", "--do-cache",  dest='docache', action='store_true', help="Enable experiment and block caching mechanisms [False].", default = False)
    parser.add_argument("-nc", "--no-cache",  dest='docache', action='store_false', help="Enable experiment and block caching mechanisms [True].")
    parser.add_argument("-dr", "--do-ros",    dest="ros", action="store_true",    default = None, help = "Do / enable ROS?")
    parser.add_argument("-nr", "--no-ros",    dest="ros", action="store_false",   default = None, help = "No / disable ROS?")
    parser.add_argument("-m", "--mode",       type=str, default="run",            help="Which subprogram to run [run], one of [run, graphviz]")
    parser.add_argument("-n", "--numsteps",   type=int, default=default_numsteps, help="Number of outer loop steps [%s]" % default_numsteps)
    parser.add_argument("-s", "--randseed",   type=int, default=None,             help="Random seed [None], if None, seed is taken from config file")
    parser.add_argument("-pg", "--plotgraph", dest="plotgraph", action="store_true", default = False, help = "Enable plot of smp graph [False]")
    parser.add_argument("-shp",  "--showplot",     dest="showplot",  action="store_true", default = None, help = "Show plots at all? [None]")
    parser.add_argument("-nshp",  "--no-showplot",     dest="showplot",  action="store_false", default = None, help = "Show plots at all? [None]")
    parser.add_argument("-sp",  "--saveplot",     dest="saveplot",  action="store_true", default = None, help = "Enable saving the plots of this experiment [None]")
    parser.add_argument("-nsp", "--no-saveplot",  dest="saveplot",  action="store_false", default = None, help = "Disable saving the plots of this experiment [None]")
    parser.add_argument("-cc", "--cache-clear",  dest='cache_clear', action='store_true', help="Clear the cache entry for cache hits of this experiment [False].", default = False)

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
        
    if not conf['params'].has_key('desc'):
        conf['params']['desc'] = conf['params']['id']
        
    # if conf['params'].has_key('lconf'):
    #     print "set_config_defaults", conf['params']['lconf']
    return conf

def set_config_commandline_args(conf, args):
    """Experiment.py.set_config_commandline_args

    Set configuration params from commandline, used to override config file setting for quick tests
    """
    # for commandline_arg in conf['params'].has_key("numsteps"):
    #     conf['params']['numsteps'] = 100
    gparams = ['numsteps', 'randseed', 'ros', 'docache', 'saveplot', 'showplot', 'cache_clear']
    for clarg in gparams:
        if getattr(args, clarg) is not None:
            conf['params'][clarg] = getattr(args, clarg)
    return conf

def make_expr_id_configfile(name = "experiment", configfile = "conf/default2.py", timestamp = True):
    """Experiment.py.make_expr_id_configfile

    Make experiment signature from name and timestamp
    """
    # split configuration path
    confs = configfile.split("/")
    # get last element config filename, split at '.' to remove '.py'
    confs = confs[-1].split(".")[0]
    # format and return
    if timestamp:
        return "%s_%s_%s" % (name, confs, make_expr_sig())
    else:
        return "%s_%s" % (name, confs)

def make_expr_id(name = "experiment"):
    """Experiment.py.make_expr_id

    Dummy callback
    """
    pass

def make_expr_sig(args =  None):
    return make_timestamp_Ymd_HMS(args = args)

def make_timestamp_Ymd_HMS(args =  None, timestamp_format = None):
    """Experiment.py.make_expr_sig

    Return Ymd_HMS-formatted timestamp
    """
    if timestamp_format is None:
        timestamp_format = "%Y%m%d_%H%M%S"
    return time.strftime(timestamp_format)

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
        # global make_expr_id
        # point at other func, global make_expr_id is used in common (FIXME please)
        make_expr_id = partial(make_expr_id_configfile, name = 'smp', configfile = args.conf, timestamp = False)

        # set random seed _before_ compiling conf
        set_random_seed(args)

        #: load, compile and run configuration from file
        #:  - returns the variables 'vars' from its eval context
        #:  - special item vars['conf'] is explicit top block configuration 'conf'
        #:  - all keys in conf['params'] should match the variables in vars
        self.conf_vars = get_config_raw(args.conf, confvar = None)
        # print "experiment.py conf_vars", self.conf_vars.keys()
        self.conf = self.conf_vars['conf']
        # print "conf.params.id", self.conf['params']['id']
        assert self.conf is not None, "%s.init: Couldn't read config file %s" % (self.__class__.__name__, args.conf)
        # check consistency
        # for pk, pv in self.conf['params'].items():
        #     if self.conf_vars.has_key(pk):
        #         print "    %s - params[%s] = %s, vars[%s] = %s" % (pv == self.conf_vars[pk], pk, type(pv), pk, type(self.conf_vars[pk]))
        #     else:
        #         print "    BANG params/vars mismatch on %s" % (pk,)

        logger.info('#' * 80)
        logger.info("experiment.Experiment init with conf = %s" % (self.conf.keys(), ))
        
        # topblock outputs: new types in addition to np.ndarray signals: 'text', 'plot', ...
        for paramkey in ['outputs', 'desc']:
            if self.conf_vars.has_key(paramkey):
                self.conf['params'][paramkey] = self.conf_vars[paramkey]
                logger.debug("    vars -> params found %s, %s" % (paramkey, self.conf['params'][paramkey], ))
        
        # fill in missing defaults
        self.conf = set_config_defaults(self.conf)
        # update conf from commandline arguments
        self.conf = set_config_commandline_args(self.conf, args)

        # print self.conf['params']['desc']
        
        # initialize ROS if needed
        if self.conf['params']['ros']:
            import rospy
            rospy.init_node("smp_graph")
            self.conf['params']['roscore'] = rospy.core
            
        # store all conf entries in self
        for k, v in self.conf.items():
            setattr(self, k, v)
            
            # selfattr = getattr(self, k)
            # if type(selfattr) is dict:
            #     print "        self.%s = %s\n" % (k, print_dict(selfattr))
            # else:
            #     print "        self.%s = %s\n" % (k, selfattr)

        self.conf['params']['cache_clear'] = self.params['cache_clear']
        logger.debug("self.params.keys() = %s", self.params.keys())
        
        # print self.desc
        self.args = args
        """

        Hash functions
        - hashlib md5/sha
        - locally sensitive hashes, lshash. this is not independent of input size, would need maxsize kludge
        """
        # for hashing the conf, strip all entries with uncontrollably variable values (function pointers, ...)
        self.conf_ = conf_strip_variables(copy.deepcopy(self.conf))
        # print "numsteps", self.conf['params']['numsteps']
        # print "numsteps_", self.conf_['params']['numsteps']
        # compute id hash of the experiment from the configuration dict string
        # print "experiment self.conf stripped", print_dict(self.conf_)
        m = make_expr_md5(self.conf_)
        # FIXME: 1) block class knows which params to hash, 2) append localvars to retain environment

        # get id from config file name
        self.conf['params']['id'] = make_expr_id()
        
        # cache: update experiments database with the current expr
        xid = self.check_experiments_store(xid = m.hexdigest())
            
        # store md5 in params _after_ we computed the md5 hash
        # set the experiment's id
        # self.conf['params']['id'] = make_expr_id() + "-" + xid
        # self.conf['params']['id'] = 
        self.conf['params']['timestamp'] = make_timestamp_Ymd_HMS()
        self.conf['params']['md5'] = xid
        self.conf['params']['datadir'] = self.args.datadir
        self.conf['params']['datadir_expr'] = '%s/%s' % (
            self.args.datadir,
            self.conf['params']['id'])
        self.conf['params']['datafile_md5'] = '%s/%s_%s' % (
            self.conf['params']['datadir_expr'],
            self.conf['params']['id'],
            self.conf['params']['md5'],
        )
        self.conf['params']['datafile_expr'] = '%s/%s_%s_%s' % (
            self.conf['params']['datadir_expr'],
            self.conf['params']['id'],
            self.conf['params']['md5'],
            self.conf['params']['timestamp'])
        self.conf['params']['docache'] = self.params['docache'] and self.cache is not None and self.cache.shape[0]
        # print "    experiment %s docache with %s" % (self.conf['params']['md5'], self.conf['params']['docache'], )

        # FIXME: check / create logging dir in data/experiment-id-and-hash
        if not check_datadir(conf = self.conf['params']):
            print "Fail creating directories"
            sys.exit(1)
        
        # instantiate topblock
        self.top = Block2(conf = self.conf, conf_vars = self.conf_vars)

        # plotting
        self.init_plotgraph(args)

    def init_plotgraph(self, args):
        self.plotgraph_flag = args.plotgraph
        if self.plotgraph_flag:
            self.plotgraph_figures = {}

            # graph plot
            self.plotgraph_savetype = 'pdf'
            self.plotgraph_filename = "%s/%s_%s_%d.%s" % (
                self.conf['params']['datadir_expr'], self.top.id, 'nxgraph', 1,
                self.plotgraph_savetype)
            
            self.plotgraph_figures['nxgraph'] = {
                'type': 'fig',
                'fig': None,
                'filename': self.plotgraph_filename,
                'label': self.top.id,
                'id': 'graph',
                'desc': 'Graph (nxgraph)',
                # 'width': '378.52pt',
                'width': 0.59,
            }

            # bus plot
            self.plotgraph_savetype = 'png' # 'pgm' # 'jpg'
            self.plotgraph_filename = "%s/%s_%s_%d.%s" % (
                self.conf['params']['datadir_expr'], self.top.id, 'bus', 1,
                self.plotgraph_savetype)
            
            self.plotgraph_figures['bus'] = {
                'type': 'fig',
                'fig': None,
                'filename': self.plotgraph_filename,
                'label': self.top.id,
                'id': 'bus',
                'desc': 'Bus',
                # 'width': '320pt',
                'width': 0.39,
            }


            # plotgraph_figures = [
            #     dict([(ik, iv) for ik, iv in v.items() if ik not in ['type', 'fig', 'label']]) for k, v in self.plotgraph_figures.items()]
            
            plotgraph_figures = dict([(k, [fv[k] for fk, fv in self.plotgraph_figures.items()]) for k in ['filename', 'id', 'desc', 'width']])
            logger.debug("plotgraph_figures = %s", plotgraph_figures)
            # copy to outputs for latex figures
            self.top.outputs['graph-bus'] = {
                'type': 'fig', 'fig': None,
                'filename': plotgraph_figures['filename'],
                'label': self.top.id,
                'id': plotgraph_figures['id'],
                'desc': plotgraph_figures['desc'],
                'width': plotgraph_figures['width'],
            }
            
    def check_experiments_store(self, xid = None):
        """Experiment.check_experiments_store

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

        # check no-cache argument
        if not self.params['docache']: return xid
        
        # prepare experiment database
        self.experiments_store = '%s/experiments_store.h5' % (self.args.datadir, )
        columns = ['id', 'timestamp', 'md5', 'topblock', 'params', 'topblock_nxgraph', 'topblock_bus']
        values = [[
            self.params['id'], pd.to_datetime(datetime.datetime.now()),
            xid,
            str(self.conf['block']), str(self.conf['params']), '', '']]

        # pandas PerformanceWarning type check
        # for k, v in zip(columns, values[0]):
        #     print "    df[%s].type = %s" % (k, type(v))
        
        # print "%s.check_experiments_store values = %s" % (self.__class__.__name__, values)
        # values = [[xid, self.conf['block'], self.conf['params']]]

        # print "Experiment.check_experiments_store"
        
        # load experiment database if one exists
        if os.path.exists(self.experiments_store):
            try:
                self.experiments = pd.read_hdf(self.experiments_store, key = 'experiments')
                print "    loaded experiments_store = %s with shape = %s" % (
                    self.experiments_store, self.experiments.shape)
                
                # search for hash
            except Exception, e:
                print "    loading store %s failed with %s, continuing without cache" % (self.experiments_store, e)
                return xid
                # sys.exit(1)
        # create a new experiment database if it does not exist
        else:
            self.experiments = pd.DataFrame(columns = columns)

        # query database about current experiment
        self.cache_index = -1
        if self.experiments.shape[0] > 0:
            self.cache_index = self.experiments.index[-1]

        # print "md5", self.experiments.md5
        # print "xid", xid, self.experiments.md5 == xid
        self.cache = self.experiments[:][self.experiments.md5 == xid]
        self.cache_loaded = False

        def new_cache_entry(values, columns, index):
            df = pd.DataFrame(data = values, columns = columns, index = index)
            dfs = [self.experiments, df]
            # print "dfs", dfs
        
            # concatenated onto main df
            self.experiments = pd.concat(dfs)
            # experiment.cache is the newly created entry
            self.cache = df
        
        # load the cached experiment if it exists
        if self.cache is not None and self.cache.shape[0] != 0:
            # experiment.cache is the loaded entry
            print "    found cached results = %s\n    %s\n    %s" % (self.cache.shape, '', '') #, self.cache, self.experiments.index)
            self.cache_loaded = True

            print "self.params.keys()", self.params.keys()
            if self.params['cache_clear']:
                # print "    cache found, dropping and recreating it", type(self.experiments)
                # self.experiments.drop(index = [(self.experiments.md5 == xid).argmax()])
                hits = self.experiments.md5 == xid
                # print "    hits = %s, %s" % (hits.index[hits], type(hits))
                # print "expr hits", hits
                # hits *= np.arange(len(hits))
                # print "expr hits", hits
                # for hit in hits.nonzero():
                for hit in hits.index[hits]:
                    # print "    hit = %s/%s" % (type(hit), hit)
                    hit_ = hit
                    self.experiments = self.experiments.drop(index = [hit_])
            new_cache_entry(values, columns, index = [self.cache_index + 1])
        # store the experiment in the cache if it doesn't exist
        else:
            print "    no cached results found, creating new entry"
            # temp dataframe
            # self.experiments.index[-1] + 1
            new_cache_entry(values, columns, index = [self.cache_index + 1])

        # write store
        self.experiments.to_hdf(self.experiments_store, key = 'experiments')

        # return the hash
        return xid
        
    def plotgraph(self, G = None, Gbus = None, G_cols = None):
        """Experiment.plotgraph

        Show a visualization of the initialized top graph defining the
        experiment in 'self.top.nxgraph'.
        """
        # axesspec = [(0, 0), (0, 1), (0, 2), (0, slice(3, None))]
        # axesspec = [(0, 0), (0,1), (1, 0), (1,1)]
        axesspec = None
        plotgraph_figures = list()
        figtitle = self.top.id.split('-')[0]
        # FIXME: uses makefig directly from smp_base.plot instead of FigPlotBlock2
        fig_nxgr = makefig(
            rows = 1, cols = 1, wspace = 0.1, hspace = 0.1,
            axesspec = axesspec, title = "%s"  % (figtitle, )) # "nxgraph")
        axi = 0
        # nxgraph_plot(self.top.nxgraph, ax = fig_nxgr.axes[0])

        # # flatten for drawing, quick hack
        # G = nxgraph_flatten(self.top.nxgraph)
        # # # debug flattened graph
        # # for node,noded in G.nodes(data=True):
        # #     print "node.id = %s\n    .data = %s\n    .graphnode = %s\n" % (node, noded, G.node[node])

        # # add edges to flattened graph
        # G = nxgraph_add_edges(G)
        # # for edge in G.edges_iter():
        # #     print "experiment.plotgraph: edge after add_edges = %s" %( edge,)

        # # plot the flattened graph
        # nxgraph_plot(G, ax = fig_nxgr.axes[0], layout_type = "spring", node_size = 300)

        assert G is not None
        # if G is None:
        #     G = self.top.nxgraph
            
        # G_, G_number_of_nodes_total = recursive_hierarchical(G)
        G_ = G
        G_number_of_nodes_total = G_.number_of_nodes()

        if G_cols is None:
            G_cols = nxgraph_get_node_colors(G_)
            
        # print "G_cols", G_cols
        # nxgraph_plot(G_, ax = fig_nxgr.axes[axi], layout_type = "linear_hierarchical", node_color = G_cols, node_size = 300)
        nxgraph_plot(G_, ax = fig_nxgr.axes[axi], layout_type = "spring", node_color = G_cols, node_size = 300)
        # fig_nxgr.axes[axi].set_aspect(1)
        axi += 1
        # plotgraph_figures.append(fig_nxgr)
        self.plotgraph_figures['nxgraph']['fig'] = fig_nxgr
        
        # # plot the nested graph
        # recursive_draw(
        #     self.top.nxgraph,
        #     ax = fig_nxgr.axes[2],
        #     node_size = 100,
        #     currentscalefactor = 1.0,
        #     shrink = 0.8)

        assert Gbus is not None
        # if Gbus is None:
        #     Gbus = self.top.bus

        if len(Gbus) < 0:
            # plot the bus with its builtin plot method
            fig_bus = makefig(
                rows = 1, cols = 1, wspace = 0.1, hspace = 0.1,
                axesspec = axesspec, title = "%s"  % (figtitle, ))

            
            axi = 0
            (xmax, ymax) = Gbus.plot(fig_bus.axes[axi])
            print "experiment plotting bus xmax = %s, ymax = %s"  % (xmax, ymax)
            # fig_bus.axes[axi].set_aspect(1)
            fig_bus.set_size_inches((5, ymax / 25.0))
            axi += 1

            # plotgraph_figures.append(fig_bus)
            self.plotgraph_figures['bus']['fig'] = fig_bus
        else:
            # bustxt = "%s\n\nbus\n%s" % (figtitle, Gbus.astable(), )
            # bustxt = "{0}\n\n{1:^40}\n\n{2}".format(figtitle, 'signal bus', Gbus.astable(), )

            # configure
            img_figtitle = "{0}".format(figtitle)
            img_figsubtitle = "{0:^40}".format('signal bus')
            img_bustable = "{0}".format(Gbus.astable(loop_compress = True))

            fontsize = 12
            lineheight = int(fontsize * 1.4)
            width = 320
            height = (img_bustable.count('\n') + 5) * lineheight
            offset_y = 10
            colwidth = 170

            # init image objects
            # image = Image.new("RGBA", (width,height), (255,255,255))
            image = Image.new(mode = "L", size = (width, height), color = 1.0)
            draw = ImageDraw.Draw(image)
            # font = ImageFont.truetype("/usr/share/fonts/truetype/ttf-dejavu/DejaVuSans.ttf", fontsize)
            font_normal = ImageFont.truetype("DejaVuSans.ttf", fontsize)
            font_bold = ImageFont.truetype("DejaVuSans-Bold.ttf", fontsize)
            # font_normal = ImageFont.truetype("FreeSans.ttf", fontsize)
            # font_bold = ImageFont.truetype("FreeSansBold.ttf", fontsize)
            fontcolor = (0,0,0)
            fontcolor = 0

            # draw header
            draw.text((10, offset_y + 0), img_figtitle, fontcolor, font = font_bold)
            draw.text((10, offset_y + 20), img_figsubtitle, fontcolor, font = font_normal)

            # draw bus items
            
            # keys = Gbus.keys()
            # for i, k in enumerate(keys):
            Gbus_items = Gbus.keys_loop_compress()
            Gbus_keys  = Gbus_items.keys()
            Gbus_keys.sort()
            
            # for i, item in enumerate(Gbus_items.items()):
            for i, k in enumerate(Gbus_keys):
                # k = item[0]
                item = (k, Gbus_items[k])
                v = item[1]['v']
                c = item[1]['cnt']
                draw.text(
                    (10, offset_y + (i * lineheight) + 40), '| %s[%d]' % (k, c), fontcolor, font = font_normal)
                draw.text(
                    (10 + colwidth, offset_y + (i * lineheight) + 40), '| %s' % (v.shape,), fontcolor, font = font_normal)
            # imgage_s = image.resize((188,45), Image.ANTIALIAS)
            # image_s = image.resize((width, height), Image.ANTIALIAS)
            self.plotgraph_figures['bus']['fig'] = image
            # filename = re.sub('\.%s' % (savetype, '.jpg', self.plotgraph_figures['bus']['filename']
            # img_resized.save()
            
            # if self.conf['params']['showplot']:
            #     image.show()

        # for fi, fig_ in enumerate(plotgraph_figures): #
        for plotk, plotv in self.plotgraph_figures.items():
            # for ax in fig_.axes:
            #     xlim = ax.get_xlim()
            #     ylim = ax.get_ylim()
            #     print "fig", fig_, "ax xlim", xlim, "ylim", ylim
            #     width = (xlim[1] - xlim[0])/1.0
            #     height = (ylim[1] - ylim[0])/1.0
            #     fig_.set_size_inches((width, height))
            # save the plot if saveplot is set
            if self.params['saveplot']:
                # savetype = self.plotgraph_savetype
                fig_ = plotv['fig']
                filename = plotv['filename'] # self.plotgraph_filename
                try:
                    logger.info("Saving experiment graph plot of type = %s to %s" % (type(fig_), filename, ))
                    if type(fig_) is Figure:
                        fig_.savefig(filename, dpi=300, bbox_inches="tight")
                    elif type(fig_) is PIL.Image.Image:
                        fig_.save(filename)
                    else:
                        print "            watn scheis\n\n\n\n"
                except Exception, e:
                    logger.error("Saving experiment graph plot to %s failed with %s" % (filename, e,))

    def printgraph_recursive(self, G, lvl = 0):
        indent = " " * 4 * lvl
        # iterate enabled nodes
        for node in nxgraph_nodes_iter(G, 'enable'):
            if G.node[node].has_key('block_'):
                # nodedata_key = 'block_'
                print "%snode = %s" % (indent, G.node[node]['block_'].id, )
                if hasattr(G.node[node]['block_'], 'nxgraph'):
                    G_ = G.node[node]['block_'].nxgraph
                    lvl += 1
                    print "%sG%d.name = %s" % (indent, lvl, G_.name)
                    print "%s  .nodes = %s" % (indent, ", ".join([G_.node[n]['params']['id'] for n in G_.nodes()]))
                    self.printgraph_recursive(G = G_, lvl = lvl)
            else:
                # nodedata_key = 'params'
                print "%snode = %s" % (indent, G.node[node]['params']['id'], )

    def printgraph(self, G = None):
        print "\nPrinting graph\n",
        # if G is None:
        #     G = self.top.nxgraph
        assert G is not None
            
        print "G.name  = %s" % (G.name,)
        # print "G.nodes = %s" % ([(G.node[n]['params']['id'], G.node[n].keys()) for n in G.nodes()])
        print " .nodes = %s" % (", ".join([G.node[n]['params']['id'] for n in G.nodes()]))

        self.printgraph_recursive(G, lvl = 1)
            
    def run(self):
        """Experiment.run

        Run the experiment by running the graph.
        """
        logger.info('#' * 80)
        logger.info("Init done, running %s" % (self.top.nxgraph.name, ))
        logger.info("    Graph: %s" % (self.top.nxgraph.nodes(), ))
        logger.info("      Bus: %s" % (self.top.bus.keys(),))
        logger.info(" numsteps: {0}/{1}".format(self.params['numsteps'], self.top.numsteps))

        # # logging
        # logger.debug("logger handlers = %s", logger.handlers)

        # TODO: try run
        #       except go interactive
        # import pdb
        # topblock_x = self.top.step(x = None)
        for i in xrange(0, self.params['numsteps'], self.top.blocksize_min):
            # print "experiment.py run i = %d" % (i, )
            # try:
            topblock_x = self.top.step(x = None)
            # except:
            # pdb.set_trace()
            # FIXME: progress bar / display        
            
        # print "final return value topblock.x = %s" % (topblock_x)

        # final writes: log store, experiment/block store?, graphics, models
        # self.top.bus.plot(fig_nxgr.axes[3])

        # initial run, no cached data: store the graph
        if self.conf['params']['docache'] and not self.cache_loaded:
            logger.info("experiment cache: storing final top level nxgraph = %s", self.top.nxgraph)
            # store the full dynamically expanded state of the toplevel nxgraph
            nxgraph_store(conf = self.conf['params'], G = self.top.nxgraph)
            self.top.bus.store_pickle(conf = self.conf['params'])
            
            # filename = "data/%s_%s.yaml" % (self.top.id, 'nxgraph',)
            # nx.write_yaml(self.top.nxgraph, filename)
            # self.cache['topblock_nxgraph'] = filename
            # self.cache['topblock_bus'] = str(self.top.bus)
            # print "    topblock.nxgraph", self.cache['topblock_nxgraph']
            # print "    topblock.bus", self.cache['topblock_bus']
            # update the experiment store
            # self.check_experiments_store(xid = self.conf['params']['md5'])
            # write store
            # self.experiments.to_hdf(self.experiments_store, key = 'experiments')
            # G = self.top.nxgraph
            # Gbus = self.top.bus

        if self.conf['params']['docache']:
            from smp_graphs.block import Bus
            # G = self.cache['topblock_nxgraph']
            # Gbus = self.cache['topblock_bus']
            logger.info("experiment cache: loading cached top level nxgraph")
            G = nxgraph_load(conf = self.conf['params'])
            Gbus = Bus.load_pickle(conf = self.conf['params'])
            # print "G", G, G.number_of_nodes()
            # print "Gbus", Gbus
            # G = self.top.nxgraph # nx.read_yaml(self.cache['topblock_nxgraph'])
            # Gbus = self.top.bus
        else:
            G, G_number_of_nodes_total = recursive_hierarchical(self.top.nxgraph)
            Gbus = self.top.bus

        # self.printgraph(G = G)
        
        # plot the computation graph and the bus
        set_interactive(True)
        if self.plotgraph_flag:
            self.plotgraph(G = G, Gbus = Gbus)

        if self.conf['params']['showplot']:
            set_interactive(False)
            plt.show()

        # close files

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
                # if not inputval.has_key('bus'): continue
                if inputval.has_key('bus'):
                    # get the buskey for that input
                    if inputval['bus'] not in ['None']:
                        k_from, v_to = inputval['bus'].split('/')
                        G.add_edge(k_from, k)
                if inputval.has_key('trigger'):
                    if inputval['trigger'] not in ['None']:
                        k_from, v_to = inputval['trigger'].split('/')
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
