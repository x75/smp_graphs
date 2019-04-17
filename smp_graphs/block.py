"""Block2 is the basic computation block and implements *smp_graphs*
support structure.

.. moduleauthor:: 2017 Oswald Berthold

The Base class :class:`Block2` is supported by two decorators,
:class:`decInit` and :class:`decStep`, and by the :class:`Bus` class
which extends a `dict` with signal routing semantics.
"""

import pdb
import uuid, sys, time, copy, re
import itertools, pickle

from collections import OrderedDict, MutableMapping
from functools import partial, wraps

# import lshash

import networkx as nx

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from smp_base.common import get_module_logger
from smp_base.common import dict_search_recursive
from smp_base.plot import plot_colors, makefig
from smp_base.plot_utils import set_interactive

import smp_graphs.utils_logging as log
from smp_graphs.utils import print_dict, ordereddict_insert, xproduct, myt

from smp_graphs.common import conf_header, conf_footer, get_input, conf_strip_variables
from smp_graphs.common import md5, get_config_raw, get_config_raw_from_string

from smp_graphs.common import set_attr_from_dict, loop_delim, compress_loop_id
from smp_graphs.common import dict_get_nodekeys_recursive, dict_replace_nodekeys_loop
from smp_graphs.common import dict_replace_idstr_recursive2
from smp_graphs.common import dict_replace_idstr_recursive
from smp_graphs.common import tuple2inttuple

from smp_graphs.graph import nxgraph_from_smp_graph, nxgraph_to_smp_graph
from smp_graphs.graph import nxgraph_node_by_id, nxgraph_node_by_id_recursive
from smp_graphs.graph import nxgraph_nodes_iter

from logging import WARNING as logging_WARNING
from logging import INFO as logging_INFO
from logging import DEBUG as logging_DEBUG
from functools import reduce
# import logging
loglevel_DEFAULT = logging_INFO
logger = get_module_logger(modulename = 'block', loglevel = logging_DEBUG)

# finally, ros
# import rospy
# from std_msgs.msg import Float64MultiArray

# caching joblib
from joblib import Memory
location = './cachedir'
memory = Memory(location, verbose=0)

# FIXME: make it optional in core
from hyperopt import STATUS_OK, STATUS_FAIL

BLOCKSIZE_MAX = 10000

# snacked from http://matplotlib.org/mpl_examples/color/colormaps_reference.py
block_cmaps = dict([('perceptually_uniform_sequential', [
            'viridis', 'plasma', 'inferno', 'magma']),
         ('sequential', [
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),
         ('sequential2', [
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper']),
         ('diverging', [
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']),
         ('qualitative', [
            'Pastel1', 'Pastel2', 'Paired', 'Accent',
            'Dark2', 'Set1', 'Set2', 'Set3',
            'tab10', 'tab20', 'tab20b', 'tab20c']),
         ('miscellaneous', [
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
            'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar'])])

block_groups = {
    'graph': {'cmap': block_cmaps['sequential2'][5]}, # 'bone'},
    'data': {'cmap': block_cmaps['sequential2'][6]}, # 'pink'},
    'comp': {'cmap': block_cmaps['sequential2'][7]}, # 'hot'},
    'measure': {'cmap': block_cmaps['sequential2'][8]}, # 'cool'},
    'output': {'cmap': block_cmaps['sequential2'][9]}, # 'copper'},
}

####################################################################
# smp_graphs types: create some types for use in configurations like
# const, bus, generator, func, ...

##################################################################################
# bus class
# from http://stackoverflow.com/questions/3387691/how-to-perfectly-override-a-dict
class Bus(MutableMapping):
    """Bus class

    A dictionary that applies an arbitrary key-altering function
    before accessing the keys.
    """

    def __init__(self, *args, **kwargs):
        self.store = dict()
        self.update(dict(*args, **kwargs))  # use the free update to set keys

    def __getitem__(self, key):
        return self.store[self.__keytransform__(key)]

    def __setitem__(self, key, value):
        self.store[self.__keytransform__(key)] = value

    def __delitem__(self, key):
        del self.store[self.__keytransform__(key)]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __keytransform__(self, key):
        return key

    def has_key(self, key):
        return key in self.store

    # custom methods
    def setval(self, k, v):
        self.store[k] = v

    def __str__(self):
        storekeys = list(self.store.keys())
        storekeys.sort()
        ret = ''
        for k in storekeys:
            v = self.store[k]
            ret += "k = %s, v.shape = %s, mu = %s, var = %s\n" % (k, v.shape, np.mean(v, axis = -1), np.var(v, axis = -1))
        return ret

    def astable(self, loop_compress = False):
        """Return a table based bus representation for debug, print, and display.

        Arguments:
         - loop_compress(bool): try to collapse each set of loop signatures into single bus
        """
        ret = ''

        if not loop_compress:
            storekeys = list(self.store.keys())
            storekeys.sort()
            # print "storekeys", storekeys
            for k in storekeys:
                v = self.store[k]
                ret += "{0:<20} | {1:<20}\n".format(k, v.shape)
            return ret

        loop_compressor = {}
        for k in list(self.store.keys()):
            # squash loop delimiters down to base node-id/signal
            k_ = compress_loop_id(k)
            # add if not there
            if k_ not in loop_compressor:
                # print "new key", k_, k
                loop_compressor[k_] = {
                    'v': self.store[k],
                    'cnt': 1,
                }
            else:
                loop_compressor[k_]['cnt'] += 1

        storekeys = list(loop_compressor.keys())
        # print "storekeys", storekeys
        storekeys.sort()
        # print "storekeys", storekeys

        # for k, v in loop_compressor.items():
        for k in storekeys:
            v = loop_compressor[k]
            # ret += "k = %s, v.shape = %s, mu = %s, var = %s\n" % (k, v.shape, np.mean(v, axis = -1), np.var(v, axis = -1))
            # print('    {0:<20}[{2}] | {1:<20}'.format(k, str(v['v'].shape), v['cnt']))
            ret += "{0:<20}[{2}] | {1:<20}\n".format(k, str(v['v'].shape), v['cnt'])
        return ret

    def keys_loop_compress(self):
        """Bus plotting helper function compressing all keys with loop index into base name
        """
        loop_compressor = {}
        for k in list(self.store.keys()):
            # squash loop delimiters down to base node-id/signal
            k_ = compress_loop_id(k)
            # add if not there
            if k_ not in loop_compressor:
                # print "new key", k_, k
                loop_compressor[k_] = {
                    'v': self.store[k],
                    'cnt': 1,
                }
            else:
                loop_compressor[k_]['cnt'] += 1
        return loop_compressor
        
    def store_pickle(self, conf = {}):
        """Bus.store_pickle

        Store bus as pickle for bus plotting from cache
        """
        bus_filetype = 'pickle' # 'gml' # , 'json', 'yaml'
        bus_filename = '%s_%s.%s' % (conf['datafile_md5'], 'bus', bus_filetype)
        tmp_ = {}
        for k, v in list(self.store.items()):
            # print "Bus k = %s, v = %s" % (k, v)
            tmp_[k] = v.shape
        pickle.dump(tmp_, open(bus_filename, 'wb'))

    @staticmethod
    def load_pickle(conf = {}):
        """Bus.load_pickle

        Load Bus from pickle for bus plotting from cache
        """
        bus_filetype = 'pickle' # 'gml' # , 'json', 'yaml'
        bus_filename = '%s_%s.%s' % (conf['datafile_md5'], 'bus', bus_filetype)
        b = Bus()
        tmp_ = pickle.load(open(bus_filename, 'rb'))
        for k, v in list(tmp_.items()):
            b[k] = np.zeros(v)
        return b
        
    def plot(self, ax = None, blockid = None):
        """Bus.plot

        Plot the bus for documentation and debugging.
        """
        assert ax is not None
        xspacing = 20
        yspacing = 10 # 3
        yscaling = 1.2 # 66
        xscaling = 0.66
        
        xmax = 0
        ymax = 0

        xpos = 0 # xspacing
        ypos = -6 # yspacing

        if blockid is None: blockid = "Block2"
            
        # ax.set_title(blockid + ".bus", fontsize = 10)
        ax.set_title('topblock.bus', fontsize = 10)
        ax.grid(0)

        # data coords / axis coords
        
        # ax.text(
        #     xpos, ypos, "Bus (%s)" % ("topblock"),
        #     fontsize = 10,
        #     bbox = dict(facecolor = 'red', alpha = 0.5, fill = False,))
        # # ax.text(0, 1.0, "Bus (%s)" % ("topblock"), transform = ax.transAxes, fontsize = 10)
        # ypos -= max(yspacing, 0)
        
        # ax.plot(np.random.uniform(-5, 5, 100), "ko", alpha = 0.1)
        i = 0
        bs_width_max = 0
        storekeys = list(self.store.keys())
        storekeys.sort()
        # for k, v in self.store.items():
        for k in storekeys:
            v = self.store[k]
            # print "k = %s, v = %s" % (k, v)
            
            # if len(k) > 8:
            #     xspacing = len(k) + 2
            # else:
            #     xspacing = 10
                
            ax.text(
                xpos, ypos, "{0: <8}\n{1: <12}".format(k, v.shape),
                family = 'monospace', fontsize = 8)
            # ax.text(xpos, ypos, "{0: <8}\n{1: <12}".format(k, v.shape), family = 'monospace', fontsize = 8, transform = ax.transAxes)

            # buf shapes: horizontal
            # elementary shape without buffersize
            ax.add_patch(
                patches.Rectangle(
                    (xpos+4, ypos + (0.4 * yspacing)),   # (x,y)
                    1.0 * xscaling,          # width
                    v.shape[0] * -yscaling,          # height
                    fill = False,
                    hatch = "|",
                    # hatch = "-",
                )
            )
            
            # full blockshape
            bs_width = -np.log10(v.shape[1] * xscaling)
            bs_height = -np.log10(v.shape[0] * yscaling)
            ax.add_patch(
                patches.Rectangle(
                    # (30, ypos - (v.shape[0]/2.0) - (-yspacing / 3.0)),   # (x,y)
                    (xpos+6, ypos + (0.4 * yspacing)),        # pos (x,y)
                    bs_width,              # width
                    v.shape[0] * -yscaling, # height
                    fill = False,
                    hatch = "|",
                    # hatch = "-",
                )
            )

            # # buf shapes: vertical
            # # elementary shape without buffersize
            # ax.add_patch(
            #     patches.Rectangle(
            #         # (30, ypos - (v.shape[0]/2.0) - (-yspacing / 3.0)),   # (x,y)
            #         (xpos+2, ypos-1),   # (x,y)
            #         v.shape[0],          # width
            #         -1 * yscaling,          # height
            #         fill = False,
            #         # hatch = "|",
            #         hatch = "-",
            #     )
            # )
            
            # # full blockshape
            # bs_height = -np.log10(v.shape[1] * yscaling)
            # ax.add_patch(
            #     patches.Rectangle(
            #         # (30, ypos - (v.shape[0]/2.0) - (-yspacing / 3.0)),   # (x,y)
            #         (xpos+2, ypos-2),   # (x,y)
            #         v.shape[0],          # width
            #         bs_height,          # height
            #         fill = False,
            #         # hatch = "|",
            #         hatch = "-",
            #     )
            # )

            # ypos = -10 # -(i+1)*yspacing
            # xpos = (i+1)*xspacing
            ypos -= max(yspacing, v.shape[0] + 1)
            # xpos +=  max(xspacing, len(k) + 2)
            if bs_width > bs_width_max: bs_width_max = bs_width
            
            if xpos > xmax: xmax = xpos
            if -(ypos - 8 - bs_height) > ymax: ymax = -(ypos - 8 - bs_height) + 2
            i+=1
            
        # ax.set_xlim((0, 100))
        # ax.set_ylim((-100, 0))
        ax.set_xlim((-1, max(xmax, xpos + 6 + bs_width_max + 1)))
        ax.set_ylim((-ymax, 4))

        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        
        plt.draw()
        plt.pause(1e-6)

        return (xmax, ymax)

def get_blocksize_input(G, buskey):
    """Get the blocksize of input element from the bus at 'buskey'

    .. warning:: Move this to :class:`Bus`?
    """
    # print "G", G.nodes(), "buskey", buskey
    assert '/' in buskey, "Malformed buskey %s without '/'" % (buskey, )
    (srcid, srcvar) = buskey.split("/") # [-1]
    # print "G.nodes()", G.nodes()
    n = nxgraph_node_by_id(G, srcid)
    
    # search graph and all subgraphs, greedy + depth first
    n_ = nxgraph_node_by_id_recursive(G, srcid)
    if len(n_) == 0:
        # didn't find anything, return a default (and possibly wrong) blocksize of 1
        return None
    # returned node, node's (sub)graph where it was found
    (n_0, g_0) = n_[0]
    # print "n_0", srcid, n_0, g_0.node[n_0]['block_'].blocksize, g_0.node[n_0]['block_'].id
    # return G.node[n_[0]]['block_'].blocksize
    return g_0.node[n_0]['block_'].blocksize
    
                
################################################################################
# Block decorator init
class decInit():
    """decInit Block2 init decorator class

    Wrap around Block2.__init__ to perform pervasive tasks
    """
    def __call__(self, f):
        def wrap(xself, *args, **kwargs):

            # FIXME: default propagation goes here probably
            
            # # print "decInit", kwargs['conf']
            # if not kwargs['conf'].has_key('inputs'):
            #     kwargs['conf']['inputs'] = {}
            # if not kwargs['conf']['inputs'].has_key('blk_mode'):
            #     kwargs['conf']['inputs']['blk_mode'] = 1.0 # enabled
                
            f(xself, *args, **kwargs)

            if xself.debug_trace_callgraph:
                from pycallgraph import PyCallGraph
                from pycallgraph import Config
                from pycallgraph.output import GraphvizOutput
                
                config = Config()
                graphviz = GraphvizOutput()
                
                graphviz.output_file = '%s.png' % xself.id
                xself.pcg = PyCallGraph(config=config, output=graphviz)
            # print "decInit", xself.id, xself.inputs.keys()
        return wrap

################################################################################
# Block decorator step
class decStep():
    """decStep Block2 step decorator class

    Wrap around Block2.step to perform tasks common to all Block2's.

    .. warning:: Fix the docstring issue with wrapping using any one
       of :mod:`functools` :func:`wraps`, manually copying __doc__,
       __name__, __module__, or the autoclass/automethod approach.
    """
    def __call__(self, f):
        """decStep.__call__ is called when decorator is called

        It is only executed once during parse (?) and returns the function pointer to f_wrap(f_orig) composite
        """
        @wraps(f)
        def wrap(xself, *args, **kwargs):
            # return pointer to wrap_l0
            # print "__debug__", __debug__
            if xself.top.do_pdb:
                return self.wrap_l0_pdb(xself, f, args, kwargs)
            else:
                return self.wrap_l0(xself, f, args, kwargs)
            
        # # manual attribute copy
        # if f.__doc__ is not None:
        #     wrap.__doc__ = f.__doc__
        # if f.__name__ is not None:
        #     wrap.__name__ = f.__name__
        # if f.__module__ is not None:
        #     wrap.__module__ = f.__module__
            
        # return the composite function
        return wrap

    def wrap_l0(self, xself, f, *args, **kwargs):
        """Block2.step decorator (decStep) wrapper level 0

        Bypass of wrap_l0_pdb
        """
        if xself.debug_trace_callgraph:
            with xself.pcg:
                return self.wrap_l1(xself, f, args, kwargs)
        else:
            return self.wrap_l1(xself, f, args, kwargs)
    
    def wrap_l0_pdb(self, xself, f, *args, **kwargs):
        """Block2.step decorator (decStep) wrapper level 0 with pdb fallback

        Wrap the function 'f' inside a try/except and enter pdb shell on exception

        FIXME: make dump environment configurable: pdb, InteractiveConsole, ...
        FIXME: make try/except depend on __debug__, xself.top.debug, xself.debug
        """
        try:
            return self.wrap_l1(xself, f, args, kwargs)
        except Exception as e:
            pdb.set_trace()
            return None

    def wrap_l1(self, xself, f, *args, **kwargs):
        """Block2.step decorator (decStep) wrapper level 1

        The actual worker function consisting of
         1. process_input: copy bus to inputs, copy inputs to outputs with the same name
         2. check for blk_mode inputs (wip)
         3. evaluate the original function
         4. post-process outputs: copy outputs in xself.attr to bus[id/attr]
        """
        # count calls
        # xself.cnt += 1 # should be min_blocksize
        # if self.topblock:
        xself.cnt += xself.top.blocksize_min
        # print "%s    %s-%s[%d] step_wrapped" %(xself.nesting_indent, xself.cname, xself.top.blocksize_min, xself.cnt)
        
        self.process_input(xself)

        if not self.get_credit(xself) or self.process_blk_mode(xself): return None
        
        f_out = self.f_eval(xself, f)
            
        self.process_output(xself)
                
        return f_out

    def process_blk_mode(self, xself):
        if hasattr(xself, 'inputs') and 'blk_mode' in xself.inputs:
            # print "blk_mode", xself.id, np.sum(xself.inputs['blk_mode']['val']) # xself.inputs['blk_mode']['val'], xself.inputs['blk_mode']['val'] < 0.1
            # if the mode input is zero, skip processing
            if np.sum(xself.inputs['blk_mode']['val']) < 0.1:
                # xself.cnt += 1
                return True
        return False

    def get_credit(self, xself):
        if 'credit' in xself.inputs:
            return np.all(xself.inputs['credit']['val'] > 0.0)
        else:
            return True
            
    def process_input(self, xself):
        sname  = self.__class__.__name__
        esname = xself.cname
        esid   = xself.id
        escnt  = xself.cnt
        
        # loop over block's inputs
        for k, v in list(xself.inputs.items()):
            # print "process_input: ", k, xself.id, xself.cnt, v['val'].shape, v['shape']
            # check input sanity
            assert v['val'].shape == v['shape'], "%s-%s's real and desired input shapes need to agree but ink = %s, %s != %s" % (
                xself.cname, xself.id, k, v['val'].shape, v['shape'])
            
            # copy bus inputs to input buffer
            if 'bus' in v: # input item is driven by external signal (bus value)
                # exec   blocksize of the input's source node
                # FIXME: search once and store, recursively over nxgraph and subgraphs
                if 'blocksize' not in v:
                    v['blocksize'] = get_blocksize_input(xself.top.nxgraph, v['bus'])
                blocksize_input = v['blocksize']
                # output blocksize of the input's source node
                blocksize_input_bus = xself.bus[v['bus']].shape[-1]
                    
                # if search didn't find the node (for whatever reason),
                # set a default from the input bus blocksize
                if blocksize_input is None:
                    blocksize_input = blocksize_input_bus


                if blocksize_input_bus == v['shape'][-1]: # for all vshape in 1,2,3..
                    # FIXME: f_process_inbus_equal_inbuf, boundary?
                    if xself.cnt % xself.blocksize == 0:
                        v['val'] = xself.bus[v['bus']].copy()
                        # print "v['val']", v['val'].shape
                    
                elif blocksize_input_bus > v['shape'][-1]: # for all vshape in 1,2,3..
                    # FIXME: f_process_inbus_gt_inbuf, boundary?
                    # if xself.cnt % v['shape'][-1] == 0:
                    if xself.cnt % xself.blocksize == 0:
                        mcnt = xself.cnt % blocksize_input_bus
                        # sl = slice(mcnt - v['shape'][-1] + 1, mcnt + 1) #
                        sls = (xself.cnt - v['shape'][-1]) % blocksize_input_bus
                        sle = sls + v['shape'][-1] # xself.cnt % blocksize_input_bus
                        sl = slice(sls, sle) #
                        v['val'] = xself.bus[v['bus']][...,sl].copy()
                        # print "bus > v", xself.id, xself.cnt, blocksize_input_bus, mcnt, k, "sl", sl, "v['val']", v['val'], "v['bus']", v['bus'], xself.bus[v['bus']].shape, xself.bus[v['bus']]
                    
                elif blocksize_input_bus < v['shape'][-1]:
                    # FIXME: f_process_inbus_lt_inbuf
                    # if extended input buffer and new data,
                    # rotate input buffer by blocksize_input_bus
                    # if v['shape'][-1] > 1 and (xself.cnt % blocksize_input) == 0:
                    if (xself.cnt % blocksize_input) == 0:
                        # print "%s-%s[%d] decStep copy inputs bs_in, bs_in_bus" % (xself.cname, xself.id, xself.cnt), blocksize_input, blocksize_input_bus
                        # shift by input blocksize along self.blocksize axis
                        # axis = len(xself.bus[v['bus']].shape) - 1
                        axis = len(v['shape']) - 1
                        # print v['val'][...,-blocksize_input_bus:]
                        v['val'] = np.roll(v['val'], shift = -blocksize_input_bus, axis = axis)
                        # print "%s.decStep v[val]" % (xself.cname), v['val'].shape, "v.sh", v['shape'], "axis", axis, "v", v['val']
                        # set inputs [last-inputbs:last] if input blocksize reached                                
                        sl = slice(-blocksize_input_bus, None)
                        # print "%s-%s" % (xself.cname, xself.id), "sl", sl, "bus", v['bus'], "bus.shape", xself.bus[v['bus']].shape, "v['val'].shape", v['val'].shape
                        try:
                            # np.fliplr(xself.bus[v[2]])
                            # v['val'][...,-blocksize_input_bus:] = self.bus[v['bus']].copy()
                            v['val'][...,-blocksize_input_bus:] = xself.bus[v['bus']].copy()
                        except Exception as e:
                            print("%s-%s[%d].decStep input copy k = %s from bus %s/%s to %s/%s, %s" % (xself.cname, xself.id, xself.cnt, k, v['bus'], xself.bus[v['bus']].shape, v['shape'], v['val'].shape, e))
                            sys.exit(1)
                            
                        # if k == 'd2':
                        # if xself.id == "plot_infth":
                        #     # debugging bus to in copy
                        #     print "%s-%s.%s[%d]\n  bus[%s] = %s to %s[%s] = %s / %s" % (esname, esid,
                        #                                          sname,
                        #                                          escnt,
                        #                                          v['bus'],
                        #                                          xself.bus[v['bus']].shape, k, sl, v['shape'], v['val'].shape)
                        #     print "blocksize_input", blocksize_input, "blocksize_input_bus", blocksize_input_bus
                        
                        #     # print xself.bus[v['bus']]
                        #     print v['val'][...,-1]
                    
            # copy input to output if inkey k is in outkeys
            if k in list(xself.outputs.keys()):
                setattr(xself, k, v['val'].copy()) # to copy or not to copy
                
                # # debug in to out copy
                # print "%s.%s[%d]  self.%s = %s" % (esname, sname, escnt, k, esk)
                # print "%s.%s[%d] outkeys = %s" % (esname, sname, escnt, xself.outputs.keys())
    def process_output(self, xself):
        """post-process xself's outputs by copying output attributes to the bus
        """
        # if scheduled by exec timing
        if xself.block_is_scheduled():
            # for all output items
            for k, v in list(xself.outputs.items()):
                # defer non-ndarray types
                # if v.has_key('type') and v['type'] != 'ndarray':
                if xself.output_is_type(k, v):
                    # print "wrong type fail", k, v
                    continue
                if not xself.output_is_triggered(k, v, xself.bus):
                    # xself._debug("not triggered")
                    continue
                # else:
                #     print "decstep process_output", k, v

                # get buskey and sanity check
                buskey = "%s/%s" % (xself.id, k)
                # if __debug__:
                #     print "copy[%d] %s.outputs[%s] = %s / %s to bus[%s], bs = %d" % (
                #         xself.cnt, xself.id, k, getattr(xself, k), getattr(xself, k).shape,
                #         buskey, xself.blocksize)
                
                assert xself.bus[v['buskey']].shape == v['shape'], "%s-%s's real and desired output shapes need to agree, but self.%s = %s != %s output['shape']" % (
                    xself.cname, xself.id, k, xself.bus[v['buskey']].shape, v['shape'])

                # FIXME: for all items in output channels = [bus, log, ros, pdf, latex, ...]
                
                # copy data onto bus
                # xself._info('xself.%s = %s, %s' % (k, type(getattr(xself, k)), getattr(xself, k)))
                xself.bus[v['buskey']] = getattr(xself, k).copy()
                
                # if v['buskey'] == 'robot1/h':
                #     logger.debug("output '%s', xself.bus[v['buskey'] = %s] = %s / %s" % (k, v['buskey'], xself.bus[v['buskey']], getattr(xself, k)))
                
                # if v.has_key('trigger'):
                #     xself._debug("    triggered: bus[%s] = %s, buskeys = %s" % (buskey, xself.bus[v['buskey']], xself.bus.keys()))
                
                # copy data into logging
                if xself.logging:
                    # if xself.id == 'b4':
                    #     print "logging %s-%s to tbl_name = %s, data = %s" % (xself.cname, xself.id, v['buskey'], xself.bus[v['buskey']])
                    # try:
                    # if xself.cname == 'SeqLoopBlock2':
                    #     print "logging", v['buskey'], xself.bus[v['buskey']]
                    log.log_pd(tbl_name = v['buskey'], data = xself.bus[v['buskey']])
                    # except:
                    # print "Logging failed"

                # copy data onto ros
                if hasattr(xself, 'ros') and xself.ros:
                    # assert rospy.core.is_initialized(), "%s-%s is configured self.ros = %s but roscore is not initialized."
                    theattr = getattr(xself, k).flatten().tolist()
                    # print "theattr", k, v, theattr
                    xself.msgs[k].data = theattr
                    xself.pubs[k].publish(xself.msgs[k])
        
        # clean up block activity
        if xself.block_is_finished():

            # define update block_store
            def update_block_store(block = None, top = None):
                import datetime
                assert block is not None, "Need some block to work on"
                # if not block.isprimitive: return

                # m = md5(str(block.conf))
                print("update_block_store block = %s (%s)" % (block, block.md5))

                print("block_store", list(block.top.block_store.keys()))

                # if hasattr(xself, 'nxgraph'):
                #     xself.cache['nxgraph'] = nxgraph_to_smp_graph(block.nxgraph, asdict = True) # str()
                #     print "block_store['/blocks'] nxgraph", xself.cache['nxgraph']
                    
                # create entry and save
                # columns = ['md5', 'timestamp', 'block', 'params', 'log_store'] # 'experiment', 
                # values = [[block.md5, pd.to_datetime(datetime.datetime.now()), str(block.conf['block']), str(block.conf['params']), 'bla.h5']]
                # df = pd.DataFrame(data = values, columns = columns)
                # print "df", df

                # block store is empty
                # if len(block.top.block_store.keys()) < 1:
                #     xself.top.block_store['blocks'] = df
                # else:
                # xself.top.block_store['blocks'] = pd.concat([xself.top.block_store['blocks'], df])
                # xself.top.block_store['blocks']['md5' == self.md5][] = 

            # perform update if configured
            if xself.top.docache and xself.cache is not None and not xself.cache_loaded:
                if hasattr(xself, 'isprimitive') and xself.isprimitive:
                    pass
                else:
                    update_block_store(block = xself, top = xself.top)

            # dump input / output buffers at episode end

    def f_eval(self, xself, f):
        # call the function on blocksize boundaries
        # FIXME: might not be the best idea to control that on the wrapper level as some
        #        blocks might need to be called every step nonetheless?
        # if (xself.cnt % xself.blocksize) == 0: # or (xself.cnt % xself.rate) == 0:
        # print "decStep.f_eval", xself.id, "xself.cnt", xself.cnt, "blocksize", xself.blocksize, "blockphase", xself.blockphase
        # if (xself.cnt % xself.blocksize) in xself.blockphase: # or (xself.cnt % xself.rate) == 0:
        # if count aligns with block's execution blocksize
        if xself.block_is_scheduled():
            # are we caching?
            if not xself.topblock and xself.top.docache and xself.cache is not None and xself.cache_loaded: # cache_block and cache_inst:
                # pass
                shapes = []
                for outk, outv in list(xself.outputs.items()):
                    # only ndarray type outputs
                    if xself.output_is_type(outk, outv):
                        continue
                    # logger.debug('types {2}, {0} {1}'.format(xself.cnt, xself.blocksize, outk))
                    setattr(xself, outk, xself.cache_data[outk][xself.cnt-int(xself.blocksize):xself.cnt,...].T)
                    # print "cache out key", outk, getattr(xself, outk)
                    # if xself.cnt < 10:
                    #     print "%s-%s[%d]" % (xself.cname, xself.id, xself.cnt), "outk", outk, getattr(xself, outk).T
                    #     # print "    ", xself.cache_data[outk].shape
                    shapes.append(xself.cache_data[outk].shape)
                    # print "outk", outk, xself.cache_data[outk] # [xself.cnt-xself.blocksize:xself.cnt]
                if xself.cnt % 100 == 0:
                    print("decStep.f_eval\n    %s-%s cache hit at %s\n        using cached data with shapes %s\n" % (xself.cname, xself.id, xself.md5, shapes))
                f_out = None
                
            # not caching
            else:
                # compute the block with step()
                # print "%s    %s-%s[%d] f_out 0" % (xself.nesting_indent, xself.cname, xself.id, xself.cnt)
                f_out = f(xself, None)
                # print "%s    %s-%s[%d] f_out 1" % (xself.nesting_indent, xself.cname, xself.id, xself.cnt)
                
        else:
            f_out = None

        return f_out

################################################################################
# Base block dummy class for testing
class DummyBlock2(object):
    """Dummy top block

    Can be used for testing Block2s which require 'top' argument and
    rely on some top properties for internal operation.
    """
    def __init__(self):
        from smp_graphs.block import Bus
        
        self.blocksize_min = np.inf
        self.bus = Bus()
        self.cnt = 0
        self.docache = False
        self.inputs = {}
        self.numsteps = 100
        self.saveplot = False
        self.topblock = True
    
################################################################################
# Base block class
class Block2(object):
    """Block2 class

    Base class for node blocks. smp_graphs nodes have a 'block' attribute which implements the node's computational function.

    Arguments:
     - conf(dict): Block configuration dictionary
     - paren(Block2): ref to Block's parent
     - top(Block2): ref to top-level node in nested graphs
     - blockid(str): override block's id assignment

    See also: :mod:`smp_graphs.block`
    """
    
    defaults = {
        'id': None,
        'debug': False,
        'debug_trace_callgraph': False, # special debug option: build and save block callgraph
        'topblock': False,
        'ibuf': 1, # input  buffer size
        'obuf': 1, # output buffer size
        'cnt': 1, # 1, FIXME: log / cache issues: replaced init_cnt = 1 with init_cnt = 0 and topblock.step() once after graph init?
        'blocksize': 1, # period of computation calls in time steps
        'blockphase': [0], # list of positions of comp calls along the counter in time steps
        'inputs': {}, # internal port, scalar / vector/ bus key, [slice]
        'outputs': {}, # name, dim
        'loglevel': loglevel_DEFAULT,
        'logging': True, # normal logging
        'rate': 1, # execution rate rel. to cnt
        'ros': False, # no ROS yet
        'phase': [0],
        'subgraph_rewrite_id': True, #
        'inputs_clamp': False,
        'block_group': 'graph',
    }

    @decInit()
    def __init__(self, conf = {}, paren = None, top = None, blockid = None, conf_vars = None):
        ################################################################################
        # general stuff
        self.conf = conf
        self.paren = paren
        self.top = top
        self.cname = self.__class__.__name__
        self.conf_vars = conf_vars

        # merge Block2 base defaults with child defaults
        defaults = {}
        defaults.update(Block2.defaults, **self.defaults)
        
        # load defaults
        # set_attr_from_dict(self, self.defaults)
        set_attr_from_dict(self, copy.copy(defaults))
                    
        # fetch existing configuration arguments
        if type(self.conf) == dict and 'params' in self.conf:
            # print "Block2 init params", self.conf['params']
            params = copy.deepcopy(self.conf['params'])
            set_attr_from_dict(self, params)
        else:
            print("What could it be? Look at %s" % (self.conf))

        # FIXME: no changes to conf['params'] after this?
            
        # check id
        assert hasattr(self, 'id'), "Block2 init: id needs to be configured"
        # FIXME: check unique id, self.id not in self.topblock.ids
        # print "%s-%s.defaults = %s" % (self.cname, self.id, defaults)

        # get the nesting level in composite graph
        self.nesting_level = self.get_nesting_level()
        self.nesting_indent = " " * 4 * self.nesting_level

        # fix the block's group
        # print "Block2 %s self.block_group" % (self.id,), self.block_group
        if type(self.block_group) is str: self.block_group = [self.block_group]

        # blocksize = 1 special cnt init
        # if self.blocksize != 1:
        self.cnt = 0
        # self.cnt = self.blocksize

        # configure logger
        if self.debug: self.loglevel = logging_DEBUG
            
        id_ = self.id
        if self.topblock:
            id_ = 'top'
        if len(id_) > 20:
            id_ = self.id[:20]
            
        self.cname_full = '%s-%s' % (self.cname, id_)
        self.modulename = self.__class__.__module__
        # self.logger = get_module_logger(modulename = '{0: >20}.{1: <20}'.format(self.modulename, self.cname_full), loglevel = self.loglevel)
        self.logger = get_module_logger(modulename = '{0}.{1}'.format(self.modulename, self.cname_full), loglevel = self.loglevel)
        # self.logger = get_module_logger(modulename = modulename_, loglevel = self.loglevel)
        # print "cname_full = %s, modulename = %s, logger.name = %s" % (self.cname_full, self.modulename, self.logger.name)
            
        ################################################################################
        # 1 topblock: init bus, set top to self, init logging
        #   all the rest should be the same as for file, dict, loop, or loopseq
        #   subgraphs
        if self.topblock:
            # print "Block2.init topblock conf.keys", self.conf['params'].keys()
            self._info('#' * 80)
            self._info("topblock init with numsteps = %s" %(self.numsteps, ))
            
            # fix the random seed
            # np.random.seed(self.randseed)

            # use debugger?
            # self.do_pdb
            
            self.top = self
            self.bus = Bus()
            # self.lsh_colors = lshash.LSHash(hash_size = 3, input_dim = 1000)

            # block_store init, topblock only
            def init_block_store():
                block_store_filename = 'data/block_store.h5'
                return pd.HDFStore(block_store_filename)

            self.block_store = init_block_store()
            self.log_store_cache = None

            # initialize pandas based hdf5 logging
            log.log_pd_init(self.conf)

            # write initial configuration to dummy table attribute in hdf5
            log.log_pd_store_config_initial(print_dict(self.conf))

            # # debug out
            # print "Block2 topblock init self.conf['params'] = {"
            # pkeys = self.conf['params'].keys()
            # pkeys.sort()
            # for pk in pkeys:
            #     pv = self.conf['params'][pk]
            #     if type(pv) is OrderedDict:
            #         pv = pv.keys()
            #     print "    k = %s, v = %s" % (pk, pv,)
            # print "}"

            # print "Block2 topblock outputs", self.outputs
            # # self.outputs
            
            # # latex output
            # self.latex_conf = {
            #     'figures': {},
            #     }
            
            # initialize the graph
            self.init_block()

            # print "init top graph", print_dict(self.top.graph)
            
            # dump the execution graph configuration to a file
            finalconf = self.dump_final_config()
            
            # # FIXME: this needs more work
            # log.log_pd_store_config_final(finalconf)
            # nxgraph_ = copy.copy(self.nxgraph)
            # del nxgraph_['block_']
            # nx.write_yaml(nxgraph_to_smp_graph(self.nxgraph), 'nxgraph.yaml')
            
            # # this always fails
            # try:
            #     nx.write_gpickle(self.nxgraph, 'nxgraph.pkl')
            # except Exception, e:
            #     print "%s-%s init pickling graph failed on downstream objects, e = %s" % (self.cname, self.id, e)
            #     # print "Trying nxgraph_dump"

            # store final static graph
            log.log_pd_store_config_final(nxgraph_to_smp_graph(self.nxgraph))

            # step the thing once
            # self.step()
            # print "Block2.init topblock self.blocksize", self.blocksize

            # # topblock init block cache
            # self.init_block_cache()
            
        # not topblock
        else:
            # check numsteps commandline arg vs. config blocksizes
            self.blocksize_clamp()

            # get top level configuration variables
            self.set_attr_from_top_conf()
            
            # get bus from topblock
            self.bus = self.top.bus

            # init block
            self.init_block()

        # print "Block2-%s.super   conf = %s" % (self.id, print_dict(self.conf))
        # print "Block2-%s.super inputs = %s" % (self.id, print_dict(self.inputs))
            
        # numsteps / blocksize
        # print "%s-%s end of init blocksize = %d" % (self.cname, self.id, self.blocksize)

    def init_block(self):
        """Block2.init_block

        Init a graph based block: topblock, hierarchical inclusion from file or dictionary, loop, loop_seq
        """

        # init block color
        self.init_colors()


        ################################################################################
        # 2 copy the config dict to exec graph if hierarchical
        if self.block_is_composite():
            """This is a composite block made up of other blocks via one of
            several mechanisms:
             - graph: is a graph configuration dict
             - subgraph: is path of configuration file
             - loopblock: loopblocks build subgraphs dynamically
             - cloneblock: we are cloning another subgraph referenced by existing
               nodeid
            """
            self._debug('init_block composite')
            # print "%s   with attrs = %s" % (self.nesting_indent, self.__dict__.keys())
            # for k,v in self.__dict__.items():
            #     print "%s-%s k = %s, v = %s" % (self.cname, self.id, k, v)

            # minimum blocksize of composite block
            self.blocksize_min = np.inf
            
            # subgraph preprocess, propagate additional subgraph configuration
            if hasattr(self, 'subgraph'):
                # print "%s%s-%s.init_block composite init_subgraph" % (self.nesting_indent, self.cname, self.id)
                self.init_subgraph()
                # set_attr_from_dict(self, self.conf['params'])

            # if self.conf['params'].has_key('outputs'):
            #     print "Block2-%s.init_block post-init_subgraph self.conf = %s" % (self.id, self.conf['params']['outputs'])
            #     print "Block2-%s.init_block post-init_subgraph self.conf = %s" % (self.id, self.outputs)
                
            # get the graph from the configuration dictionary
            # compute color from with lsh?
            # a = ''.join([chr(np.random.randint(32, 120)) for i in range(np.random.randint(10, 200))])

            # linelen = 1000
            # b = np.fromstring(str(self.conf), dtype=np.uint8)
            # bmod = b.shape[0] % linelen
            # print "b", b.shape, "mod", bmod
            # # if bmod != 0:
            # b = np.pad(b, (0, linelen - bmod), mode = 'constant')
            # print "b", b.shape
            # c = b.reshape((-1, linelen)).mean(axis = 0)
            
            # print "c", c.shape
            # # d = self.lsh_colors.index(c)
            # # d_ = self.lsh_colors.query(c)
            # print "d_", d_[0][0]
            # # np.pad(b, (0, max(0, (b.shape[0]/10) * 11)), mode = 'constant').shape

            self.nxgraph = nxgraph_from_smp_graph(self.conf)

            # if isinstance(self, SeqLoopBlock2):
            #     print "Block2.init_block   nxgraph", self.nxgraph.name, self.nxgraph.number_of_nodes()
            #     print "Block2.init_block self.conf", print_dict(self.conf)

            # experimental: node cloning
            self.node_cloning()
            
            # for n in self.nxgraph.nodes():
            #     print "%s-%s g.node[%s] = %s" % (self.cname, self.id, n, self.nxgraph.node[n])
        
            # 2.1 init_pass_1: instantiate blocks and init outputs, descending into hierarchy
            self.init_graph_pass_1()
                                        
            # if self.conf['params'].has_key('outputs'):
            #     print "Block2-%s.init_block post igp1 self.conf = %s" % (self.id, self.outputs, )
                
            # 2.2 init_pass_2: init inputs, again descending into hierarchy
            self.init_graph_pass_2()
            
            # if self.conf['params'].has_key('outputs'):
            #     print "Block2-%s.init_block igp2 self.conf = %s" % (self.id, self.outputs, )
                
            self.init_outputs()
            
            # initialize block logging
            self.init_logging()

            # if self.conf['params'].has_key('outputs'):
            #     print "Block2-%s.init_block init_outputs self.conf = %s" % (self.id, self.outputs, )
            
            if self.blocksize_min < self.top.blocksize_min:
                self.top.blocksize_min = self.blocksize_min
                
        ################################################################################
        # 3 initialize a primitive block
        else:
            self.init_primitive()

        # minimum blocksize of composite block
        # if not self.topblock and self.blocksize < self.top.blocksize_min:
        if self.blocksize < self.top.blocksize_min:
            self.top.blocksize_min = self.blocksize
            
        # initialize block caching, FIXME: before logging?
        self.init_block_cache()

    def node_cloning(self):
        if True:
            ##############################################################################
            # node cloning (experimental)
            if hasattr(self, 'graph') \
              and type(self.graph) is str \
              and self.graph.startswith('id:'):
                # search node
                print("top graph", self.top.nxgraph.nodes())
                targetid = self.graph[3:] # id template
                targetnode = nxgraph_node_by_id_recursive(self.top.nxgraph, targetid)
                print("%s-%s" % (self.cname, self.id), "targetid", targetid, "targetnode", targetnode)
                if len(targetnode) > 0:
                    print("    targetnode id = %d, node = %s" % (
                        targetnode[0][0],
                        targetnode[0][1].node[targetnode[0][0]]))
                # copy node
                clone = {}
                tnode = targetnode[0][1].node[targetnode[0][0]]
                for k in list(tnode.keys()):
                    print("cloning: subcloning: k = %s, v = %s" % (k, tnode[k]))
                    clone[k]  = copy.copy(tnode[k])
                    if k == 'block_':
                        clone[k].inputs  = copy.deepcopy(tnode[k].inputs)
                        clone[k].outputs = copy.deepcopy(tnode[k].outputs)
                    # clone[k]  = copy.deepcopy(tnode[k])
                    
                # clone = copy.deepcopy(targetnode[0][1].node[targetnode[0][0]])
                # clone = copy.copy(targetnode[0][1].node[targetnode[0][0]])
                """
                # block configuration reference

                targetnode_ = {
                    'block_': "<smp_graphs.block_models.ModelBlock2 object at 0x7f586c3e2710>",
                    'params': {
                        'inputs': {
                            'pre_l0': {
                                'bus': 'pre_l0/pre',
                                'shape': (1, 1),
                                'val': np.array([[ 0.]])},
                            'pre_l1': {
                                'bus': 'pre_l1/pre_l1',
                                'shape': (1, 1),
                                'val': np.array([[ 0.]])},
                            'meas_l0': {
                                'bus': 'robot1/s_proprio',
                                'shape': (1, 1),
                                'val': np.array([[ 0.]])}},
                        'models': {
                            'fwd': {
                                'inst_': "<smp_graphs.block_models.model object at 0x7f586c3e2b10>",
                                'odim': 1,
                                'type': 'actinf_m1',
                                'algo': 'knn',
                                'idim': 2}},
                        'blocksize': 1,
                        'rate': 1,
                        'blockphase': [0],
                        'outputs': {
                            'pre': {
                                'buskey': 'pre_l0/pre',
                                'shape': (1, 1),
                                'logging': True,
                                'init': True},
                            'tgt': {
                                'buskey': 'pre_l0/tgt',
                                'shape': (1, 1),
                                'logging': True,
                                'init': True},
                            'err': {
                                'buskey':
                                'pre_l0/err',
                                'shape': (1, 1),
                                'logging': True,
                                'init': True}},
                        'id': 'pre_l0'},
                    'block': "<class 'smp_graphs.block_models.ModelBlock2'>"}
                """
                        
                # replace id refs
                id_orig = copy.copy(clone['params']['id'])
                clone['params']['id'] = id_orig + "_clone"
                clone['block_'].id    = id_orig + "_clone"
                # replace input refs
                for k, v in list(clone['block_'].inputs.items()):
                    # if v['bus']
                    # v['bus'].split("/")[0]
                    # v['bus'].split("/")[0] + "_clone"
                    if hasattr(self, 'inputs'):
                        v = self.inputs[k]
                    else:
                        # replace all occurences of original id with clone id
                        v['bus'] = re.sub(id_orig, clone['params']['id'], v['bus'])
                    clone['block_'].inputs[k] = copy.deepcopy(v)
                    print("%s.init cloning  input k = %s, v = %s" % (self.cname, k, clone['block_'].inputs[k]))
                    
                # replace output refs
                for k, v in list(clone['block_'].outputs.items()):
                    # v['buskey'].split("/")[0], v['buskey'].split("/")[0] + "_clone"
                    v['buskey'] = re.sub(id_orig, clone['params']['id'], v['buskey'])
                    print("%s.init cloning output k = %s, v = %s" % (self.cname, k, v))
                    clone['block_'].outputs[k] = copy.deepcopy(v)
                print("cloning: cloned block_.id = %s" % (clone['block_'].id))
                
                # add the modified node
                self.nxgraph.add_node(0, **clone)

                # puh!

            # end node clone
            ##############################################################################
            
    def init_primitive(self):
        """Block2.init_primitive

        Initialize primitive block
        """
        # remember being primitive
        self.isprimitive = True
        
        # initialize block output
        self.init_outputs()
            
        # initialize block logging
        self.init_logging()
        
    def init_block_cache(self):
        """init_block_cache

        Block result caching. FIXME: unclear about exec spec / blocksize, step-wise or batch output
        """
        # initialize block cache
        def check_block_store(block = None, top = None):
            import datetime
            assert block is not None, "Need some block to work on"

            # failsafe top ref
            if top is None: top = block.top
            assert hasattr(top, 'block_store'), "Top block needs to have a pandas.HDF5Store type 'block_store' attribute"
                
            # strip naturally varying parts of the config
            conf_ = conf_strip_variables(block.conf)
            
            # append top_md5 to ensure the right context in terms of numsteps, randseed, ...
            conf_['top_md5'] = top.md5

            # if not 
            
            # # debug
            # print "%s   cache %s-%s top.md5 = %s" % (block.nesting_indent, block.cname, block.id, top.md5)
            # print "%s   cache %s-%s conf_.keys = %s" % (block.nesting_indent, block.cname, block.id, conf_.keys())
            # print "%s   cache %s-%s conf_ = %s" % (block.nesting_indent, block.cname, block.id, conf_)

            # compute hash of stripped conf 'conf_'
            m = md5(str(conf_))
            block.md5 = m.hexdigest()
            block.cache = None
            block.cache_loaded = False
            block.cache_num_entries = 0

            # return if we don't want caching here
            if not top.docache or hasattr(block, 'nocache') and block.nocache:
                return

            self._debug("block_store keys = %s" % (list(top.block_store.keys()), ))
            self._debug("block_store has blocks? is %s" % (hasattr(top.block_store, 'blocks'), ))
            # print "top.block_store['/blocks'] is top.block_store.blocks? is %s" % (top.block_store['/blocks'] is top.block_store.blocks, )
            
            # store exists
            # if hasattr(top.block_store, 'blocks'):
            if len(list(top.block_store.keys())) > 0:
                # blocks dataframe ref
                blocks = top.block_store.blocks
                # print "    blocks", blocks
                
                # print "init_block_cache: top.block_store.keys()", top.block_store.keys()
                # print "init_block_cache: top.block_store.has_key('/blocks')", '/blocks' in top.block_store.keys()
                # print "check_block_store", blocks.shape
                # print "init_block_cache: block.md5", block.cname, block.id, self.md5
                # print blocks['md5'] == block.md5

                # get number of cache entries
                block.cache_num_entries = blocks.index[-1]
                
                # try to load the cache entry from the store
                try:
                    # query = top.block_store['/blocks']['md5'] == block.md5
                    # block.cache = top.block_store['/blocks'][:][query]
                    query = blocks.md5 == block.md5
                    # print "    blocks query\n", query
                    block.cache = blocks[:][query]
                    if block.cache is not None and block.cache.shape[0] < 1:
                        block.cache = None
                except Exception as e:
                    print("%s%s-%s.check_block_store cache retrieval for %s failed with %s" % (
                        block.nesting_indent,
                        block.cname, block.id, block.md5, e))
                    block.cache = None
                    
                # print "check_block_store", block.md5, blocks['md5']
                # print "init_block_cache: block.cache", block.md5, block.cache, blocks.md5
                
            # cache loaded successfully
            if top.docache and block.cache is not None and block.cache.shape[0] != 0:
                print("%s%s-%s.check_block_store" % (block.nesting_indent, block.cname, block.id))
                print("%s    cache found for %s\n%s    cache['log_stores'] = %s" % (
                    block.nesting_indent, block.md5,
                    block.nesting_indent, block.cache['log_store'].values))

                if top.cache_clear:
                    logstr = "%s    cache_clear set, deleting cache entry %s / %s" % (block.nesting_indent, block.md5, blocks.shape)
                    self._debug(logstr)
                    
                    # hits = pd.Index(blocks.md5 == block.md5)
                    hits = blocks.md5 == block.md5
                    if hits is not None:
                        # for hit in hits.nonzero():
                        for hit in hits.index[hits]:
                            # print "%s    hit = %s/%s" % (block.nesting_indent, type(hit), hit)
                            # hit_ = hit[0] + 1
                            hit_ = hit
                            # print "hits", hits.nonzero()
                            # hits *= np.arange(0, len(hits) + 0)
                            # hits = pd.Index(hits)
                            # print "hits", hits
                            # print "hits deref", blocks[hits]
                            blocks_pre_drop_shape = blocks.shape
                            blocks = blocks.drop(index = [hit_])
                            print("%s    cache_clear set, dropping, pre = %s, post = %s" % (block.nesting_indent, blocks_pre_drop_shape, blocks.shape))
                    # # save
                    # blocks.to_hdf(top.block_store, key = 'blocks')
                    block.cache = None
                    
                else:
                    # load cached data only once
                    # FIXME: check experiment.cache to catch randseed and numsteps
                    if top.log_store_cache is None:
                        top.log_store_cache = pd.HDFStore(block.cache['log_store'].values[0])
                    
                    block.cache_data = {}
                    for outk, outv in list(block.outputs.items()):
                        if 'type' in outv and outv['type'] != 'ndarray': continue
                        # x_ = block.cache_h5['%s/%s' % (block.id, outk)].values
                        x_ = top.log_store_cache['%s/%s' % (block.id, outk)].values
                        print("%s%s-%s.check_block_store:     loading output %s cached data = %s" % (
                            block.nesting_indent,
                            block.cname, block.id, outk, x_))
                        # FIXME: check cache and runtime shape geometry
                        block.cache_data[outk] = x_.copy()
                        setattr(block, outk, block.cache_data[outk][...,[0]])

                    # # load the graph in case of composite block
                    # if str(block.cache['nxgraph']) != 'nxgraph':
                    #     nxgraph_new = nxgraph_from_smp_graph(str(block.cache['nxgraph']))
                    #     print "type(nxgraph_new)", str(nxgraph_new)
                    #     setattr(block, 'nxgraph', nxgraph_new)
                    #     print "type(nxgraph)", str(block.nxgraph)
                
                    # block.cache_h5.close()
                    block.cache_loaded = True
                
            # else:
            # no cache entry was found, cache was found but cleared, or loading failed for some other reason
            # cache loaded successfully
            if top.docache and block.cache is None:
                # print "%s%s-%s.check_block_store: no cache exists for %s, storing %s at %s" % (
                #     block.nesting_indent,
                #     block.cname, block.id, block.md5, 'block.conf', block.md5)
                
                # create entry and save it
                columns = [
                    'md5', 'timestamp', 'block', 'params', 'log_store', 'nxgraph'] # 'experiment',
                values = [[
                    block.md5, pd.to_datetime(datetime.datetime.now()),
                    str(block.conf['block']), str(block.conf['params']),
                    log.log_store.filename, str('nxgraph')]]
                index = block.cache_num_entries + 1
                df = pd.DataFrame(data = values, index = [index], columns = columns) # index = 
                # print "df =\n", df

                # block store is empty
                if len(list(top.block_store.keys())) < 1:
                    top.block_store['/blocks'] = df
                    blocks = top.block_store.blocks
                    print("%scache None found, no cache entries exist %s, %s" %(
                        block.nesting_indent, blocks.shape, df.shape))
                else:
                    print("%s    cache None found, cache entries exist %s" % (
                        block.nesting_indent, blocks.shape))
                    print("%s    cache_new blocks pre  concat = %s" % (block.nesting_indent, blocks.shape))
                    blocks = pd.concat([blocks, df])
                    print("%s    cache_new blocks post concat = %s" % (block.nesting_indent, blocks.shape))
                    # print "df2 concat(store, df)", df2.shape, df2
                    #  top.block_store['/blocks'] = df2
                    print("%s    cache_new inserted entry at index %s with md5 = %s" % (
                        block.nesting_indent, index, block.md5)) # , df.md5, df2.shape)

                # re-get cache
                block.cache = blocks[:][blocks.md5 == block.md5]
                # print "block.cache",block.cache
                
                # # save
                # blocks.to_hdf(top.block_store, key = 'blocks')

            # synchronize these two
            top.block_store['/blocks'] = blocks
            print("%s    cache block_store final shape = %s" % (
                block.nesting_indent, blocks.shape, ))

        # if self.top.docache:
        check_block_store(block = self)
        
    def init_subgraph(self):
        """Block2.init_subgraph

        Initialize a Block2's subgraph

        Subgraph is a filename of another full graph config as opposed
        to a graph which is specified directly as a dictionary.
        """

        # self._debug('lconf = %s' % (self.lconf))

        # subgraph is dict / OrderdedDict
        if type(self.subgraph) is OrderedDict:
            # print "subgraph is OrderedDict", self.subgraph
            subconfk = 'bla'
            subconf_ = OrderedDict([
                ('%s' % (subconfk, ), {
                    'block': Block2,
                    'params': {
                        'id': None,
                        'numsteps': self.numsteps,
                        'graph': self.subgraph, # [subconfk]
                    },
                }),
            ])
            subconf = subconf_[subconfk]
            
        # subgraph is a path pointing to an smp_graph config
        else:
            # print "subgraph is type = %s" % (type(self.subgraph))
            # for k,v in self.top.conf_vars.items():
            #     print "subgraph conf_vars[%s] = %s" % (k, v)

            # local configuration
            if hasattr(self, 'lconf'):
                # print("using lconf = %s" % (self.lconf, ))
                self._debug("using lconf = %s" % (self.lconf, ))
                subconf_vars = get_config_raw(conf = self.subgraph, confvar = None, lconf = self.lconf)
                # print "init_subgraph %s returning localvars = %s" % (self.id, subconf_vars.keys())
                subconf = subconf_vars['conf']
            # no local configuration
            else:
                # print "subgraph is single block, no lconf, using topblock config"
                lconf = {}
                # lconf.update(self.top.conf_vars)
                # for k in ['graph', 'IBlock2', 'Model']:
                #     del lconf[k]
                for k in ['numsteps', 'recurrent', 'cnf']:
                    lconf[k] = self.top.conf_vars[k]
                subconf_vars = get_config_raw(self.subgraph, confvar = None, lconf = lconf) # 'graph')
                subconf = subconf_vars['conf']
                
        assert subconf is not None, "Block2.init_subgraph subconf not initialized"
        assert type(subconf) is dict
        # print "type(subconf)", type(subconf)
        # print "subconf = %s" % (subconf, )
        # print "subconf['params']['graph'] = %s" % (subconf['params']['graph'], )
        # print "Block2.init_subgraph subconf keys = %s, subconf['params'].keys = %s" % (subconf.keys(), subconf['params'].keys(), )
        
        # make sure subordinate number of steps is less than top level numsteps
        assert subconf['params']['numsteps'] <= self.top.numsteps, "enclosed numsteps = %d greater than top level numsteps = %d" % (subconf['params']['numsteps'], self.top.numsteps)

        # set the graph params
        self.conf['params']['graph'] = copy.deepcopy(subconf['params']['graph'])
        # self.conf['params']['graph'] = subconf['params']['graph']

        # selectively ignore nodes in 'subgraph_ignore_nodes' list from included subgraph
        if hasattr(self, 'subgraph_ignore_nodes'):
            for ignk in self.subgraph_ignore_nodes:
                del self.conf['params']['graph'][ignk]
        
        # additional configuration for the subgraph
        if hasattr(self, 'subgraphconf'):
            # modifications happen in conf space since graph init pass 1 and 2 are pending
            for confk, confv in list(self.subgraphconf.items()):
                # split block id from config parameter
                (confk_id, confk_param) = confk.split("/")
                # get the block's node
                confnode = dict_search_recursive(self.conf['params']['graph'], confk_id)
                # return on fail
                if confnode is None: continue

                # debug info
                self._debug('subgraphconf node = %s' % (confnode['block'], ))
                self._debug('               param = %s' % (confk_param, ))
                self._debug('               val_bare = %s' % (confv, ))
                self._debug('               val_old = %s' % (confnode['params'][confk_param], ))
                # overwrite param dict
                # tmp = {}
                # tmp.update(confnode['params'][confk_param], **confv)
                if type(confv) is dict:
                    # fetch existing dict
                    tmp_ = copy.copy(confnode['params'][confk_param])
                    # update existing dict with subgraphconf replacement
                    tmp_.update(**confv)
                    # write dict back to reconfigured node's params
                    confnode['params'][confk_param].update(tmp_)
                    self._debug('               val_new = %s, tmp_ = %s' % (confnode['params'][confk_param], tmp_))
                else:
                    self._debug('               val_new = %s' % (confv, ))
                    confnode['params'][confk_param] = confv
                    
                # print "               val_new = %s, val_old = %s" % (confv, confnode['params'][confk_param])
                # debug print
                for paramk, paramv in list(confnode['params'].items()):
                    self._debug('    %s = %s' % (paramk, paramv))
        # # debug
        # print self.conf['params']['graph']['brain_learn_proprio']['params']['graph'][confk_id]

        # rewrite id strings?
        if hasattr(self, 'subgraph_rewrite_id') and self.subgraph_rewrite_id:
            # self.outputs_copy = copy.deepcopy(self.conf['params']['outputs'])
            nks_0 = dict_get_nodekeys_recursive(self.conf['params']['graph'])
            self._debug("    nodekeys = %s" % (nks_0, ))
            # xid = self.conf['params']['id'][-1:]
            # v['bus'] = re.sub(id_orig, clone['params']['id'], v['bus'])
            # p = re.compile('(.*)(_ll[0-9]+)$')
            
            # p = re.compile('(.*)((_ll[0-9]))+([_0-9])*')
            # p = re.compile('(.*)_((ll[0-9]))*([_0-9])*')
            # m = p.match(self.conf['params']['id'])
            # print "m", m.group(0), m.group(1), m.group(2), m.group(3), m.group(4)
            # if m.group(2) is not None:
            # xid = (m.group(2), m.group(4))
            # else:
            # xid = self.conf['params']['id'].split('_')[-1]
            # print "m", "split", xid

            # id string rewriting
            xid_ = self.conf['params']['id'].split('_')
            self._debug("    conf['params']['id'] = %s, xid_ = %s" % (self.conf['params']['id'], xid_, ))
            if len(xid_) == 1:
                xid = '0'
            elif xid_[-1].isdigit():
                xid = xid_[-1][len(loop_delim[1:]):]
            else:
                xid = (
                    '_' + '_'.join([item for item in xid_ if item[:2] == loop_delim[1:]]),
                    xid_[-1][len(loop_delim[1:]):]
                )
            
            # print  "    init_subgraph subgraph_rewrite_id", self.conf['params']['id'], "xid", xid

            # replace id strings recursively in graph dict
            self.conf['params']['graph'] = dict_replace_idstr_recursive2(
                d = self.conf['params']['graph'], xid = xid)
            
            # replace id strings recursively in outputs dict
            nks_l = dict_get_nodekeys_recursive(self.conf['params']['graph'])
            if 'outputs' in self.conf['params']:
                # get the outputs dict
                d_outputs = self.conf['params']['outputs']
                # replace bus references
                d_outputs = dict_replace_nodekeys_loop(d_outputs, nks_0, xid)
                # update outputs attribute
                self.outputs = d_outputs

            # print "nks", xid, nks_0, nks_l
            # print "%s-%s d_outputs = %s" % (self.cname, self.id, self.conf['params']['outputs']) # d_outputs
            
            # self.conf['params']['outputs'] = dict_get_nodekeys_recursive(
            #     d = self.conf['params']['outputs'])
            # , xid = self.conf['params']['id'][-1:])
        
    def init_graph_pass_1(self):
        """Block2.init_graph_pass_1

        Initialize this block's graph by instantiating all graph nodes
        """
        # if we're coming from non topblock init
        # self.graph = self.conf['params']['graph']

        self._debug("{2}{0: <20}-{3}.init_graph_pass_1 graph.keys = {1}".format(self.cname[:20], self.nxgraph.nodes(), self.nesting_indent, self.id))
        
        # if hasattr(self, 'graph'):
        #     print "    graph", self.graph, "\n"
        #     print "    nxgraph", self.nxgraph, "\n"
        
        # pass 1 init
        for i in nxgraph_nodes_iter(self.nxgraph, 'enable'):
            v = self.nxgraph.node[i]
            k = v['params']['id']
            # print "%s-%s.init_graph_pass_1 node = %s" % (self.cname, k, v.keys()), v['params'].keys()
            # if v['params'].has_key('outputs'):
            #     print v['params']['outputs']
            # v = n['params']
            # self.debug_print("__init__: pass 1\nk = %s,\nv = %s", (k, print_dict(v)))

            # debug timing
            self._debug("{3}{0: <20}.init pass 1 k = {1: >5}, v = {2: >20}".format(
                self.__class__.__name__[:20], k[:20], v['block'].__name__, self.nesting_indent))
            then = time.time()

            # print v['block_']
            
            # instantiate block
            # self.graph[k]['block'] = self.graph[k]['block'](conf = v, paren = self, top = self.top)
            v['block_'] = v['block'](conf = v, paren = self, top = self.top)
            # print "block_color", self.nxgraph.node[i]['block_'].block_color
            
            # print "%s init self.top.graph = %s" % (self.cname, self.top.graph.keys())
            
            # complete time measurement
            # self._debug("{3}{0: <20}.init pass 1 k = {1: >5}, v = {2: >20}, time = {4}".format(
            #     self.__class__.__name__[:20], k, v['block_'].cname,
            #     self.nesting_indent, then,
            # ))
            now = time.time()
            self._debug("{3}{0: <20}.init pass 1 k = {1: >5}, v = {2: >20}, took = {5}s".format(
                self.__class__.__name__[:20], k, v['block_'].cname,
                self.nesting_indent, now, now - then,
            ))
            
            # print "%s self.graph[k]['block'] = %s" % (self.graph[k]['block'].__class__.__name__, self.graph[k]['block'].bus)
            if v['block_'].blocksize < self.blocksize_min:
                self.blocksize_min = v['block_'].blocksize
        # done pass 1 init

    def init_graph_pass_2(self):
        """Block2.init_graph_pass_2

        Pass 2 of graph initialization: Iterate nodes and call pass2 of block instance init.

        Arguments: None

        Returns: None
        """
        # iterate over nxgraph's nodes
        # for i in range(self.nxgraph.number_of_nodes()):
        for i in nxgraph_nodes_iter(self.nxgraph, 'enable'):
            v = self.nxgraph.node[i]
            k = v['params']['id']
            
            # self.debug_print("__init__: pass 2\nk = %s,\nv = %s", (k, print_dict(v)))
            # print "%s.init pass 2 k = %s, v = %s" % (self.__class__.__name__, k, v['block'].cname)
            self._debug("{3}{0: <20}.init pass 2 k = {1: >5}, v = {2: >20}".format(self.__class__.__name__[:20], k, v['block_'].cname, self.nesting_indent))
            then = time.time()
            # self.graph[k]['block'].init_pass_2()
            v['block_'].init_pass_2()
            self._debug("{3}{0: <20}.init pass 2 k = {1: >5}, v = {2: >20}, took = {4}s".format(
                self.__class__.__name__[:20], k, v['block_'].cname, self.nesting_indent, time.time() - then))

        # for k, v in self.graph.items():
        #     v['block'].step()
            
    def init_outputs(self):
        """Block2.init_outputs

        Initialize this block's outputs:
        1. check if ROS enabled and create pub/sub dicts
        2. iterate over self.outputs and init logging, outkey self attr, bus
        """
        # print "%s.init_outputs: inputs = %s" % (self.cname, self.inputs)
        # create outputs
        # new format: outkey = str: outval = {val: value, shape: shape, dst: destination, ...}
        self.oblocksize = 0
        
        # ros?
        if hasattr(self, 'ros') and self.ros:
            self.pubs = {}
            self.subs = {}
            self.msgs = {}

        for k, v in list(self.outputs.items()): # problematic
        # for k, v in self.conf['params']['outputs'].items():
            # print "%s.init_outputs: outk = %s, outv = %s" % (self.cname, k, v)
            assert type(v) is dict, "Old config of block %s, output %s, type %s, %s" % (self.id, k, type(v), v)

            # check type
            if 'type' not in v:
                v['type'] = 'ndarray'
                
            if v['type'] == 'ndarray':
                self.init_outputs_ndarray(k = k, v = v)
            elif v['type'] == 'latex':
                self.init_outputs_latex(k = k, v = v)
            elif v['type'] in ['fig', 'plot']:
                self.init_outputs_plot(k = k, v = v)

    def init_outputs_latex(self, k = None, v = None):
        # print "Block2.init_outputs_latex k", k, "v", v
        pass
        
    def init_outputs_plot(self, k = None, v = None):
        # print "Block2.init_outputs_plot k", k, "v", v
        pass
                
    def init_outputs_ndarray(self, k = None, v = None):

        assert k is not None, "init_outputs_ndarray called with output key = None"
        assert v is not None, "init_outputs_ndarray called with output val = None"

        # auto-fix shape for buscopy
        if 'buscopy' in v and 'shape' not in v and v['buscopy'] in self.bus:
            v['shape'] = self.bus[v['buscopy']].shape
        
        # assert v.keys()[0] in ['shape', 'bus'], "Need 'bus' or 'shape' key in outputs spec of %s" % (self.id, )
        assert 'shape' in v, "%s-%s's output spec %s needs 'shape' param but has %s " % (self.cname, self.id, k, list(v.keys()))
        # if v.has_key('shape'):
        assert len(v['shape']) > 1, "Block %s, output %s 'shape' tuple is needs at least (dim1 x output blocksize), v = %s" % (self.id, k, v)
        # # create new shape tuple by appending the blocksize to original dimensions
        # if v['shape'][-1] != self.blocksize: # FIXME: heuristic
        #     v['bshape']  = v['shape'] + (self.blocksize,)
        #     v['shape']   = v['bshape']
        if v['shape'][-1] > self.oblocksize:
            self.oblocksize = v['shape'][-1]

        # compute buskey from id and variable name
        v['buskey'] = "%s/%s" % (self.id, k)

        # logging by output item
        if 'logging' not in v:
            v['logging'] = True
                
        # set self attribute to that shape
        v['shape'] = tuple2inttuple(v['shape'])
        if not hasattr(self, k) or getattr(self, k).shape != v['shape']:
            setattr(self, k, np.zeros(v['shape']))
            
        # print "%s.init_outputs: %s.bus[%s] = %s" % (self.cname, self.id, v['buskey'], getattr(self, k).shape)
        self.bus[v['buskey']] = getattr(self, k).copy()
        # self.bus.setval(v['buskey'], getattr(self, k))

        # ros?
        if hasattr(self, 'ros') and self.ros:
            import rospy
            from std_msgs.msg import Float64MultiArray
            self.msgs[k] = Float64MultiArray()
            self.pubs[k] = rospy.Publisher('%s/%s' % (self.id, k, ), Float64MultiArray, queue_size = 2)
            
        # output item initialized
        v['init'] = True

    def init_logging(self):
        # initialize block exec logging
        if self.debug:
            self.loglevel = logging_DEBUG
        else:
            self.loglevel = logging_INFO
        
        # initialize block data logging
        if not self.logging: return

        # assume output's initialized        
        for k, v in list(self.outputs.items()):
            if ('init' not in v) or (not v['init']) or (not v['logging']): continue
                
            # FIXME: ellipsis
            tbl_columns_dims = "_".join(["%d" for axis in v['shape'][:-1]])
            tbl_columns = [tbl_columns_dims % tup for tup in xproduct(itertools.product, v['shape'][:-1])]
            # print "tbl_columns", tbl_columns

            # initialize the log table for this block
            log.log_pd_init_block(
                tbl_name    = v['buskey'], # "%s/%s" % (self.id, k),
                tbl_dim     = v['shape'],
                tbl_columns = tbl_columns,
                numsteps    = (self.top.numsteps / self.blocksize) * self.oblocksize,
                blocksize   = self.blocksize,
            )
                
        # # FIXME: make one min_blocksize bus group for each node output
        # for outkey, outparams in self.nodes[nk].outputs.items():
        #     nodeoutkey = "%s/%s" % (nk, outkey)
        #     print "bus %s, outkey %s, odim = %d" % (nk, nodeoutkey, outparams[0])
        #     self.bus[nodeoutkey] = np.zeros((self.nodes[nk].odim, 1))
                                
    def init_pass_2(self):
        """Block2.init_pass_2

        Second init pass which is needed for connecting inputs to outputs after
        they have been initialized to infer the bus shapes.
        """
        if not self.topblock:
            # create inputs by mapping from constants or bus
            # that's actually for pass 2 to enable recurrent connections
            # old format: variable: [buffered const/array, shape, bus]
            # new format: variable: {'val': buffered const/array, 'shape': shape, 'src': bus|const|generator?}
            for k, v in list(self.inputs.items()):
                self._debug("init_pass_2 input items ink = %s, inv = %s" % (k, v, ))
                assert len(v) > 0
                # FIXME: when is inv not a dict?
                # assert type(v) is dict, "input value %s in block %s/%s must be a dict but it is a %s, probably old config" % (k, self.cname, self.id, type(v))
                # assert v.has_key('shape'), "input dict of %s/%s needs 'shape' entry, or do something about it" % (self.id, k)
                
                # set input from bus
                if 'bus' in v:
                    if 'shape' in v:
                        # init input buffer from configuration shape
                        # print "input config shape = %s" % (v['shape'][:-1],)
                        # if len(v['shape']) == 1:
                        #     vshape = (v['shape'][0], self.blocksize)
                        # else:
                        #     # vshape = v['shape'][:-1] + (self.ibuf,)
                        #     vshape = v['shape']
                        assert len(v['shape']) > 1, "Shape must be length == 2"

                        # clamp input_shape[1] to min(numsteps, input_shape[1])
                        # FIXME: collision args.numsteps and overproducing nodes (example_windowed)
                        #        input clamp necessary at all?
                        if self.inputs_clamp:
                            v['shape'] = (v['shape'][0], min(self.top.numsteps, v['shape'][1]))

                        # tuple of ints to make sure
                        v['shape'] = tuple2inttuple(v['shape'])
                            
                        # initialize input buffer
                        v['val'] = np.zeros(v['shape']) # ibuf >= blocksize
                        
                        # bus item does not exist yet
                        if v['bus'] not in self.bus:
                            
                            # FIXME: hacky
                            for i in range(1): # 5
                                # if i == 0: print "\n"
                                # print "%s-%s init (pass 2) WARNING: bus %s doesn't exist yet and will possibly not be written to by any block, buskeys = %s" % (self.cname, self.id, v['bus'], self.bus.keys())
                                self._warning("init (pass 2) WARNING: nonexistent bus %s at nesting %s" % (v['bus'], self.nesting_indent))
                                # if not self.top.recurrent: time.sleep(1)
                                    
                            # pre-init that bus from constant
                            self.bus[v['bus']] = v['val'].copy()
                        else:
                            # sl = slice(-blocksize_input, None)
                            blocksize_input_bus = self.bus[v['bus']].shape[-1]
                            # vs = v['shape']
                            # vv = v['val']
                            # bus_vbus = self.bus[v['bus']]
                            if blocksize_input_bus > v['shape'][-1]:
                                # mcnt = xself.cnt % blocksize_input_bus
                                # sl = slice(mcnt - v['shape'][-1] + 1, mcnt + 1) #
                                sls = (self.cnt - v['shape'][-1]) % blocksize_input_bus
                                sle = sls + v['shape'][-1] # xself.cnt % blocksize_input_bus
                                sl = slice(sls, sle) #
                                
                                v['val'] = self.bus[v['bus']][...,sl].copy()
                                # print "#" * 80
                                # print "sl", sl, v['val']
                            else:
                                # print "\nsetting", self.cname, v
                                assert v['shape'][0] == self.bus[v['bus']].shape[0], "%s-%s's input buffer and input bus shapes need to agree (besides blocksize) for input %s, buf: %s, bus: %s/%s" % (self.cname, self.id, k, v['shape'], v['bus'], self.bus[v['bus']].shape)
                                v['val'][...,-blocksize_input_bus:] = self.bus[v['bus']].copy()
                            # v['val'] = self.bus[v['bus']].copy() # inbus
                            # v['val'][...,0:inbus.shape[-1]] = inbus
                        # print "Blcok2: init_pass_2 v['val'].shape", self.id, v['val'].shape
                        
                    elif 'shape' not in v:
                        # check if key exists or not. if it doesn't, that means this is a block inside dynamical graph construction
                        # print "\nplotblock", self.bus.keys()

                        assert v['bus'] in self.bus, "%s-%s requested bus item = %s which is not in bus = %s.\n    Cannot infer shape. Add shape to input %s of block %s?" % (self.cname, self.id, v['bus'], list(self.bus.keys()), k, self.id)
                    
                        # enforce bus blocksize smaller than local blocksize, tackle later
                        # CHECK
                        # assert self.bus[v['bus']].shape[-1] <= self.blocksize, "input block size needs to be less than or equal self blocksize in %s/%s, in %s, should %s, has %s\ncheck blocksize param" % (self.cname, self.id, v['bus'], self.bus[v['bus']].shape[-1], self.blocksize)
                        # get shortcut
                        inbus = self.bus[v['bus']]
                        # print "init_pass_2 inbus.sh = %s, self.bs = %s" % (inbus.shape[:-1], self.blocksize)
                        # if no shape given, take busdim times input buffer size
                        # v['val'] = np.zeros(inbus.shape[:-1] + (self.blocksize,)) # ibuf >= blocksize inbus.copy()
                        # v['val'] = np.zeros(inbus.shape[:-1] + (self.blocksize, ))
                        # think just inbus is better, otherwise enforce shape spec
                        v['val'] = np.zeros(inbus.shape)
                                        
                    v['shape'] = v['val'].shape # self.bus[v['bus']].shape
                    
                    # print "\n%s init_pass_2 ink %s shape = %s / %s" % (self.id, k, v['val'].shape, v['shape'])
                    self._debug("init_pass_2 input items ink = %s, bus[%s] = %s, input = %s" % (k, v['bus'], self.bus[v['bus']].shape, v['val'].shape))
                # elif type(v[0]) is str:
                #     # it's a string but no valid buskey, init zeros(1,1)?
                #     if v[0].endswith('.h5'):
                #         setattr(self, k, v[0])
                else:
                    assert 'bus' in v or 'val' in v, "%s-%s's input spec needs either 'bus' or 'val' entry in %s" % (
                        self.cname, self.id, list(v.keys()))
                    # expand scalar to vector
                    if np.isscalar(v['val']):
                        # check for shape info
                        if 'shape' not in v:
                            v['shape'] = (1,1)
                        # create ones multiplied by constant
                        v['val'] = np.ones(v['shape']) * v['val']
                    else:
                        # print "isarray", v['val']
                        # write array shape back into config
                        v['shape'] = v['val'].shape
                    # self.inputs[k].append(None)
                # add input buffer
                # stack??
                # self.inputs[k][0] = np.hstack((np.zeros((self.inputs[k][1][0], self.ibuf-1)), self.inputs[k][0]))
                # self._debug("init_pass_2 k = %s, v = %s" % (k, str(v)[:60], ))
                self._debug("init_pass_2 input items ink = %s inv['val'].shape/inv['shape'] = %s / %s" % (k, v['val'].shape, v['shape']))
            
    def block_is_scheduled(self):
        """Block is scheduled when its count modulo its blocksize is element of the blockphase array
        """
        # print "self.cnt", self.cnt, "self.blocksize", self.blocksize
        return (self.cnt % self.blocksize) in self.blockphase

    def block_is_finished(self):
        """Block is finished when its count equals toplevel number of steps
        """
        # and hasattr(xself, 'isprimitive') and self.isprimitive:
        # print "block_is_finished self.cnt = %d, top.numsteps = %d" % (self.cnt, self.top.numsteps)
        return self.cnt == self.top.numsteps
    
    def block_is_composite(self):
        # list of necessary conditions for compositeness
        conditions = [
            # block conf directly contains graph dict
            hasattr(self, 'graph'),
            # block conf contains subgraph either as dict or as filename
            hasattr(self, 'subgraph'),
            # block is a LoopBlock2 with the implicit semantics that the loop is unrolled into anexplicit subgraph during init
            (hasattr(self, 'loopblock') and len(self.loopblock) != 0),
            # block is a ModelBlock2 with multiple models supplied in its 'models' dict
            (hasattr(self, 'models') and len(self.models) > 1),
        ]
        # return OR term as reduction of list
        return reduce(lambda t1,t2: t1 or t2, conditions)
            
    def output_is_type(self, k, v, typematch = 'ndarray'):
        return 'type' in v and v['type'] != typematch

    def output_is_triggered(self, k, v, bus):
        # return true if is 'trigger' but trigger bus inactive
        if 'trigger' not in v: return True # False
        istriggered = 'trigger' in v and v['trigger'] in bus and np.any(bus[v['trigger']] > 0)
        # return not istriggered
        if istriggered:
            if 'trigger_func' in v:
                v['trigger_func'](self)
        return istriggered
    
    def set_attr_from_top_conf(self):
        """set self attributes copied from corresponding toplevel attributes
        
        FIXME: namespace foo
        """
        for attr in ['saveplot']:
            top_attr = getattr(self.top, attr)
            if top_attr is not None and attr in self.conf['params']:
                # print "Block2.set_attr_from_top_conf copying top.%s = %s to conf['params'] %s" % (attr, top_attr, self.conf['params']['saveplot'])
                self.conf['params'][attr] = top_attr
                setattr(self, attr, top_attr)

    def get_nesting_level(self):
        """get the current graphs nesting level in a composite graph
        """
        nl = 0
        newparen = self
        while newparen is not None and not newparen.topblock:
            nl += 1
            # print "nl", nl, newparen, 
            if newparen == newparen.paren:
                newparen is None
            else:
                newparen = newparen.paren
                
        return nl + 1

    def blocksize_clamp(self):
        """Block2.blocksize_clamp

        Clamp blocksize to numsteps if numsteps < blocksize
        """
        self.blocksize = min(self.top.numsteps, self.blocksize)

    def init_colors(self):
        """Compute block identity and infer the node's plot color
        """
        # print "block_cmaps", block_cmaps
        def get_color_from_confstr(confstr):
            # print "type", type(plot_colors)
            linelen = 20 # len(plot_colors.keys()) # 1000
            b = np.fromstring(str(self.conf), dtype=np.uint8)
            bmod = b.shape[0] % linelen
            # print "Block2-%s.init_colors conf bitvec = %s, modlen = %s" % (self.id, b.shape, bmod)
            # if bmod != 0:
            b = np.pad(b, (0, linelen - bmod), mode = 'constant')
            # print "id", self.id, "b", b.shape
            # print "Block2-%s.init_colors conf bitvec = %s, modlen = %s" % (self.id, b.shape, bmod)
            # c = b.reshape((-1, linelen)).mean(axis = 0)
            # c = (b.min(), b.mean(), b.max())
            # c = np.mean(b/np.max(b))
            c = int(np.sum(b) % linelen)
        
            ck = list(plot_colors.keys())[c]
            # print "Block2-%s.init_colors k = %s, ck = %s, color = %s" % (self.id, c, ck, plot_colors[ck])
            return plot_colors[ck]

        def get_color_from_plot_colors():
            if not hasattr(self.top, 'colorcnt'):
                self.top.colorcnt = 0
            else:
                self.top.colorcnt += 1
            ck =  list(plot_colors.keys())[self.top.colorcnt]
            return plot_colors[ck]
            
        def get_color_from_cmap():
            # if not hasattr(self.top, 'colormap'):
            #     self.top.colormap = plt.get_cmap('gist_ncar')
                
            if not hasattr(self.top, 'colorcnt'):
                self.top.colorcnt = 0
            else:
                self.top.colorcnt += 1
                
            # return plot_colors.keys()[self.top.colorcnt]
            group_cmap = block_groups[self.block_group[0]]['cmap']
            # print "%s init_colors is group %s, cmap = %s" % (self.id, self.block_group[0], group_cmap)
            return plt.get_cmap(group_cmap)(self.top.colorcnt / 77.0)
            
        # ck = get_color_from_plot_colors()
        
        # block color
        # self.block_color = get_color_from_plot_colors()
        self.block_color = get_color_from_cmap()
    
    def debug_print(self, fmtstring, data):
        """only print if debug is enabled for this block"""
        fmtstring = "\n%s[%d]." + fmtstring
        data = (self.cname,self.cnt) + data
        if self.debug:
            print(fmtstring % data)

    def step_cache(self):
        """Block2.step_cache

        Compute data for all subordinate nodes by outputting this block's cache
        """
        print("%s-%s cached" % (self.cname,self.id,))
        pass

    def step_compute(self):
        """Block2.step_compute

        Compute or recompute data for all nodes in the graph, as
        opposed to cache playback
        """
        for i in nxgraph_nodes_iter(self.nxgraph, 'enable'):
            # get node
            v = self.nxgraph.node[i]
            # get node id
            k = v['params']['id']
            # step node
            v['block_'].step()
            # debug
            # print "%s-%s.step[%d]: k = %s, v = %s" % (self.cname, self.id, self.cnt, k, type(v))
            
        # all Block2's
        if (self.cnt % self.blocksize) in self.blockphase:
            # buscopy: copy outputs from subblocks as configured in enclosing block outputs spec
            self.bus_copy()
            
    # Block2.step
    @decStep()
    def step(self, x = None):
        """Block2.step

        block step function: compute one step

        if topblock iterate graph and step each node block, reiterate graph and do the logging for each node, store the log every n steps
        """
        # if self.topblock or hasattr(self, 'subgraph'):
        # if hasattr(self, 'graph') or hasattr(self, 'subgraph'):
        # mode 1 for handling hierarchical blocks: graph is flattened during init, only topblock iterates nodes
        # first step all
        # for i in range(self.nxgraph.number_of_nodes()):
        # print "%s-%s[%d] step before cache" % (self.cname, self.id, self.cnt, )
        if not self.topblock and self.top.docache and self.cache_loaded: # hasattr(self, 'cache') and self.cache is not None and len(self.outputs) > 0:
            self.step_cache()
        else:
            self.step_compute()
        
        if self.topblock:

            # print "topblock step cnt = %d" % (self.cnt, )
            # store log incrementally
            if (self.cnt) % 500 == 0 or self.cnt == (self.numsteps - 1) or self.cnt == (self.numsteps - self.blocksize_min):
                self._info("storing log @iter % 4d/%d" % (self.cnt, self.numsteps))
                log.log_pd_store()

            # store log finally: on final step, also copy data attributes to log attributes
            if self.cnt == self.numsteps:
                # close all outputs
                
                # store and close logging
                self.log_close()
                
                # save plot figures
                self.plot_close()

                # latex output?
                self.latex_close()
                
        # need to to count ourselves
        # self.cnt += 1

    def bus_copy(self):
        """Compute block output by copying data from the bus argument
        """
        for k, v in [(k_, v_) for k_, v_ in list(self.outputs.items()) if 'buscopy' in v_]:
            buskey = v['buscopy']
            assert buskey in self.bus, "Assuming in %s-%s that bus has key %s but %s" % (self.cname, self.id, buskey, list(self.bus.keys()))
            # if buskey.startswith('b4') and np.mean(self.bus[buskey]) != 0.0: # or 'measure' in buskey:
            #     # , getattr(self, k), self.bus[buskey], self.bus.keys()
            #     print "buscopy[%d]: from buskey = %s to bus %s/%s" % (self.cnt, buskey, self.id, k)
            #     print "         data = %s" % (self.bus[buskey], )
            self._debug("    buscopy outputs[%s] from bus[%s] = %s" % (k, buskey, self.bus[buskey]))
            setattr(self, k, self.bus[buskey])
            
        # for k, v in self.outputs.items():
        #     if v.has_key('buscopy'):
        #         # print "Block2-%s.step[%d].buscopy: buskey = %s, buskeys = %s" % (self.id, self.cnt, v['buscopy'], self.bus.keys())
        #         buskey = v['buscopy']
        #         # if self.bus.has_key(buskey):
        #         setattr(self, k, self.bus[buskey])
        #         # else:
        #         # print "%s-%s[%d] self.%s = %s from bus %s / %s" % (
        #         #     self.cname, self.id, self.cnt, k, getattr(self, k), v['buscopy'], self.bus[v['buscopy']])
        
    def latex_close(self):
        """close latex output channel if one is configured

        Map topblock conf variables into a latex template to generate
        a fragment that describes the experiment and includes all
        relevant output items, usually plots.

        Freestyle text copied verbatim from conf keys:
         - id, title, desc_expr, desc_result

        Depending on the configuration, relevant parameter
        descriptions and figures are generated systematically.

        FIXME: use pylatex (if available)?
        """
        # check if latex output is configured
        output_latex = [(k, v) for k, v in list(self.outputs.items()) if 'type' in v and v['type'] == 'latex']
        if len(output_latex) < 1:
            self._debug("latex_close: no latex output configured, output_latex = %s" % (output_latex, ))
            return

        # write latex fragment for experiment
        latex_filename = '%s/%s_texfrag.tex' % (self.datadir_expr, self.id)
        f = open(latex_filename, 'w')

        texbuf = ''
        # for k, v in self.latex_conf.items():
        id_ = re.sub('_', '-', self.id)
        texbuf = '\\subsection*{%s}\n\\label{sec:%s}\n\n' % (self.expr_name, id_, )

        # copy expr description from conf
        id_base_ = '\_'.join(self.id.split('_')[0:2])
        texbuf += '\mypara{%s description}%s\n' % (id_base_, self.desc, )

        outputs_keys = list(self.outputs.keys())
        outputs_keys.sort()
        self._debug("outputs_keys = %s" % (outputs_keys, ))
        output_figures = [(k, self.outputs[k]) for k in outputs_keys if 'type' in self.outputs[k] and self.outputs[k]['type'] in ['fig', 'plot']]
        # output_figures = [(k, v) for k, v in self.outputs.items() if v.has_key('type') and v['type'] in ['fig', 'plot']]
        # print "Block2.latex_close: |output_figures| = %s" % ( len(output_figures), )

        """
        if len(output_figures) > 0:
            descs = []
            refs = []
            texbuf += "\\begin{figure}[!htbp]\n  \\centering\n"
            for figk, figv in output_figures:
                print "Block2.latex_close: output_figures loop k = %s, v = %s" % (figk, figv)
                # if k is 'figures':
                # if len(v) > 0:
                # for figk, figv in v.items():
                desc = figv['desc']
                figlabel_ = re.sub('_', '-', figv['label'])
                figid_ = re.sub('_', '-', figv['id'])
                ref = "%s-%s" % (figlabel_, figid_, ) # 
                texbuf += "  \\begin{subfigure}[]{0.99\\textwidth}\n    \\centering\n\
    \\includegraphics[width=0.99\\textwidth]{%s}\n\
	\\caption{\\label{fig:%s}}\n\  \\end{subfigure}\n" % (figv['filename'], ref, )
                refs.append(ref)
                descs.append(desc)

            c_ = ','.join(["%s \\autoref{fig:%s}" % (desc, ref) for (desc, ref) in zip(descs, refs)])
            caption = "  \\caption{\\label{fig:%s}%s.}\n\\end{figure}\n" % (figv['label'], c_)
            texbuf += caption
        """
        
        figc = 0
        figrefs = []
        figbuf_all = ''
        if len(output_figures) > 0:
            for figk, figv in output_figures:
                descs = []
                refs = []
                figbuf = "%% figure key = {0}, label = {1}\n".format(figk, figv['label'])
                # figbuf += "\\begin{figure}[!htbp]\n  \\centering\n"
                # figbuf += "\\begin{figure}[!htbp]\n  \\captionsetup[subfigure]{position=t}\n  \\centering\n"
                figbuf += "\\addfig{\\begin{minipage}[]{\\textwidth}\n  \\centering\n"
                # print "                    output_figures fig key = %s, fig val = %s" % (figk, figv)
                # if k is 'figures':
                # if len(v) > 0:
                # for figk, figv in v.items():

                # get filename for includegraphics
                figfilename = figv['filename']
                
                # fix array repr for scalar entries
                if type(figfilename) is str:
                    figfilename = [figfilename]
                    figdesc = [figv['desc']]
                    figlabel = [figv['label']]
                    figid = [figv['id']]
                else:
                    # figfilename = figfilename
                    figdesc = figv['desc']
                    figlabel = [figv['label']] * len(figfilename)
                    figid = figv['id']
                    
                # loop over subfigure array
                for i, figfilename_ in enumerate(figfilename):
                    figdesc_ = figdesc[i]
                    figlabel_ = re.sub('_', '-', figlabel[i])
                    figid_ = re.sub('_', '-', figid[i])
                    figref_ = "%s-%s" % (figlabel_, figid_, )
                    
                    # print "                    i = %d, filename = %s" % (i, figfilename_)
                    subfigref_ = '%s-%d' % (figref_, i)

                    figwidth_ = '0.99\\textwidth'
                    if 'width' in figv:
                        if type(figv['width']) is list:
                            if figv['width'][i] is not None:
                                figwidth_ = figv['width'][i]
                        else:
                            figwidth_ = figv['width']
                    if type(figwidth_) is float:
                        figwidth_ = '%f\\textwidth' % (figwidth_, )
                    subfigwidth_ = '0.99\\textwidth' # figwidth_

                    figbuf += "  \\includegraphics[width={0}]{{{2}}}\n\\label{{fig:{3}}}\n".format(figwidth_, subfigwidth_, figfilename_, subfigref_, )
                    # figbuf += "  \subcaptionbox{{\\label{{fig:{3}}}}}{{\\includegraphics[width={0}]{{{2}}}}}\n".format(figwidth_, subfigwidth_, figfilename_, subfigref_, )
  #                   figbuf += "  \\begin{subfigure}[]{%s}\n    \\centering\n\
  #   \\includegraphics[width=%s]{%s}\n\
  #   \\caption{\\label{fig:%s}}\n\
  # \\end{subfigure}\n" % (figwidth_, subfigwidth_, figfilename_, subfigref_, )

                    refs.append(subfigref_)
                    descs.append(figdesc_)

                # figrefs.append(figref_)
                figrefs.append('%s-%s' % (figlabel_, figk))
                
                # c_ = ', '.join(["%s \\autoref{fig:%s}" % (desc, ref) for (desc, ref) in zip(descs, refs)])
                c_ = ', '.join(["%s" % (desc, ) for (desc, ref) in zip(descs, refs)])
                caption = "  \\end{minipage}\n\\captionof{figure}{\\label{fig:%s-%s}Experiment \\ref{sec:%s}-%d %s}}\n\n\n" % (figlabel_, figk, id_, figc + 1, c_)
                # caption = "  \\caption{\\label{fig:%s-%s}%s %s.}\n\\end{figure}\n\n\n" % (figlabel_, figk, id_, c_)
                figbuf += caption
                
                figbuf_all += figbuf
                # if figc > 0 and figc % 2 == 0:
                if figc % 2 == 1:
                    figbuf_all += '\\mbox{}\\smpnewpage\n'
                figc += 1

        # Compile list with references to all figures in this experiment
        figrefsbuf = 'The results are shown in the figures ' + ', '.join(['\\ref{{fig:{0}}}.'.format(label) for label in figrefs]) + '\n'

        # append figure refs to tex buffer
        texbuf += figrefsbuf

        # append figure code to tex buffer
        texbuf += figbuf_all
                
        # call optional closing hooks
        texbuf += '\\mbox{}\\smpnewpage\n'

        # write the text buffer and close the file
        f.write(texbuf)
        f.flush()
        f.close()
        
    def plot_close(self):
        """plot_close saves all plot block figures configured for saving

        Iterate the top graph, search for nodes that have 'saveplot'
        attr set and call their `node.save()` function.
        """
        # print "%s-%s\n    .plot_close closing %d nodes" % (self.cname, self.id, self.nxgraph.number_of_nodes())
        for n in nxgraph_nodes_iter(self.nxgraph, 'enable'):
            node = self.nxgraph.node[n]['block_']
            if hasattr(node, 'nxgraph'):
                # descend
                node.plot_close()

            if hasattr(node, 'saveplot'):
                self._debug("%s-%s.plot_close examining node %s (%s)" % (self.cname, self.id, node.id, node.saveplot))
            
                # if type(node) is Block2 and hasattr(node, 'saveplot') and node.saveplot:
                if node.saveplot:
                    self._debug("%s-%s.plot_close closing node, saving plot %s" % (self.cname, self.id, node.id,))
                    try:
                        node.save()
                    except Exception as e:
                        logger.error('%s-%s.plot_close node(%s-%s).save() failed with %s' % (self.cname, self.id, node.cname, node.id, e))

    def _debug(self, s, *args, **kwargs):
        if self.debug and self.loglevel == logging_DEBUG: # <= logger.loglevel:
            self._log(logging_DEBUG, s, *args, **kwargs)
                        
    def _info(self, s, *args, **kwargs):
        self._log(logging_INFO, s, *args, **kwargs)
            
    def _warning(self, s, *args, **kwargs):
        self._log(logging_WARNING, s, *args, **kwargs)
            
    def _log(self, loglevel = logging_INFO, s = '', *args, **kwargs):
        """Block2._log wrapper

        Logging facility for blocks. Calls the calling module's logger
        `log` function taking care of the block's loglevel.

        Arguments:
         - s(str): log string

        Returns:
         - None
        """
        self.logger.log(loglevel, s, *args, **kwargs)
        
    def log_close(self):
        self._info("storing log @final iter %04d" % (self.cnt, ))
        # store
        log.log_pd_store()
        # recursively copy the attributes
        self.log_attr()
        # close the file
        log.log_pd_deinit()
                    
    def log_attr(self):
        """Block2.log_attr: enumerate all nodes in hierarchical graph and copy the node's output attributes to table attributes"""
        # for i in range(self.nxgraph.number_of_nodes()):
        for i in nxgraph_nodes_iter(self.nxgraph, 'enable'):
            # assume output's initialized
            node = self.nxgraph.node[i]['block_']
            # print "%s-%s log_attr\n    node = %s, logging = %s" % (self.cname, self.id, node, node.logging)
            if not node.logging: continue

            # depth first
            if hasattr(node, 'nxgraph'):
                # recur
                node.log_attr()

            # loop output items
            for k,v in list(node.outputs.items()):
                if ('init' not in v) or (not v['init']) or (not v['logging']): continue
        
                tbl_columns_dims = "_".join(["%d" for axis in v['shape'][:-1]])
                tbl_columns = [tbl_columns_dims % tup for tup in xproduct(itertools.product, v['shape'][:-1])]

                # set the log table attributes for this output item
                log.log_pd_init_block_attr(
                    tbl_name    = v['buskey'], # "%s/%s" % (node.id, k),
                    tbl_dim     = v['shape'],
                    numsteps    = (node.top.numsteps / node.blocksize) * node.oblocksize,
                    blocksize   = node.blocksize,
                )
                    
    def get_config(self):
        """Block2.get_config: get the current node instance as a dictionary"""
        params = {}
        for k, v in list(self.__dict__.items()):
            # FIXME: include bus, top, paren?
            if k not in ['conf', 'bus', 'top', 'paren']:
                params[k] = v
            
        conf = ('%s' % self.id,
                {
                    'block': self.cname,
                    'params': params
                })
        return conf

    def dump_final_config(self):
        """Block2.dump_final_config get the current configuration and dump it into a text file"""
        finalconf = self.get_config()
        # print "Block2.dump_final_config", type(finalconf), finalconf
        # pickle.dump(finalconf, open("data/%s_conf.pkl" % (self.id, ), "wb"))
        # dump_final_config_file = "data/%s.conf" % (self.id)
        dump_final_config_file = "%s_final.conf" % (self.datafile_expr, )
        f = open(dump_final_config_file, "w")
        # confstr = repr(finalconf[1]['params'])
        confstr = print_dict(pdict = finalconf[1]['params']['graph'])
        # print "graph", confstr
        # confstr_ = "numsteps = %d\nconf = {'block': 'Block2', 'params': %s}" % (self.numsteps, confstr, )
        confstr_ = conf_header + "numsteps = %d\ngraph = %s" % (self.numsteps, confstr, ) + conf_footer
        # print confstr_
        f.write(confstr_)
        f.flush()
        # print "%s.dump_final_config wrote config, closing file %s" % (self.cname, dump_final_config_file,)
        f.close()

        return confstr_
        # log.log_pd_store_config_final(confstr_)

    def check_attrs(self, attrs):
        """Block2: check if a block has been given all necessary attributes via configuration"""

        for attr in attrs:
            assert hasattr(self, attr), "%s.check_attrs: Don't have attr = %s" % (self.__class__.__name__, attr)

    def get_input(self, k):
        """get_input

        failsafe preprocessing of input tensor using input specification
        """
        if k not in self.inputs:
            logger.error('%s-%s get_input failed, no key = %s in self.inputs.keys = %s' % (self.cname, self.id, k, list(self.inputs.keys())))
            return np.zeros((1,1))
        
        if 'embedding' in self.inputs[k]:
            emblen = self.inputs[k]['embedding']
            embshp = self.inputs[k]['shape']
            assert len(embshp) == 2
            ret = np.zeros((embshp[0] * emblen, embshp[1]))
            for i in range(embshp[1]):
                if i < emblen: continue
                tmp = np.hstack(tuple(self.inputs[k]['val'][:,i-j] for j in range(emblen)))
                # print "tmp", tmp.shape, # tmp
                ret[:,i] = tmp
            # print "ret", ret.shape, # 
            return ret
        else:
            # return self.inputs[k]['val']
            return get_input(self.inputs, k)

class FuncBlock2(Block2):
    """FuncBlock2 class

    Function block: wrap the function given by the 'func' configuration
    parameter in a block
    """
    defaults = {
        'func': lambda x: {'x': x['x']['val']},
        'inputs': {
            'x': {'bus': 'cnt/x'},
        },
        'outputs': {
            'x': {'shape': (1,1)},
        },
        'block_group': 'comp',
    }
        
    def __init__(self, conf = {}, paren = None, top = None):
        """FuncBlock2.__init__
        """
        # print "FuncBlock2 defaults", Block2.defaults, self.defaults
        Block2.__init__(self, conf = conf, paren = paren, top = top)

        self.check_attrs(['func'])
        
        self.func = memory.cache(self.func)
        
        # FIXME: check return type of func at init time and set step function

    @decStep()
    def step(self, x = None):
        """FuncBlock2.step

        Function block step function
        """
        # print "%s.step inputs[%d] = %s" % (self.cname, self.cnt, self.inputs)
        # self.inputs is a dict with values [array]
        # assert self.inputs.has_key('x'), "%s.inputs expected to have key 'x' with function input vector. Check config."

        self._debug("step[%d]: x = %s" % (self.cnt, self.inputs,))

        # assumes func to be smp_graphs aware and map the input/output onto the inner function 
        f_val = self.func(self.inputs)
        if type(f_val) is dict:
            for k, v in list(f_val.items()):
                setattr(self, k, v)
        else:
            for outk, outv in list(self.outputs.items()):
                # print "k, v", outk, outv, f_val
                setattr(self, outk, f_val)
            self.y = f_val
            self._debug("step[%d]: y = %s", (self.cnt, self.y,))
            
class LoopBlock2(Block2):
    """LoopBlock2 class

    Dynamically create block variations from template according to
    variations specified via list or function

    Two loop modes in smp_graphs: parallel mode (LoopBlock2) which
    modifies the graph structure at config time and instantiates all
    block variations concurrently. Sequential mode (SeqLoopBlock2)
    modifies graph at execution time by instantiating and running each
    variation one after the other.

    Parallel / LoopBlock2 parameters:
    - loop: the loop specification. either a list of tuples or a
      function returning tuples. Tuples have the form ('param', value)
      and param is a configuration parameter of the inner loopblock.
    - loopblock: conf dict for the block which is being looped
    - loopmode: used during graph construction (graphs.py)

    Examples for loop specification::

        [('inputs', {'x': {'val': 1}}), ('inputs', {'x': {'val': 2}})]

    or::

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
    defaults = {
        'loop': [1],
        'loopblock': {},
        'loopmode': 'parallel',
        'numsteps': 1,
        }
    def __init__(self, conf = {}, paren = None, top = None):
        # self.defaults['loop'] = [1]
        # self.defaults['loopblock'] = {}

        # # force loopmode 'parallel'
        # self.loopmode = 'parallel'

        # merge defaults
        self.defaults.update(conf['params'])
        conf['params'] = self.defaults
        
        # sanity check: loop specification
        assert 'loop' in conf['params'], "LoopBlock2: loop spec missing"
        assert 'loopmode' in conf['params'], "LoopBlock2: loopmode missing"

        # unroll loop into dictionary
        conf['params']['subgraph'] = LoopBlock2.subgraph_from_loop_unrolled(self, conf, paren, top)

        # print "LoopBlock2 params", conf['params'].keys()
        
        # parent takes care of initializing the subgraph
        Block2.__init__(self, conf = conf, paren = paren, top = top)

        # done

    @staticmethod
    def subgraph_from_loop_unrolled(blockref, conf, paren, top):
        loopgraph_unrolled = OrderedDict()
        logger.debug("nxgraph_from_smp_graph loopblock %s unroll %s",
            conf['params']['id'],
            conf['params']['loop'])
        
        # construction loop
        for i, item in enumerate(conf['params']['loop']):
            
            # at least a one element list of key:val tuples
            if type(item) is tuple:
                item = [item]

            cid = conf['params']['id']
            xid = '%d' % (i, )

            # # ['params']['loopblock']
            # # get parent conf
            # parenconf = copy.deepcopy(conf['params']['loopblock'])
            # # parenconf['params'].pop('loopblock')
            # # parenconf['params'].pop('loopmode')
            # # parenconf['params'].pop('loop')
            # parenconf = dict_replace_idstr_recursive(
            #     d = parenconf,
            #     cid = cid,
            #     xid = xid,)
            
            # get template and copy
            if 'loopblock' in conf['params']:
                lpconf = copy.deepcopy(conf['params']['loopblock'])
            elif 'models' in conf['params']:
                lpconf = copy.deepcopy(conf)

            # rewrite raw graph dict to composite block with raw graph as subgraph
            if type(lpconf) is OrderedDict:
                lpconf_ = {
                        'block': Block2,
                        'params': {
                            'id': None,
                            'subgraph': lpconf,
                            # 'graph': lpconf,
                        },
                }
                lpconf = lpconf_
                
            # check for numsteps
            if 'numsteps' not in lpconf['params']:
                lpconf['params']['numsteps'] = top.numsteps
                
            # rewrite block ids with loop count
            lpconf = dict_replace_idstr_recursive(d = lpconf, cid = cid, xid = xid)

            # # seems to work too
            # lpconf = dict_replace_idstr_recursive2(
            #     d = lpconf, xid = xid)
            
            # print "         - nxgraph_from_smp_graph loopblock unroll-%d lpconf = %s" % (i, lpconf, )
            # sys.exit(1)
            
            # conf['params']['subgraph'] = conf['params']['loopblock']
            # G, nc = check_graph_subgraph(conf, G, nc)
            
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
                logger.debug("    replacing conf from loop paramk = %s, paramv = %s", paramk, paramv)
                # lpconf['params'][paramk] = paramv # .copy()
                # FIXME: include id/params syntax in loop update
                if 'subgraph' in lpconf['params'] and type(lpconf['params']) is OrderedDict:
                    for (blockk, blockv) in list(lpconf['params']['subgraph'].items()):
                        if paramk in blockv['params']:
                            blockv['params'][paramk] = paramv # .copy()
                else:
                    lpconf['params'][paramk] = paramv # .copy()

            # print "nxgraph_from_smp_graph", lpconf['params']['id']
            # print "nxgraph_from_smp_graph", print_dict(lpconf['params'])
            # print "|G|", G.name, G.number_of_nodes()

            # loopgraph_unrolled[lpconf['params']['id']] = lpconf
            # parenconf['params']['subgraph'] = lpconf
            loopgraph_unrolled[lpconf['params']['id']] = lpconf
                
            # G.add_node(nc, **lpconf)
            # # print "|G|", G.name, G.number_of_nodes()
            # nc += 1

        # print "        loopgraph_unrolled", loopgraph_unrolled.keys()
        for k, v in list(loopgraph_unrolled.items()):
            logger.debug("loop %s, params = %s", k, list(v['params'].keys()))
        # sys.exit(1)
        # conf['params']['subgraph'] = loopgraph_unrolled
        # conf['params']['graph'] = loopgraph_unrolled
        return loopgraph_unrolled
        
    def step(self, x = None):
        """LoopBlock2.step

        Loop over self.nxgraph items and step

        Disregards blocksize and calls children in every step. Children need to do their own exec timing.

        No inputs.

        No output.
        """
        # print "%s.step[%d], blocksize %d" % (self.cname, self.cnt, self.blocksize, )
        for i in range(self.nxgraph.number_of_nodes()):
            v = self.nxgraph.node[i]
            k = v['params']['id']
            # print "node %d, k = %s" % (i, k)

            v['block_'].step()

        # self.cnt += 1
        self.cnt += self.blocksize_min

class SeqLoopBlock2(Block2):
    """SeqLoopBlock2 is a *sequential* loop block.

    Sequential means dynamic instantiation of 'loopblock' within each
    loop iterations during graph execution. The instantiated block is
    grafted onto the existing top.nxgraph on-the-go.

    Outputs: SeqLoopBlock2's outputs are computed by searching for a
    matching output key in the 'loopblock' and copying its contents if
    it exists. Otherwise a warning is issued and the output is left at
    its previous value.
    """
    def __init__(self, conf = {}, paren = None, top = None):
        self.defaults['loop'] = [1]
        self.defaults['loopmode'] = 'sequential'
        self.defaults['loopblock'] = {}

        # force loopmode set
        self.loopmode = 'sequential'

        Block2.__init__(self, conf = conf, paren = paren, top = top)

        # # aha
        # self.init_primitive()
        
        # initialize block output
        self.init_outputs()
        
        # check 'loop' parameter type and set the loop function
        if type(self.loop) is list: # it's a list
            self.f_loop = self.f_loop_list
            # assert len(self.loop) == (self.numsteps/self.blocksize), "%s step numsteps / blocksize (%s/%s = %s) needs to be equal the loop length (%d)" % (self.cname, self.numsteps, self.blocksize, self.numsteps/self.blocksize, len(self.loop))
        else: # it's a func
            self.f_loop = self.f_loop_func

        # store dynamically constructed graph for visualization / debugging
        self.confgraph = OrderedDict([])
        # self.nxgraph = None

    # loop function for self.loop = list 
    def f_loop_list(self, i, f_obj):
        # print "self.loop", i, self.loop
        results = f_obj(self.loop[i])
        return results

    # loop function for self.loop = func
    def f_loop_func(self, i, f_obj):
        self._debug("f_loop_func i = %d, f_obj = %s" % (i, f_obj))
        # self.loop is the function configured in block config
        results = self.loop(self, i, f_obj)
        return results

    # @decStep()
    def step_cache(self, x = None):
        """Compute step from cached data
        """

        print("%s-%s[%d] this should never print but be caught by step decorator :)" % (self.cname, self.id, self.cnt))
        print("%s-%s[%d] is cached at %s\n    cache = %s" % (self.cname, self.id, self.cnt, self.md5, self.cache))
        if isinstance(self, SeqLoopBlock2): # hasattr(self, 'cache_data') and 
            print("    ready for batch playback of %s" % (list(self.cache_data.keys()),))
        for outk in list(self.outputs.keys()):
            print("cached self.%s = %s" % (outk, getattr(self, outk)))
            pass
    
    # @decStep()
    def step_compute(self, x = None):
        """Compute step

        SeqLoopBlock2's step works like this:
         - if scheduled, iterate i over loop length
         - depending wether 'loop' is a list or a func, the
           corresponding subordinate private function is called for
           each iteration, which instantiates the loopblock in the 'dynblock' variable.
         - outputs are copied from dynblock's outputs if one exists with the same name
        """
        # self._debug("%s.step:\n\tx = %s,\n\tbus = %s,\n\tinputs = %s,\n\toutputs = %s",
        #                      (self.__class__.__name__,self.outputs.keys(),
        #                           self.bus, self.inputs, self.outputs))

        self._debug("step_compute[%d]" % (self.cnt, ))
        
        def f_obj(lparams):
            """instantiate the loopblock and run it
            """
            # print "f_obj lparams", lparams
            # copy params
            loopblock_params = {
                'numsteps': self.numsteps,
            }

            # FIXME: randseed hack
            if lparams[0] == 'randseed':
                # save old state
                randstate = np.random.get_state()
                # reset the seed for this loop
                np.random.seed(lparams[1])
                
            for k, v in list(self.loopblock['params'].items()):
                # print "SeqLoopBlock2.step.f_obj loopblock params", k, v # , lparams[0]

                if k == 'id':
                    loopblock_params[k] = "%s%s%d" % (self.id, loop_delim, i)
                elif k == lparams[0]: # FIXME: multiloop k in lparams.keys()
                    loopblock_params[k] = lparams[1]
                else:
                    loopblock_params[k] = v

            self._debug(
                "step.f_obj: loopblock_params = %s" % (loopblock_params, ))
            
            # create dynamic conf, beware the copy (!!!)
            loopblock_conf = {'block': self.loopblock['block'], 'params': copy.deepcopy(loopblock_params)}
            
            self._debug(
                "step.f_obj: loopblock_conf = %s" % (loopblock_conf, ))
            
            # instantiate block loopblock
            self.dynblock = self.loopblock['block'](
                conf = loopblock_conf,
                paren = self.paren,
                top = self.top)

            loopblock_conf['block_'] = self.dynblock
            
            # second pass
            self.dynblock.init_pass_2()

            # add to dynamic graph conf
            self.confgraph['%s' % loopblock_params['id']] = copy.copy(loopblock_conf)
            
            # this is needed for using SeqLoop as a sequencer / timeline with full sideway time
            # run the block starting from cnt = 1
            for j in range(1, int(self.dynblock.numsteps)+1):
                # print "%s-%s trying %s-%s.step[%d]" % (self.cname, self.id, self.dynblock.cname, self.dynblock.id, j)
                # print self.dynblock.x.shape, self.dynblock.y.shape
                self.dynblock.step()

            # print "%s.step.f_obj did %d %s.steps" % (self.cname, j, self.dynblock.cname)

            # FIXME: randseed hack
            if lparams[0] == 'randseed':
                # save old state
                np.random.set_state(randstate)

            
            # copy looped-block outputs to loop-block outputs
            d = {}
            for outk, outv in list(self.dynblock.outputs.items()):
                d[outk] = getattr(self.dynblock, outk)
            # print "j", j, "d", d
            return d
        
        def f_obj_hpo(params):
            # print "f_obj_hpo: params", params
            x = np.array([params]).T
            # XXX
            lparams = ('inputs', {'x': {'val': x}}) # , x.shape, self.outputs['x']
            self._debug("step[%d] f_obj_hpo lparams = %s" % (self.cnt, lparams, ))
            # normal f_obj, create block, run it, return results
            f_obj(lparams)

            # now we have self.dynblock
            assert hasattr(self, 'dynblock')

            # compute the loss for hpo from dynblock outputs
            loss = 0
            for outk in list(self.outputs.keys()):
                # omit input values / FIXME
                if outk in list(self.dynblock.inputs.keys()): continue
                # print "outk", outk, getattr(dynblock, outk)
                # FIXME: if outk == 'y' as functions result
                loss += np.mean(getattr(self.dynblock, outk), axis = 1, keepdims = True)

            # print "loss", loss
                
            rundata = {
                'loss': loss[0,0], # compute_complexity(Xs)
                'status': STATUS_OK, # compute_complexity(Xs)
                # 'dynblock': self.dynblock,
                'lparams': lparams,
                # "M": M, # n.networks["slow"]["M"], # n.M
                # "timeseries": Xs.copy(),
                # "loss": np.var(Xs),
            }
            return rundata
        # {'loss': , 'status': STATUS_OK, 'dynblock': None, 'lparams': lparams}

        # set loop function wether loop body is list or func
        if type(self.loop) is list:
            self._debug("loop is list")
            f_obj_ = f_obj
        else:
            self._debug("loop is func")
            f_obj_ = f_obj_hpo

        # loop the loop
        then = time.time()
        # print "%s-%s.step[%d]" % (self.cname, self.id, self.cnt)
        # loopblock loop
        for i in range(int(self.numsteps/self.loopblocksize)):
            # sys.stdout.flush()
            then = time.time()

            # run the loop, if it's a func loop: need input function from config
            self._debug("step[%d] loop iter %d, trying to get results, time = %s" % (self.cnt, i, then))
            results = self.f_loop(i, f_obj_)
            # print "results", results
            self._debug("step[%d] f_loop results[%d] = %s" % (self.cnt, i, results, ))

            # FIXME: WORKS for loop example hpo, model sweeps,
            #        BREAKS for real experiment with measure output
            # # copy dict to self attrs
            # if results is not None:
            #     for k, v in results.items():
            #         self._debug("SeqLoopBlock2.step loop %d result k = %s, v = %s", (i, k, v))
            #         setattr(self, k, v)
                    
            # dynblock = results['dynblock']
            # lparams = results['lparams']
            
            # print "%s.steploop[%d], %s, %s" % (self.cname, i, lparams, self.loopblock['params'])
            # for k,v in dynblock.outputs.items():
            #     print "dynout", getattr(dynblock, k)

            # setting SeqLoopBlock2's outputs
            for outk in list(self.outputs.keys()):
                # print "SeqLoopBlock2.step[%d] loop iter %d, outk = %s, dynblock outk = %s" % (self.cnt, i, outk, self.dynblock.outputs.keys(), )
                outvar = getattr(self, outk)
                # print "SeqLoopBlock2.step[%d] loop iter %d, outk = %s, outvar = %s" % (self.cnt, i, outk, outvar, )
                
                # func: need output function from config
                # FIXME: handle loopblock blocksizes greater than one
                # self.__dict__[outk][:,[i]] = np.mean(getattr(dynblock, outk), axis = 1, keepdims = True)
                
                # FIXME: which breaks more?
                # outslice = slice(i*self.dynblock.blocksize, (i+1)*self.dynblock.blocksize)

                # FIXME: which breaks more?
                # assert self.dynblock.outputs.has_key(outk), "Assuming %s-%s.outputs has key %s, but %s" % (self.dynblock.cname, self.dynblock.id, outk, self.dynblock.outputs.keys())
                if outk not in self.dynblock.outputs:
                    self._warning("Output %s not found in %s-%s's outputs %s" % (outk, self.dynblock.cname, self.dynblock.id, list(self.dynblock.outputs.keys())))
                    continue

                # compute slice
                outslice = slice(i*self.dynblock.outputs[outk]['shape'][-1], (i+1)*self.dynblock.outputs[outk]['shape'][-1])

                # debug
                self._debug(
                    "step[%d] setting output %s = %s, outslice = %s" % (
                        self.cnt, outk, getattr(self, outk).shape, outslice,
                    )
                )
                    
                # set the attribute
                outvar[...,outslice] = getattr(self.dynblock, outk).copy()
                
                # print "dynblock-%s outslice = %s, outvar = %s/%s%s, dynblock.out[%s] = %s" %(self.dynblock.id, outslice, outvar.shape, outvar[...,:].shape, outvar[...,outslice].shape, outk, getattr(self.dynblock, outk).shape)
                
        # sys.stdout.write('\n')

        confgraph_full = {'block': Block2, 'params': {'id': self.id, 'graph': self.confgraph}}

        # print "%s-%s.step dynamic graph = %s" % (self.cname, self.id, confgraph_full)
        self.nxgraph = nxgraph_from_smp_graph(confgraph_full)

        for outk in list(self.outputs.keys()):
            logstr = "step[%d/%d] output %s = %s" % (self.top.cnt, self.cnt, outk, getattr(self, outk))
            self._debug(logstr)

        self._debug('step[%d] bus state = %s' % (self.cnt, list(self.bus.keys())))
        # # hack for checking hpo minimum
        # if hasattr(self, 'hp_bests'):
        #     print "%s.step: bests = %s, %s" % (self.cname, self.hp_bests[-1], f_obj_hpo(tuple([self.hp_bests[-1][k] for k in sorted(self.hp_bests[-1])])))
    
class PrimBlock2(Block2):
    """PrimBlock2 class

    Base class for primitive blocks as opposed to compositional ones
    containing graphs themselves.
    """
    defaults = {
        'block_group': 'comp',
    }
    @decInit()
    def __init__(self, conf = {}, paren = None, top = None):
        # merge Block2 base defaults with child defaults
        defaults = {}
        defaults.update(Block2.defaults)
        defaults.update(PrimBlock2.defaults, **self.defaults)
        self.defaults = defaults
        
        Block2.__init__(self, conf = conf, paren = paren, top = top)

    @decStep()
    def step(self, x = None):
        """PrimBlock2.step: step the block, decorated, blocksize boundaries"""
        # print "primblock step id %s, v = %s" % (self.id, self.x)
        pass
    
class IBlock2(PrimBlock2):
    """IBlock2 class

    Integrator block: add input 'k_i' in input.keys to current value
    of output state 'Ik_i'

    Params: inputs ['x'], outputs ['Ix'], leakrate [1.0]

    .. note:: Presence of 'leak' parameter controls choice of step function. No 'leak' param maps to np.cumsum over batch of blocksize / input-blocksize.
    """
    defaults = {
        # 'leak': 0.0,
        'inputs': {'x': {'shape': (1,1), 'val': np.zeros((1,1))}},
        'outputs': {},
        }
        
    @decInit()
    def __init__(self, conf = {}, paren = None, top = None):
        # default out
        if 'outputs' not in conf['params']:
            conf['params']['outputs'] = {}

        # create output states
        for k, v in list(conf['params']['inputs'].items()):
            logger.debug('%s.init inkeys %s', self.__class__.__name__, k)
            logger.debug('IBlock2 conf[\'params\'] keys = %s', list(conf['params'].keys()))
            logger.debug('IBlock2 conf[\'params\'][\'outputs\'] keys = %s' % (list(conf['params']['outputs'].keys()), )) # ["I%s" % k]
            outk = 'I%s' % (k, )
            busk = conf['params']['inputs'][k]
            # logger.debug('IBlock2 outk = %s, busk = %s" % (outk, busk, )
            if outk in conf['params']['outputs']:
                # conf['params']['outputs'][outk] = {'shape': top.bus[busk['bus']].shape} # ['val'].shape]}
                pass
            else:
                conf['params']['outputs'][outk] = {'shape': v['shape']} # {'shape': top.bus[busk['bus']].shape} # ['val'].shape]}
            setattr(self, outk, np.zeros(v['shape']))
            
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)

        # logger.debug('IBlock2 outputs", self.outputs
        
        if hasattr(self, 'leak'):# and self.leak > 0.0:
            self.step = self.step_leak
        else:
            self.step = self.step_all

    @decStep()
    def step_leak(self, x = None):
        # loop over blocksize
        for i in range(self.blocksize):
            # do it for all inputs
            for ink in list(self.inputs.keys()):
                # get input
                inv = self.get_input(ink)#[...,[i]]
                self._debug('input %s current %s' % (ink, inv))
                # with corresponding output
                outk = "I%s" % ink
                # get last state
                tmp_ = getattr(self, outk).copy()
                self._debug('input %s last int state %s' % (ink, tmp_))
                # integrate with leak
                tmp_[...,[i]] = ((1 - self.leak) * tmp_[...,[i-1]]) + (inv * self.leak)
                #  store state
                setattr(self, outk, tmp_)
        self._debug('self.%s = %s' % (outk, getattr(self, outk)))

    @decStep()
    def step_all(self, x = None):
        for ink, inv in list(self.inputs.items()):
            outk = 'I%s' % ink
            # input integral / cumsum
            Iin = np.cumsum(inv['val'], axis = 1) # * self.d
            # print getattr(self, outk)[:,[-1]].shape, self.inputs[ink][0].shape, Iin.shape
            # single step
            # setattr(self, outk, getattr(self, outk)[:,[-1]] + Iin)
            # setattr(self, outk, getattr(self, outk) + (self.inputs[ink][0] * 1.0))
            # multi step / batch
            setattr(self, outk, Iin)
            logger.debug('IBlock2.step[%d] self.%s = %s / %s' % (self.cnt, outk, getattr(self, outk).shape, self.outputs[outk]['shape']))

class dBlock2(PrimBlock2):
    """dBlock2 class

    Differentiator block: compute differences of input and write to output

    Params: inputs, outputs, leakrate / smoothrate?
    """
    defaults = {
        'd': 1.0,
    }
    @decInit()
    def __init__(self, conf = {}, paren = None, top = None):
        """dBlock2 init"""
        if 'outputs' not in conf['params']:
            conf['params']['outputs'] = {}
            
        for ink in list(conf['params']['inputs'].keys()):
            # get input shape
            buskey = conf['params']['inputs'][ink]['bus']
            if buskey in top.bus:
                inshape = top.bus[buskey].shape
            else:
                inshape = conf['params']['inputs'][ink]['shape']
                
            # inshape = conf['params']['inputs'][ink]['shape']
            # print "inshape", inshape
            # alloc copy of previous input block 
            setattr(self, "%s_" % ink, np.zeros(inshape))
            # set output members
            conf['params']['outputs']["d%s" % ink] = {'shape': inshape}
            
        # base block init
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)

    @decStep()
    def step(self, x = None):
        """dBlock2 step"""
        for ink in list(self.inputs.keys()):
            # output key
            outk = "d%s" % ink
            # input from last block
            ink_ = "%s_" % ink
            inv_ = getattr(self, ink_)
            # stack last and current block input
            tmp_   = np.hstack((inv_, self.inputs[ink]['val']))
            # slice -(blocksize + 1) until now
            tmp_sl = slice(self.blocksize - 1, self.blocksize * 2)
            # compute the diff in the input
            din = np.diff(tmp_[:,tmp_sl], axis = 1) * self.d
            # which should be same shape is input
            assert din.shape == self.inputs[ink]['val'].shape
            self._debug("step[%d] id = %s, outk %s = %s, ink %s = %s, din = %s" % (
                self.cnt, self.id,
                outk, getattr(self, outk)[:,[-1]].shape,
                ink, self.inputs[ink]['val'].shape,
                din.shape)
            )
            setattr(self, outk, din)
            # store current input
            setattr(self, ink_, self.inputs[ink]['val'].copy())
            # print "dBlock2.step", self.id, self.cnt, getattr(self, outk), din

class DelayBlock2(PrimBlock2):
    """DelayBlock2 class

    Delay block: delay input 'input-key' by delays 'input-key' with an
    internal ringbuffer delay line.

    Params: inputs, delay in steps / shift

    FIXME: pull and sift existing embedding code: smp/sequence, pointmasslearner/reservoir, smp/neural, mdp's TimeDelaySlidingWindowNode
    FIXME: consolidate / merge with tappings, im2col, conv, ...
    """
    defaults = {
        'flat': False, # flatten output - [2, 5, 2000] 2 x 10000
        'flat2': False, # flatten output differently - [2, 5, 2000] 10 x 2000
        'full': False, # full contiguous output up to delay_max
        'outputs': {},
    }
    @decInit() # outputs from inputs block decorator
    def __init__(self, conf = {}, paren = None, top = None):
        """DelayBlock2 init"""
        params = conf['params']

        for dk, dv in list(self.defaults.items()):
            if dk not in params:
                params[dk] = dv

        delays_ = {}

        # loop over input items
        for ink, inv in list(params['inputs'].items()):
            # get input shape
            # assert top.bus.has_key(params['inputs'][ink]['bus']), "DelayBlock2 needs existing bus item at %s to infer delay shape" % (params['inputs'][ink]['bus'], )
            if inv['bus'] in top.bus:
                inshape = top.bus[inv['bus']].shape
            else:
                inshape = inv['shape']
                
            # alloc delay block, checking for different configuration options
            if 'delays' in params: 
                # assert params['inputs'].keys() == params['delays'].keys()
                delays_[ink] = params['delays'][ink]
                # setattr(self, "%s_" % ink, np.zeros((inshape[0], inshape[1] + params['delays'][ink])))
            else:
                delays_[ink] = inv['delay']
                # setattr(self, "%s_" % ink, np.zeros((inshape[0], inshape[1] + inv['delay'])))
                
            # fix lazy one-element lists
            if type(delays_[ink]) is not list:
                delays_[ink] = [delays_[ink]]
                
            # get max delay
            delay_max = np.max(delays_[ink])
            delay_num = len(delays_[ink])

            # set output members
            # params['outputs']["d%s" % ink] = {'shape': inshape}
            # params['outputs']["d%s" % ink] = {'shape': getattr(self, '%s_' % ink).shape}
            # FIXME: output modifiers: struct/flat, dense/sparse, ...
            if params['flat']:
                # outshape = (inshape[0], inshape[1] * delay_num )
                outshape = (inshape[0] * delay_num, inshape[1])
            else:
                outshape = (inshape[0], delay_num, inshape[1] )
            params['outputs']["d%s" % ink] = {'shape': outshape}
            bufshape = tuple2inttuple((inshape[0], inshape[1] + (delay_max + 0) ))
            setattr(self, "%s_buf" % ink, np.zeros(bufshape))

        # rename to canonical
        params['delays'] = delays_

        # params['has_stepped'] = False
        
        # base block init
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)

        # init the delay lines
        self.delaytaps = {}
        for k, v in list(self.delays.items()):
            # print "Adding delayed input %s / %s to delaytaps" % (k, v)
            # delay_tap = -np.array(v) - 1
            delay_tap = -np.array(v) - 0
            assert 'shape' in self.inputs[k], "%s-%s requires input 'shape' attribute' for input %s with attributes %s" % (self.__class__.__name__, self.id, k, list(self.inputs[k].keys()))
            blocksize_input = int(self.inputs[k]['shape'][-1])
            # blocksize_input = self.blocksize
            delay_tap_bs = (delay_tap - np.tile(np.array(list(range(blocksize_input, 0, -1))), (delay_tap.shape[0],1)).T).T
            if self.flat or self.flat2:
                delay_tap_bs = delay_tap_bs.flatten()
            self.delaytaps[k] = delay_tap_bs.copy()
            self._debug('delay %s with taps %s added delaytaps_bs = %s' % (k, v, self.delaytaps[k]))
        
        # print "DelayBlock2.init blocksize", self.blocksize
        # print "DelayBlock2.init delays", self.delays
        # print "DelayBlock2.init delays", self.delaytaps
        
    @decStep()
    def step(self, x = None):
        """DelayBlock2 step"""
        # loop over input items
        for ink in list(self.inputs.keys()):
            # blocksize vs. input blocksize
            blocksize_input = self.inputs[ink]['shape'][-1]
            # multichannel delay hack for different blocksizes
            if self.cnt % blocksize_input not in self.blockphase:
                self._debug('input %s[%d] deferred for shape end = %s overriding blocksize = %s' % (ink, self.cnt, blocksize_input, self.blocksize))
                continue
            
            # get output key
            outk = "d%s" % ink
            # get buffer key
            ink_ = "%s_buf" % ink
            # stack last and current block input
            inv_ = getattr(self, ink_)

            # print "outk", outk,"ink_", ink_, "inv_", inv_.shape
            # # copy current input into beginning of delay line inv_
            # # sl = slice()
            # inv_[...,-self.blocksize:] = self.inputs[ink]['val'].copy()

            # # copy delayed input into output
            # delay_tap = -np.array(self.delays[ink]) - 1
            # print "delay_tap", delay_tap
            # setattr(self, outk, inv_[...,delay_tap])
            
            # sl = slice(self.delays[ink], self.delays[ink]+self.blocksize)
            
            # write blocksize most current input data into buffer
            sl = slice(-self.blocksize, None)
            if blocksize_input != self.blocksize:
                sl = slice(-blocksize_input, None)
                self._debug('input %s shape end = %s, blocksize = %s' % (ink, self.inputs[ink]['shape'][-1], self.blocksize))
                
            self._debug('input %s slice sl = %s' % (ink, sl))

            inv_[...,sl] = self.inputs[ink]['val'].copy()
            # if block executes only once
            # if input blocksize == numsteps
            if blocksize_input == self.top.numsteps:
                self._debug('input %s bsi > numsteps, slice sl.start = %s, sl.stop = %s, inv_.shape = %s' % (ink, sl.start, sl.stop, inv_.shape))
                # cyclic wrap of input chunk
                inv_[...,slice(None, sl.start)] = inv_[...,slice(-sl.start, None)]
            
            # perform delay immediately after input
            setattr(self, ink_, np.roll(inv_, shift = -self.blocksize, axis = -1))
            
            # print "DelayBlock2: ink", ink, "sl", sl, "inv_", inv_.shape, "input", self.inputs[ink]['val'].shape
            # print "DelayBlock2: ink", ink, inv_[...,sl].shape

            # outputs
            delaytap = self.delaytaps[ink]
            # print "delaytap", delaytap.shape, delaytap, self.blocksize
            self._debug('input %s delaytap = %s' % (ink, delaytap))            
            if self.flat:
                setattr(self, outk, inv_[...,delaytap])
            elif self.flat2:
                setattr(self, outk, inv_[...,delaytap].reshape((-1, self.blocksize)))
            else:
                setattr(self, outk, inv_[...,delaytap])
                
            self._debug('output %s = %s' % (outk, getattr(self, outk).shape))
            
            # setattr(self, outk, inv_[...,slice(0, self.blocksize)])

            # self._debug("DelayBlock2 outk %s shape" %(outk,), self.inputs[ink]['val'].shape, getattr(self, ink_).shape, getattr(self, outk) # [...,[-1]].shape, self.inputs[ink]['val'].shape #, din.shape
            
            # # delay current input for blocksize steps
            # setattr(self, ink_, np.roll(inv_, shift = -self.blocksize, axis = -1))
            
            self._debug('self.%s = %s' % (ink_, getattr(self, ink_)))
            self._debug('self.%s = %s' % (outk, getattr(self, outk)))
                        
class SliceBlock2(PrimBlock2):
    """SliceBlock2

    Cut slices from the input tensor

    FIXME: make slicing a general function of block i/o
    FIXME: generic ndim slicing?
    """
    def __init__(self, conf = {}, paren = None, top = None):
        params = conf['params']
        if 'outputs' not in params:
            params['outputs'] = {}
            
        for k, v in list(params['inputs'].items()):
            slicespec = params['slices'][k]
            assert 'shape' in v, 'SliceBlock2 needs input shape spec at configuration time but only has %s' % (list(v.keys()),)
            # print slicespec
            for slk, slv in list(slicespec.items()):
                logger.debug('init ink = %s, slicekey = %s, sliceval = %s', k, slk, slv)
                # really use the specified output shape, not execution blocksize
                oblocksize = v['shape'][-1]
                outk = "%s_%s" % (k, slk)
                if type(slv) is slice:
                    params['outputs'][outk] = {'shape': (slv.stop - slv.start, oblocksize)} # top.bus[params['inputs'][k][0]].shape
                elif type(slv) is list:
                    params['outputs'][outk] = {'shape': (len(slv), oblocksize)} # top.bus[params['inputs'][k][0]].shape
                elif type(slv) is tuple:
                    params['outputs'][outk] = {'shape': (slv[1] - slv[0], oblocksize)} # top.bus[params['inputs'][k][0]].shape
            
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)

    @decStep()
    def step(self, x = None):
        for ink in list(self.inputs.keys()):
            self._debug('step[%d] ink = %s, inv = %s' % (
                self.cnt, ink, self.inputs[ink]))
                
            slicespec = self.slices[ink]
            for slk, slv in list(slicespec.items()):
                outk = "%s_%s" % (ink, slk)
                setattr(self, outk, self.inputs[ink]['val'][slv])
                self._debug('step[%d] outk = %s, outsh = %s, out = %s' % (self.cnt, outk, getattr(self, outk).shape, getattr(self, outk)))

class StackBlock2(PrimBlock2):
    """StackBlock2 class

    Stack block can combine input slices into a single output item
    """
    def __init__(self, conf = {}, paren = None, top = None):
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)

    @decStep()
    def step(self, x = None):
        st = [inv['val'] for ink, inv in list(self.inputs.items())]
        # print "Stack st = %s" % ( len(st))
        self.y = np.vstack(st)
                    
class ConstBlock2(PrimBlock2):
    """ConstBlock2 class

    Constant block: output is a constant vector
    """
    defaults = {
        'block_group': 'data',
        }
    def __init__(self, conf = {}, paren = None, top = None):
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)
        
        # either column vector to be replicated or blocksize already
        # assert self.x.shape[-1] in [1, self.blocksize]
        
        # print"self.inputs['c']['val'].shape[:-1] == self.x.shape[:-1]", self.inputs['c']['val'].shape[:-1], self.x.shape[:-1]
        # assert self.inputs['c']['val'].shape[:-1] == self.x.shape[:-1], "ConstBlock2 input / output shapes must agree: %s == %s?" % (self.inputs['c']['val'].shape[:-1], self.x.shape[:-1])
        
        # replicate column vector
        # if self.x.shape[1] == 1: # this was wrong
        if self.inputs['c']['val'].shape[1] == 1:
            self.x = np.tile(self.inputs['c']['val'], self.blocksize) # FIXME as that good? only works for single column vector
        else:
            self.x = self.inputs['c']['val'].copy() # FIXME as that good? only works for single column vector

class CountBlock2(PrimBlock2):
    """CountBlock2 class

    Count block, output is a counter value updated every blocksize
    steps by incr
    """
    defaults = {
        # 'cnt': np.ones((1,1)),
        'outputs': {
            'x': {'shape': (1,1)}
            },
        'block_group': 'data',
        }
        
    @decInit()
    def __init__(self, conf = {}, paren = None, top = None):
        # defaults
        self.scale  = 1
        self.offset = 0
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)
        # single output key
        self.outk = list(self.outputs.keys())[0]
        # init cnt_ of blocksize
        # self.cnt_ = np.zeros(self.outputs[self.outk]['shape'] + (self.blocksize,))
        # self.cnt = None # ???
        self.cnt_ = np.zeros(self.outputs[self.outk]['shape'])
        # print self.inputs
        # FIXME: modulo / cout range with reset/overflow

        # print "\n%s endofinit bus = %s\n" % (self.cname, self.bus.keys())
    @decStep()
    def step(self, x = None):
        """CountBlock step: if blocksize is 1 just copy the counter, if bs > 1 set cnt_ to range"""
        outshape = self.outputs[self.outk]['shape']
        outshapenum = outshape[-1]
        # if self.blocksize > 1:
        if outshapenum > 1:
            newcnt = np.tile(np.arange(self.cnt - outshapenum, self.cnt), outshape[:-1]).reshape(outshape)
            # self.cnt_[...,-outshapenum:] = newcnt
            self.cnt_ = newcnt
        else:
            self.cnt_[...,0] = self.cnt
        # FIXME: make that a for output items loop
        # print "getattr self.outk", self.outk, getattr(self, self.outk), self.cnt_
        setattr(self, self.outk, (self.cnt_ * self.scale) + self.offset)

class TrigBlock2(PrimBlock2):
    """TrigBlock2 class

    Trig block, output is a triger value updated every blocksize
    steps by incr
    """
    defaults = {
        # 'cnt': np.ones((1,1)),
        'debug': False,
        'outputs': {
            # 'x': {'shape': (1,1)}
            },
        'block_group': 'data',
    }
        
    @decInit()
    def __init__(self, conf = {}, paren = None, top = None):
        # defaults
        # self.scale  = 1
        # self.offset = 0
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)
        
        # single output key
        # self.outk = self.outputs.keys()[0]
        # init cnt_ of blocksize
        # self.cnt_ = np.zeros(self.outputs[self.outk]['shape'] + (self.blocksize,))
        # self.cnt = None # ???
        # self.cnt_ = np.zeros(self.outputs[self.outk]['shape'])
        # print self.inputs
        # FIXME: modulo / cout range with reset/overflow
        # print "\n%s endofinit bus = %s\n" % (self.cname, self.bus.keys())
        
    @decStep()
    def step(self, x = None):
        """TrigBlock step: if blocksize is 1 just copy the triger, if bs > 1 set cnt_ to range"""
        # outshape = self.outputs[self.outk]['shape']
        # outshapenum = outshape[-1]
        # # if self.blocksize > 1:
        # if outshapenum > 1:
        #     newcnt = np.tile(np.arange(self.cnt - outshapenum, self.cnt), outshape[:-1]).reshape(outshape)
        #     # self.cnt_[...,-outshapenum:] = newcnt
        #     self.cnt_ = newcnt
        # else:
        #     self.cnt_[...,0] = self.cnt
        # FIXME: make that a for output items loop
        # print "getattr self.outk", self.outk, getattr(self, self.outk), self.cnt_
        for outk, outv in list(self.outputs.items()):
            if self.cnt in self.trig:
                setattr(self, outk, np.ones(outv['shape']))
                # self._debug("    triggered %s = %s" % (outk, getattr(self, outk),))
            else:
                setattr(self, outk, np.zeros(outv['shape']))
                # self._debug("not triggered %s = %s" % (outk, getattr(self, outk),))
        
class ThreshBlock2(PrimBlock2):
    """ThreshBlock2 class

    Outputs 1 when threshold 'thr' is exceeded, 0 otherwise.
    """
    defaults = {
        'debug': False,
        'outputs': {
        },
        'block_group': 'data',
    }
        
    @decInit()
    def __init__(self, conf = {}, paren = None, top = None):
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)
                
    @decStep()
    def step(self, x = None):
        """ThreshBlock step: if blocksize is 1 just copy the trigger, if bs > 1 set cnt_ to range"""
        # outshape = self.outputs[self.outk]['shape']
        # outshapenum = outshape[-1]
        # # if self.blocksize > 1:
        # if outshapenum > 1:
        #     newcnt = np.tile(np.arange(self.cnt - outshapenum, self.cnt), outshape[:-1]).reshape(outshape)
        #     # self.cnt_[...,-outshapenum:] = newcnt
        #     self.cnt_ = newcnt
        # else:
        #     self.cnt_[...,0] = self.cnt
        # FIXME: make that a for output items loop
        # print "getattr self.outk", self.outk, getattr(self, self.outk), self.cnt_

        thr = self.get_input('thr')
        x = self.get_input('x')
        outk = 'trig'
        outv = self.outputs[outk]
        
        # if np.any(x > thr):
        if x[0,0] > thr[0,0]:
            setattr(self, outk, np.ones(outv['shape']))
        else:
            setattr(self, outk, np.zeros(outv['shape']))
        # self._debug("not triggered %s = %s" % (outk, getattr(self, outk),))
                
class RouteBlock2(PrimBlock2):
    """RouteBlock2 class

    Route block selects one of its inputs to route to the output.

    FIXME: RoutingMatrix?
    """
    defaults = {
        # 'cnt': np.ones((1,1)),
        'debug': False,
        'outputs': {
            # 'x': {'shape': (1,1)}
        },
        'block_group': 'data',
    }
        
    @decInit()
    def __init__(self, conf = {}, paren = None, top = None):
        conf['params']['inputkeys'] = copy.copy(list(conf['params']['inputs'].keys()))
        conf['params']['inputkeys'].pop(0)
        conf['params']['inputkey'] = 0

        ink = conf['params']['inputkeys'][0]
        conf['params']['outputs'] = {'y': {'shape': conf['params']['inputs'][ink]['shape']}}
        
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)
        self._debug('self.inputkeys = %s' % (self.inputkeys))
                
    @decStep()
    def step(self, x = None):
        """RouteBlock step"""
        
        if not self.block_is_scheduled(): return
        self.inputkey = int(self.get_input('r')[0,0])
        setattr(self, 'y', self.get_input(self.inputkeys[self.inputkey]))
        
class UniformRandomBlock2(PrimBlock2):
    """UniformRandomBlock2 class

    Generate uniform random numbers, output is a vector random sample
    from uniform distribution.
    """
    defaults = {
        'block_group': 'data',
    }
    @decInit()
    def __init__(self, conf = {}, paren = None, top = None):
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)
        
        # self.lo = 0
        # self.hi = 1
        # print "UniformRandomBlock2 keys", self.__dict__.keys()
        # print "out x", self.outputs['x']['shape']
        # print "out x", self.outputs['x']
        for k, v in list(self.outputs.items()):
            x_ = np.random.uniform(-1e-1, 1e-1, v['shape'])
            if hasattr(self, 'lo') and hasattr(self, 'hi'):
                x_ = np.random.uniform(self.lo, self.hi, v['shape'])
            setattr(self, k, x_.copy())
        # self.x = np.random.uniform(
        #     self.inputs['lo']['val'], self.inputs['hi']['val'],
        #     self.outputs['x']['shape'])
        
    @decStep()
    def step(self, x = None):
        self._debug("step:\n\tx = %s,\n\tinputs = %s,\n\toutputs = %s" % (list(self.outputs.keys()), self.inputs, self.outputs))

        # FIXME: relation rate / blocksize, remember cnt from last step, check difference > rate etc
        if self.cnt % self.rate == 0:
            # FIXME: take care of rate/blocksize issue
            for k, v in list(self.outputs.items()):
                # x = np.random.uniform(self.inputs['lo'][0][:,[-1]], self.inputs['hi'][0][:,[-1]], (self.outputs[k][0]))
                # print 'lo', self.inputs['lo']['val'], '\nhi', self.inputs['hi']['val'], '\noutput', v['bshape']
                x = np.random.uniform(self.inputs['lo']['val'], self.inputs['hi']['val'], size = v['shape'])
                setattr(self, k, x)
                self._debug("step self.%s = %s" % (k, x))
        
        # # loop over outputs dict and copy them to a slot in the bus
        # for k, v in self.outputs.items():
        #     buskey = "%s/%s" % (self.id, k)
        #     self.bus[buskey] = getattr(self, k)
        # self.bus[self.id] = self.x
        return None
