"""block.py: Basic computation blocks and support structure for smp_graphs

.. moduleauthor:: 2017 Oswald Berthold

The Base class is :class:`Block2` supported by decorators
:class:`decInit` and :class:`decStep` and the :class:`Bus` class. 

The set of standard blocks includes :class:`FuncBlock2`,
:class:`LoopBlock2`, :class:`SeqLoopBlock2`, :class:`PrimBlock2`,
:class:`IBlock2`, :class:`dBlock2`, :class:`DelayBlock2`,
:class:`SliceBlock2`, :class:`StackBlock2`, :class:`ConstBlock2`,
:class:`CountBlock2`, :class:`UniformRandomBlock2`
"""

import pdb
import uuid, sys, time, copy, re
from collections import OrderedDict, MutableMapping
import itertools
from functools import partial

# import lshash

import networkx as nx

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from smp_base.plot import plot_colors, makefig, set_interactive

import smp_graphs.logging as log
from smp_graphs.utils import print_dict, xproduct, myt

from smp_graphs.common import conf_header, conf_footer, get_input
from smp_graphs.common import md5, get_config_raw, get_config_raw_from_string

from smp_graphs.common import set_attr_from_dict
from smp_graphs.common import dict_get_nodekeys_recursive, dict_replace_nodekeys_loop
from smp_graphs.common import dict_search_recursive, dict_replace_idstr_recursive2

from smp_graphs.graph import nxgraph_from_smp_graph, nxgraph_to_smp_graph
from smp_graphs.graph import nxgraph_node_by_id, nxgraph_node_by_id_recursive

# finally, ros
import rospy
from std_msgs.msg import Float64MultiArray

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

################################################################################
# utils, TODO move to utils.py
def ordereddict_insert(ordereddict = None, insertionpoint = None, itemstoadd = []):
    """ordereddict_insert

    Self rolled ordered dict insertion from http://stackoverflow.com/questions/29250479/insert-into-ordereddict-behind-key-foo-inplace
    """
    assert ordereddict is not None
    assert insertionpoint in ordereddict.keys(), "insp = %s, keys = %s, itemstoadd = %s" % (insertionpoint, ordereddict.keys(), itemstoadd)
    new_ordered_dict = ordereddict.__class__()
    for key, value in ordereddict.items():
        new_ordered_dict[key] = value
        if key == insertionpoint:
            # check if itemstoadd is list or dict
            if type(itemstoadd) is list:
                for item in itemstoadd:
                    keytoadd, valuetoadd = item
                    new_ordered_dict[keytoadd] = valuetoadd
            else:
                for keytoadd, valuetoadd in itemstoadd.items():
                    new_ordered_dict[keytoadd] = valuetoadd
        # else:
        #     print "insertionpoint %s doesn't exist in dict" % (insertionpoint)
        #     sys.exit(1)
    ordereddict.clear()
    ordereddict.update(new_ordered_dict)
    return ordereddict

################################################################################
# smp_graphs types: create some types for use in configurations like const, bus, generator, func, ...

################################################################################
# bus class
# from http://stackoverflow.com/questions/3387691/how-to-perfectly-override-a-dict
class Bus(MutableMapping):
    """Bus class

    A dictionary that applies an arbitrary key-altering function before accessing the keys
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
        return self.store.has_key(key)

    # custom methods
    def setval(self, k, v):
        self.store[k] = v

    def plot(self, ax = None, blockid = None):
        """Bus.plot

        Plot the bus for documentation and debugging.
        """
        assert ax is not None
        xspacing = 10
        yspacing = 2
        yscaling = 0.66
        
        xmax = 0
        ymax = 0

        ypos = -yspacing
        xpos = 0

        if blockid is None: blockid = "Block2"
            
        ax.set_title(blockid + ".bus")
        ax.text(10, 0, "Bus (%s)" % ("topblock"), fontsize = 10)
        ax.grid(0)
        # ax.plot(np.random.uniform(-5, 5, 100), "ko", alpha = 0.1)
        i = 0
        for k, v in self.store.items():
            # print "k = %s, v = %s" % (k, v)
            # ypos = -10 # -(i+1)*yspacing
            # xpos = (i+1)*xspacing
            # ypos += -yspacing
            xpos +=  xspacing
            if len(k) > 8:
                xspacing = len(k) + 2
            else:
                xspacing = 10
                
            ax.text(xpos, ypos, "{0: <8}\n{1: <12}".format(k, v.shape), family = 'monospace', fontsize = 8)
            # elementary shape without buffersize
            ax.add_patch(
                patches.Rectangle(
                    # (30, ypos - (v.shape[0]/2.0) - (yspacing / 3.0)),   # (x,y)
                    (xpos+2, ypos-1),   # (x,y)
                    v.shape[0],          # width
                    -1 * yscaling,          # height
                    fill = False,
                    # hatch = "|",
                    hatch = "-",
                )
            )
            
            # full blockshape
            bs_height = -np.log10(v.shape[1] * yscaling)
            ax.add_patch(
                patches.Rectangle(
                    # (30, ypos - (v.shape[0]/2.0) - (yspacing / 3.0)),   # (x,y)
                    (xpos+2, ypos-2),   # (x,y)
                    v.shape[0],          # width
                    bs_height,          # height
                    fill = False,
                    # hatch = "|",
                    hatch = "-",
                )
            )

            if xpos > xmax: xmax = xpos
            if -(ypos - 8 - bs_height) > ymax: ymax = -(ypos - 8 - bs_height) + 2
            i+=1
            
        # ax.set_xlim((0, 100))
        # ax.set_ylim((-100, 0))
        ax.set_xlim((0, xmax + xspacing))
        ax.set_ylim((-ymax, 0))

        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        
        plt.draw()
        plt.pause(1e-6)

def get_blocksize_input(G, buskey):
    """block.py.get_blocksize_input

    Get the blocksize of input element from the bus at 'buskey'
    """
    # print "G", G.nodes(), "buskey", buskey
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

            # # print "decInit", kwargs['conf']
            # if not kwargs['conf'].has_key('inputs'):
            #     kwargs['conf']['inputs'] = {}
            # if not kwargs['conf']['inputs'].has_key('blk_mode'):
            #     kwargs['conf']['inputs']['blk_mode'] = 1.0 # enabled
                
            f(xself, *args, **kwargs)

            # print "decInit", xself.id, xself.inputs.keys()
        return wrap

################################################################################
# Block decorator step
class decStep():
    """decStep Block2 step decorator class

    Wrap around Block2.step to perform tasks common to all Block2's.
    """

    def process_input(self, xself):
        sname  = self.__class__.__name__
        esname = xself.cname
        esid   = xself.id
        escnt  = xself.cnt
        
        # loop over block's inputs
        for k, v in xself.inputs.items():
            # print "process_input: ", k, xself.id, xself.cnt, v['val'].shape, v['shape']
            # check input sanity
            assert v['val'].shape == v['shape'], "real and desired input shapes need to agree for block %s, ink = %s, %s != %s" % (xself.id, k, v['val'].shape, v['shape'])
            
            # copy bus inputs to input buffer
            if v.has_key('bus'): # input item is driven by external signal (bus value)
                # exec   blocksize of the input's source node
                # FIXME: search once and store, recursively over nxgraph and subgraphs
                blocksize_input     = get_blocksize_input(xself.top.nxgraph, v['bus'])
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
                        except Exception, e:
                            print "%s-%s[%d].decStep input copy k = %s from bus %s/%s to %s/%s, %s" % (xself.cname, xself.id, xself.cnt, k, v['bus'], xself.bus[v['bus']].shape, v['shape'], v['val'].shape, e)
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
            if k in xself.outputs.keys():
                setattr(xself, k, v['val'].copy()) # to copy or not to copy
                
                # # debug in to out copy
                # print "%s.%s[%d]  self.%s = %s" % (esname, sname, escnt, k, esk)
                # print "%s.%s[%d] outkeys = %s" % (esname, sname, escnt, xself.outputs.keys())

    def process_blk_mode(self, xself):
        if hasattr(xself, 'inputs') and xself.inputs.has_key('blk_mode'):
            # print "blk_mode", xself.id, np.sum(xself.inputs['blk_mode']['val']) # xself.inputs['blk_mode']['val'], xself.inputs['blk_mode']['val'] < 0.1
            if np.sum(xself.inputs['blk_mode']['val']) < 0.1:
                # print "blub"
                xself.cnt += 1
                return True
        return False
            
    def __call__(self, f):
        def wrap(xself, *args, **kwargs):
            # print "xself", xself.__dict__.keys()
            # print xself.id, xself.inputs
            # if not xself.topblock and hasattr(xself, 'inputs') and xself.inputs['blk_mode'] == 0.0:
            self.process_input(xself)

            if self.process_blk_mode(xself): return None
            
            # call the function on blocksize boundaries
            # FIXME: might not be the best idea to control that on the wrapper level as some
            #        blocks might need to be called every step nonetheless?
            # if (xself.cnt % xself.blocksize) == 0: # or (xself.cnt % xself.rate) == 0:
            # print "xself.cnt", xself.cnt, "blocksize", xself.blocksize, "blockphase", xself.blockphase
            if (xself.cnt % xself.blocksize) in xself.blockphase: # or (xself.cnt % xself.rate) == 0:
                # if count aligns with block's execution blocksize

                if xself.top.cached and hasattr(xself, 'isprimitive') and xself.isprimitive and xself.cache is not None and xself.cache.shape[0] != 0:
                    # pass
                    # print xself.cache_data['x'].shape
                    for outk, outv in xself.outputs.items():
                        setattr(xself, outk, xself.cache_data[outk][xself.cnt-xself.blocksize:xself.cnt,...].T)
                        # print "%s-%s" % (xself.cname, xself.id), "outk", outk, getattr(xself, outk).T
                        # print "outk", outk, xself.cache_data[outk] # [xself.cnt-xself.blocksize:xself.cnt]
                    f_out = None
                else:
                    # compute the block with step()
                    f_out = f(xself, None)

                # copy output to bus
                for k, v in xself.outputs.items():
                    buskey = "%s/%s" % (xself.id, k)
                    # print "copy[%d] %s.outputs[%s] = %s / %s to bus[%s], bs = %d" % (xself.cnt, xself.id, k, getattr(xself, k), getattr(xself, k).shape, buskey, xself.blocksize)
                    assert xself.bus[v['buskey']].shape == v['shape'], "real and desired output shapes need to agree block %s, outk = %s, %s != %s" % (xself.id, k, xself.bus[v['buskey']].shape, v['shape'])
                    # copy data onto bus
                    xself.bus[v['buskey']] = getattr(xself, k).copy()
                    # print "xself.bus[v['buskey'] = %s]" % (v['buskey'], ) , xself.bus[v['buskey']]
                    
                    # do logging
                    if xself.logging:
                        # try:
                        log.log_pd(tbl_name = v['buskey'], data = xself.bus[v['buskey']])
                        # except:
                        # print "Logging failed"

                    if hasattr(xself, 'ros') and xself.ros:
                        theattr = getattr(xself, k).flatten().tolist()
                        # print "theattr", k, v, theattr
                        xself.msgs[k].data = theattr
                        xself.pubs[k].publish(xself.msgs[k])
            else:
                f_out = None

            # print "Block2.step[%d] not topblock" % (xself.cnt,)
            if xself.cnt == xself.top.numsteps and hasattr(xself, 'isprimitive') and xself.isprimitive:
                # print "Block2.step end of episode and primitive"
                
                def update_block_store(block = None, top = None):
                    import datetime
                    assert block is not None, "Need some block to work on"
                    # if not block.isprimitive: return

                    # m = md5(str(block.conf))
                    print "update_block_store block = %s (%s)" % (block, block.md5)

                    print "block_store", block.top.block_store.keys()
                    
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

                # update_block_store(block = xself)
                
            # count calls
            xself.cnt += 1 # should be min_blocksize
            # xself.ibufidx = xself.cnt % xself.ibufsize
            
            return f_out
        # return the new func
        return wrap

################################################################################
# Base block class
class Block2(object):
    """Block2 class

    Block base class
    Arguments
    - conf: Block configuration dictionary
    - paren: ref to Block's parent
    - top: ref to top-level node in nested graphs
    - blockid: override block's id assignment
    """
    
    defaults = {
        'id': None,
        'debug': False,
        'topblock': False,
        'ibuf': 1, # input  buffer size
        'obuf': 1, # output buffer size
        'cnt': 1,
        'blocksize': 1, # period of computation calls in time steps
        'blockphase': [0], # list of positions of comp calls along the counter in time steps
        'inputs': {}, # internal port, scalar / vector/ bus key, [slice]
        'outputs': {}, # name, dim
        'logging': True, # normal logging
        'rate': 1, # execution rate rel. to cnt
        'ros': False, # no ROS yet
        'phase': [0],
        'subgraph_rewrite_id': True, #
        'inputs_clamp': False,
        'block_group': 'graph',
    }

    @decInit()
    def __init__(self, conf = {}, paren = None, top = None, blockid = None, conf_localvars = None):
        # general stuff
        self.conf = conf
        self.paren = paren
        self.top = top
        self.cname = self.__class__.__name__
        self.conf_localvars = conf_localvars

        # merge Block2 base defaults with child defaults
        defaults = {}
        defaults.update(Block2.defaults, **self.defaults)
        
        # load defaults
        # set_attr_from_dict(self, self.defaults)
        set_attr_from_dict(self, copy.copy(defaults))
                    
        # fetch existing configuration arguments
        if type(self.conf) == dict and self.conf.has_key('params'):
            # print "Block2 init params", self.conf['params']
            params = copy.deepcopy(self.conf['params'])
            set_attr_from_dict(self, params)
        else:
            print "What could it be? Look at %s" % (self.conf)

        # FIXME: no changes to conf['params'] after this?
            
        # check id
        assert hasattr(self, 'id'), "Block2 init: id needs to be configured"
        # FIXME: check unique id, self.id not in self.topblock.ids
        # print "%s-%s.defaults = %s" % (self.cname, self.id, defaults)

        # get the nesting level in composite graph
        self.nesting_level = self.get_nesting_level()
        self.nesting_indent = " " * 4 * self.nesting_level

        # fix the block's group
        print "Block2 %s self.block_group" % (self.id,), self.block_group
        if type(self.block_group) is str: self.block_group = [self.block_group]
        
        ################################################################################
        # 1 general top block stuff: init bus, set top to self, init logging
        #   all the rest should be the same as for hier from file, hier from dict, loop, loop_seq
        if self.topblock:
            # print "Block2.init topblock conf.keys", self.conf['params'].keys()
            # print "Block2.init topblock numsteps", self.numsteps
            # fix the random seed
            # np.random.seed(self.randseed)
                
            self.top = self
            self.bus = Bus()
            # self.lsh_colors = lshash.LSHash(hash_size = 3, input_dim = 1000)
            
            # block_store init, topblock only
            def init_block_store():
                block_store_filename = 'data/block_store.h5'
                return pd.HDFStore(block_store_filename)

            self.block_store = init_block_store()

            # initialize pandas based hdf5 logging
            log.log_pd_init(self.conf)

            # write initial configuration to dummy table attribute in hdf5
            log.log_pd_store_config_initial(print_dict(self.conf))

            # initialize the graph
            self.init_block()

            # print "init top graph", print_dict(self.top.graph)
            
            # dump the execution graph configuration to a file
            finalconf = self.dump_final_config()
            
            # # this needs more work
            # log.log_pd_store_config_final(finalconf)
            # nx.write_yaml(self.nxgraph, 'nxgraph.yaml')
            try:
                nx.write_gpickle(self.nxgraph, 'nxgraph.pkl')
            except Exception, e:
                print "%s-%s init pickling graph failed on downstream objects, e = %s" % (self.cname, self.id, e)
                # print "Trying nxgraph_dump"

            log.log_pd_store_config_final(nxgraph_to_smp_graph(self.nxgraph))

            # print "Block2.init topblock self.blocksize", self.blocksize

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
        
    def set_attr_from_top_conf(self):
        for attr in ['saveplot']:
            top_attr = getattr(self.top, attr)
            if top_attr is not None and self.conf['params'].has_key(attr):
                print "Block2.set_attr_from_top_conf copying top.%s = %s to conf['params'] %s" % (attr, top_attr, self.conf['params']['saveplot'])
                self.conf['params'][attr] = top_attr
                setattr(self, attr, top_attr)

    def get_nesting_level(self):
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
        """Compute block identity in color
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
        
            ck = plot_colors.keys()[c]
            # print "Block2-%s.init_colors k = %s, ck = %s, color = %s" % (self.id, c, ck, plot_colors[ck])
            return plot_colors[ck]

        def get_color_from_plot_colors():
            if not hasattr(self.top, 'colorcnt'):
                self.top.colorcnt = 0
            else:
                self.top.colorcnt += 1
            ck =  plot_colors.keys()[self.top.colorcnt]
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
            print "%s init_colors is group %s, cmap = %s" % (self.id, self.block_group[0], group_cmap)
            return plt.get_cmap(group_cmap)(self.top.colorcnt / 77.0)
            
        # ck = get_color_from_plot_colors()
        
        # block color
        # self.block_color = get_color_from_plot_colors()
        self.block_color = get_color_from_cmap()
    
    def init_block(self):
        """Block2.init_block

        Init a graph based block: topblock, hierarchical inclusion from file or dictionary, loop, loop_seq
        """

        # init block color
        self.init_colors()
        
        ################################################################################
        # 2 copy the config dict to exec graph if hierarchical
        if hasattr(self, 'graph') or hasattr(self, 'subgraph') \
          or (hasattr(self, 'loopblock') and len(self.loopblock) != 0):
            """This is a composite block made up of other blocks via one of
            several mechanisms:
             - graph: is a graph configuration dict
             - subgraph: is path of configuration file
             - loopblock: loopblocks build subgraphs dynamically
             - cloneblock: we are cloning another subgraph referenced by existing
               nodeid
            """

            # print "has all these attrs %s-%s" % (self.cname, self.id)
            # for k,v in self.__dict__.items():
            #     print "%s-%s k = %s, v = %s" % (self.cname, self.id, k, v)

            # subgraph preprocess, propagate additional subgraph configuration
            if hasattr(self, 'subgraph'):
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

            ##############################################################################
            # node cloning (experimental)
            if hasattr(self, 'graph') \
              and type(self.graph) is str \
              and self.graph.startswith('id:'):
                # search node
                print "top graph", self.top.nxgraph.nodes()
                targetid = self.graph[3:] # id template
                targetnode = nxgraph_node_by_id_recursive(self.top.nxgraph, targetid)
                print "%s-%s" % (self.cname, self.id), "targetid", targetid, "targetnode", targetnode
                if len(targetnode) > 0:
                    print "    targetnode id = %d, node = %s" % (
                        targetnode[0][0],
                        targetnode[0][1].node[targetnode[0][0]])
                # copy node
                clone = {}
                tnode = targetnode[0][1].node[targetnode[0][0]]
                for k in tnode.keys():
                    print "cloning: subcloning: k = %s, v = %s" % (k, tnode[k])
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
                for k, v in clone['block_'].inputs.items():
                    # if v['bus']
                    # v['bus'].split("/")[0]
                    # v['bus'].split("/")[0] + "_clone"
                    if hasattr(self, 'inputs'):
                        v = self.inputs[k]
                    else:
                        # replace all occurences of original id with clone id
                        v['bus'] = re.sub(id_orig, clone['params']['id'], v['bus'])
                    clone['block_'].inputs[k] = copy.deepcopy(v)
                    print "%s.init cloning  input k = %s, v = %s" % (self.cname, k, clone['block_'].inputs[k])
                    
                # replace output refs
                for k, v in clone['block_'].outputs.items():
                    # v['buskey'].split("/")[0], v['buskey'].split("/")[0] + "_clone"
                    v['buskey'] = re.sub(id_orig, clone['params']['id'], v['buskey'])
                    print "%s.init cloning output k = %s, v = %s" % (self.cname, k, v)
                    clone['block_'].outputs[k] = copy.deepcopy(v)
                print "cloning: cloned block_.id = %s" % (clone['block_'].id)
                
                # add the modified node
                self.nxgraph.add_node(0, clone)

                # puh!

            # end node clone
            ##############################################################################
                
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
                
        ################################################################################
        # 3 initialize a primitive block
        else:
            self.init_primitive()
            
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

        self.init_block_cache()
        
    def init_block_cache(self):
        """init_block_cache

        Block result caching. FIXME: unclear about exec spec / blocksize, step-wise or batch output
        """
        # initialize block cache
        def check_block_store(block = None, top = None):
            import datetime
            assert block is not None, "Need some block to work on"
            # if not block.isprimitive: return
            m = md5(str(block.conf))
            self.md5 = m.hexdigest()
            self.cache = None
            
            # store exists?
            if len(block.top.block_store.keys()) > 0:
                # print "init_block_cache: block.top.block_store.keys()", block.top.block_store.keys()
                # print "init_block_cache: block.top.block_store.has_key('/blocks')", '/blocks' in block.top.block_store.keys()
                # print "check_block_store", block.top.block_store['blocks'].shape
                # print "init_block_cache: self.md5", self.cname, self.id, self.md5
                # print "blocks", type(block.top.block_store['/blocks'])
                # print block.top.block_store['blocks']['md5'] == self.md5
                # print block.top.block_store['/blocks']
                
                try:
                    self.cache = block.top.block_store['blocks'][:][block.top.block_store['blocks']['md5'] == self.md5]
                except Exception, e:
                    print "%s-%s.check_block_store cache retrieval for %s failed with %s" % (self.cname, self.id, self.md5, e)
                    self.cache = None
                    
                # print "check_block_store", self.md5, block.top.block_store['blocks']['md5']
                # print "init_block_cache: self.cache", self.cache
                
            # found cache
            if self.top.cached and self.cache is not None and self.cache.shape[0] != 0:
                print "%s-%s.check_block_store: found cache for %s\n   cache['log_stores'] = %s" % (self.cname, self.id, self.md5, self.cache['log_store'].values)
                
                # FIXME: check experiment.cache to catch randseed and numsteps
                # load cached data
                self.cache_h5 = pd.HDFStore(self.cache['log_store'].values[0])
                self.cache_data = {}
                for outk in self.outputs.keys():
                    x_ = self.cache_h5['%s/%s' % (self.id, outk)].values
                    print "output %s cached data = %s" % (outk, x_.shape)
                    self.cache_data[outk] = x_.copy()
            else:
                print "%s-%s.check_block_store: no cache exists for %s, storing %s at %s" % (self.cname, self.id, self.md5, self.conf, self.md5)
                
                # create entry and save
                columns = ['md5', 'timestamp', 'block', 'params', 'log_store'] # 'experiment',
                values = [[self.md5, pd.to_datetime(datetime.datetime.now()), str(block.conf['block']), str(block.conf['params']), log.log_store.filename]]
                df = pd.DataFrame(data = values, columns = columns)
                # print "df =\n", df

                # block store is empty
                if len(block.top.block_store.keys()) < 1:
                    block.top.block_store['/blocks'] = df
                else:
                    block.top.block_store['/blocks'] = pd.concat([block.top.block_store['/blocks'], df])

        if self.top.cached:
            check_block_store(block = self)
        
    def init_subgraph(self):
        """Block2.init_subgraph

        Initialize a Block2's subgraph

        Subgraph is a filename of another full graph config as opposed
        to a graph which is specified directly as a dictionary.
        """

        # print "lconf", self.lconf

        # subgraph dictionary / orderdeddict
        if type(self.subgraph) is OrderedDict:
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
            
        # subgraph file containing the OrderedDict
        else:
            # print "init_subgraph config globals", self.top.conf_localvars.keys()
            if hasattr(self, 'lconf'):
                # print "lconf", self.lconf
                subconf_localvars = get_config_raw(conf = self.subgraph, confvar = None, lconf = self.lconf)
                # print "init_subgraph returning localvars", subconf_localvars.keys()
                subconf = subconf_localvars['conf']
            else:
                subconf_localvars = get_config_raw(self.subgraph, confvar = None, lconf = self.top.conf_localvars) # 'graph')
                subconf = subconf_localvars['conf']
                
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
            for confk, confv in self.subgraphconf.items():
                # split block id from config parameter
                (confk_id, confk_param) = confk.split("/")
                # get the block's node
                confnode = dict_search_recursive(self.conf['params']['graph'], confk_id)
                # return on fail
                if confnode is None: continue
                print "subgraphconf node = %s" % (confnode['block'], )
                print "               param = %s" % (confk_param, )
                print "               val_bare = %s" % (confv, )
                print "               val_old = %s" % (confnode['params'][confk_param], )
                # overwrite param dict
                # tmp = {}
                # tmp.update(confnode['params'][confk_param], **confv)
                if type(confv) is dict:
                    tmp_ = copy.copy(confnode['params'][confk_param])
                    tmp_.update(**confv)
                    print "               val_new = %s, tmp_ = %s" % (confnode['params'][confk_param], tmp_)
                else:
                    print "               val_new = %s" % (confv, )
                    confnode['params'][confk_param] = confv
                    
                # print "               val_new = %s, val_old = %s" % (confv, confnode['params'][confk_param])
                # debug print
                for paramk, paramv in confnode['params'].items():
                    print "    %s = %s" % (paramk, paramv)
        # # debug
        # print self.conf['params']['graph']['brain_learn_proprio']['params']['graph'][confk_id]

        # rewrite id strings?
        if hasattr(self, 'subgraph_rewrite_id') and self.subgraph_rewrite_id:
            # self.outputs_copy = copy.deepcopy(self.conf['params']['outputs'])
            nks_0 = dict_get_nodekeys_recursive(self.conf['params']['graph'])
            # xid = self.conf['params']['id'][-1:]
            xid = self.conf['params']['id'].split('_')[-1]
            self.conf['params']['graph'] = dict_replace_idstr_recursive2(
                d = self.conf['params']['graph'], xid = xid)
            nks_l = dict_get_nodekeys_recursive(self.conf['params']['graph'])

            if self.conf['params'].has_key('outputs'):
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

        print "{2}{0: <20}-{3}.init_graph_pass_1 graph.keys = {1}".format(self.cname[:20], self.nxgraph.nodes(), self.nesting_indent, self.id)
        
        # if hasattr(self, 'graph'):
        #     print "    graph", self.graph, "\n"
        #     print "    nxgraph", self.nxgraph, "\n"
        
        # pass 1 init
        # for k, v in self.graph.items():
        # for n in self.nxgraph.nodes_iter():
        for i in range(self.nxgraph.number_of_nodes()):
            v = self.nxgraph.node[i]
            k = v['params']['id']
            # print "%s-%s.init_graph_pass_1 node = %s" % (self.cname, k, v.keys()), v['params'].keys()
            # if v['params'].has_key('outputs'):
            #     print v['params']['outputs']
            # v = n['params']
            # self.debug_print("__init__: pass 1\nk = %s,\nv = %s", (k, print_dict(v)))

            # debug timing
            print "{3}{0: <20}.init pass 1 k = {1: >5}, v = {2: >20}".format(
                self.__class__.__name__[:20], k[:20], v['block'].__name__, self.nesting_indent)
            then = time.time()

            # print v['block_']
            
            # instantiate block
            # self.graph[k]['block'] = self.graph[k]['block'](conf = v, paren = self, top = self.top)
            v['block_'] = v['block'](conf = v, paren = self, top = self.top)
            # print "block_color", self.nxgraph.node[i]['block_'].block_color
            
            # print "%s init self.top.graph = %s" % (self.cname, self.top.graph.keys())
            
            # complete time measurement
            print "{3}{0: <20}.init pass 1 k = {1: >5}, v = {2: >20}".format(self.__class__.__name__[:20], k, v['block_'].cname, self.nesting_indent),
            print "took %f s" % (time.time() - then)
            
            # print "%s self.graph[k]['block'] = %s" % (self.graph[k]['block'].__class__.__name__, self.graph[k]['block'].bus)
        # done pass 1 init

    def init_graph_pass_2(self):
        """Block2.init_graph_pass_2

        Pass 2 of graph initialization: Iterate nodes and call pass2 of block instance init.

        Arguments: None

        Returns: None
        """
        # iterate over nxgraph's nodes
        for i in range(self.nxgraph.number_of_nodes()):
            v = self.nxgraph.node[i]
            k = v['params']['id']
            
            self.debug_print("__init__: pass 2\nk = %s,\nv = %s", (k, print_dict(v)))
            # print "%s.init pass 2 k = %s, v = %s" % (self.__class__.__name__, k, v['block'].cname)
            print "{3}{0: <20}.init pass 2 k = {1: >5}, v = {2: >20}".format(self.__class__.__name__[:20], k, v['block_'].cname, self.nesting_indent),
            then = time.time()
            # self.graph[k]['block'].init_pass_2()
            v['block_'].init_pass_2()
            print "took %f s" % (time.time() - then)

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

        for k, v in self.outputs.items(): # problematic
        # for k, v in self.conf['params']['outputs'].items():
            # print "%s.init_outputs: outk = %s, outv = %s" % (self.cname, k, v)
            assert type(v) is dict, "Old config of %s output %s with type %s, %s" % (self.id, k, type(v), v)
            # print "v.keys()", v.keys()
            # assert v.keys()[0] in ['shape', 'bus'], "Need 'bus' or 'shape' key in outputs spec of %s" % (self.id, )
            assert v.has_key('shape'), "Output spec %s: %s needs 'shape' param" % (k, v)
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
            if not v.has_key('logging'):
                v['logging'] = True
                
            # set self attribute to that shape
            if not hasattr(self, k) or getattr(self, k).shape != v['shape']:
                setattr(self, k, np.zeros(v['shape']))
            
            # print "%s.init_outputs: %s.bus[%s] = %s" % (self.cname, self.id, v['buskey'], getattr(self, k).shape)
            self.bus[v['buskey']] = getattr(self, k).copy()
            # self.bus.setval(v['buskey'], getattr(self, k))

            # ros?
            if hasattr(self, 'ros') and self.ros:
                self.msgs[k] = Float64MultiArray()
                self.pubs[k] = rospy.Publisher('%s/%s' % (self.id, k, ), Float64MultiArray, queue_size = 2)
            
            # output item initialized
            v['init'] = True

    # def init_colors(self):
    #     self.nodecolor

    def init_logging(self):
        # initialize block logging
        if not self.logging: return

        # assume output's initialized        
        for k, v in self.outputs.items():
            if (not v.has_key('init')) or (not v['init']) or (not v['logging']): continue
                
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
            for k, v in self.inputs.items():
                self.debug_print("__init__: pass 2\n    in_k = %s,\n    in_v = %s", (k, v))
                assert len(v) > 0
                # FIXME: when is inv not a dict?
                # assert type(v) is dict, "input value %s in block %s/%s must be a dict but it is a %s, probably old config" % (k, self.cname, self.id, type(v))
                # assert v.has_key('shape'), "input dict of %s/%s needs 'shape' entry, or do something about it" % (self.id, k)
                
                # set input from bus
                if v.has_key('bus'):
                    if v.has_key('shape'):
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
                        
                        # initialize input buffer
                        v['val'] = np.zeros(v['shape']) # ibuf >= blocksize
                        
                        # bus item does not exist yet
                        if not self.bus.has_key(v['bus']):
                            
                            # FIXME: hacky
                            for i in range(1): # 5
                                if i == 0: print "\n"
                                # print "%s-%s init (pass 2) WARNING: bus %s doesn't exist yet and will possibly not be written to by any block, buskeys = %s" % (self.cname, self.id, v['bus'], self.bus.keys())
                                print "%s%s-%s init (pass 2) WARNING: nonexistent bus %s" % (self.cname, self.id, v['bus'], self.nesting_indent)
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
                                print "#" * 80
                                print "sl", sl, v['val']
                            else:
                                # print "\nsetting", self.cname, v
                                assert v['shape'][0] == self.bus[v['bus']].shape[0], "%s-%s's input buffer and input bus shapes need to agree (besides blocksize) for input %s, buf: %s, bus: %s/%s" % (self.cname, self.id, k, v['shape'], v['bus'], self.bus[v['bus']].shape)
                                v['val'][...,-blocksize_input_bus:] = self.bus[v['bus']].copy()
                            # v['val'] = self.bus[v['bus']].copy() # inbus
                            # v['val'][...,0:inbus.shape[-1]] = inbus
                        # print "Blcok2: init_pass_2 v['val'].shape", self.id, v['val'].shape
                        
                    elif not v.has_key('shape'):
                        # check if key exists or not. if it doesn't, that means this is a block inside dynamical graph construction
                        # print "\nplotblock", self.bus.keys()

                        assert self.bus.has_key(v['bus']), "Requested bus item %s is not in buskeys %s" % (v['bus'], self.bus.keys())
                    
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
                    self.debug_print("%s.init_pass_2: %s, bus[%s] = %s, input = %s", (self.cname, self.id, v['bus'], self.bus[v['bus']].shape, v['val'].shape))
                # elif type(v[0]) is str:
                #     # it's a string but no valid buskey, init zeros(1,1)?
                #     if v[0].endswith('.h5'):
                #         setattr(self, k, v[0])
                else:
                    assert v.has_key('val'), "Input spec needs either a 'bus' or 'val' entry in %s" % (v.keys())
                    # expand scalar to vector
                    if np.isscalar(v['val']):
                        # check for shape info
                        if not v.has_key('shape'):
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
                self.debug_print("%s.init k = %s, v = %s", (self.cname, k, v))
                self.debug_print("init_pass_2 %s in_k.shape = %s / %s", (self.id, v['val'].shape, v['shape']))
            
    def debug_print(self, fmtstring, data):
        """only print if debug is enabled for this block"""
        fmtstring = "\n%s[%d]." + fmtstring
        data = (self.cname,self.cnt) + data
        if self.debug:
            print fmtstring % data

    # undecorated step, need to count ourselves
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
        for i in range(self.nxgraph.number_of_nodes()):
            v = self.nxgraph.node[i]
            k = v['params']['id']
            
            # for k, v in self.graph.items():
            v['block_'].step()
            # try:
            #     v['block_'].step()
            # except:
            #     pdb.set_trace()
            # print "%s-%s.step[%d]: k = %s, v = %s" % (self.cname, self.id, self.cnt, k, type(v))

        if self.topblock:
            # # then log all
            # # for k, v in self.graph.items(): # self.bus.items():
            # for i in range(self.nxgraph.number_of_nodes()):
            #     v = self.nxgraph.node[i]
            #     k = v['params']['id']
            #     # return if block doesn't want logging
            #     if not v['block_'].logging: continue
            #     # if (self.cnt % v['block'].blocksize) != (v['block'].blocksize - 1): continue
            #     if (self.cnt % v['block_'].blocksize) > 0: continue
            #     # print debug foo
            #     self.debug_print("step: node k = %s, v = %s", (k, v))
            #     # do logging for all of the node's output variables
            #     for k_o, v_o in v['block_'].outputs.items():
            #         buskey = "%s/%s" % (v['block_'].id, k_o)
            #         # print "%s step outk = %s, outv = %s, bus.sh = %s" % (self.cname, k_o, v_o, self.bus[buskey].shape)
            #         log.log_pd(tbl_name = buskey, data = self.bus[buskey])

            # store log incrementally
            if (self.cnt) % 500 == 0 or self.cnt == (self.numsteps - 1):
                print "storing log @iter % 4d/%d" % (self.cnt, self.numsteps)
                log.log_pd_store()

            # store log finally: on final step, also copy data attributes to log attributes
            if self.cnt == self.numsteps:
                # store and close logging
                self.log_close()
                # save plot figures
                self.plot_close()
                
        # all Block2's
        if (self.cnt % self.blocksize) in self.blockphase:
            # buscopy: copy outputs from subblocks as configured in enclosing block outputs spec
            self.bus_copy()
            
        # need to to count ourselves
        # self.cnt += 1

    def bus_copy(self):
        for k, v in [(k_, v_) for k_, v_ in self.outputs.items() if v_.has_key('buscopy')]:
            buskey = v['buscopy']
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

    def plot_close(self):
        # print "%s-%s\n    .plot_close closing %d nodes" % (self.cname, self.id, self.nxgraph.number_of_nodes())
        for n in self.nxgraph.nodes_iter():
            node = self.nxgraph.node[n]['block_']
            if hasattr(node, 'nxgraph'):
                # descend
                node.plot_close()

            if hasattr(node, 'saveplot'):
                # print "%s-%s\n    .plot_close examining node %s (%s)" % (self.cname, self.id, node.id, node.saveplot)
            
                # if type(node) is Block2 and hasattr(node, 'saveplot') and node.saveplot:
                if node.saveplot:
                    # print "%s-%s\n    .plot_close closing node saving plot %s" % (self.cname, self.id, node.id,)
                    node.save()
            
    def log_close(self):
        print "storing log @final iter %04d" % (self.cnt, )
        # store
        log.log_pd_store()
        # recursively copy the attributes
        self.log_attr()
        # close the file
        log.log_pd_deinit()
                    
    def log_attr(self):
        """Block2.log_attr: enumerate all nodes in hierarchical graph and copy the node's output attributes to table attributes"""
        for i in range(self.nxgraph.number_of_nodes()):
            # assume output's initialized
            node = self.nxgraph.node[i]['block_']
            # print "%s-%s log_attr\n    node = %s, logging = %s" % (self.cname, self.id, node, node.logging)
            if not node.logging: continue

            # depth first
            if hasattr(node, 'nxgraph'):
                # recur
                node.log_attr()

            # loop output items
            for k,v in node.outputs.items():
                if (not v.has_key('init')) or (not v['init']) or (not v['logging']): continue
        
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
        for k, v in self.__dict__.items():
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
        dump_final_config_file = "data/%s.conf" % (self.id)
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

        preprocess input tensor using input specification

        FIXME: 
        """
        if self.inputs[k].has_key('embedding'):
            emblen = self.inputs[k]['embedding']
            embshp = self.inputs[k]['shape']
            assert len(embshp) == 2
            ret = np.zeros((embshp[0] * emblen, embshp[1]))
            for i in range(embshp[1]):
                if i < emblen: continue
                tmp = np.hstack(tuple(self.inputs[k]['val'][:,i-j] for j in range(emblen)))
                # print "tmp", tmp.shape, # tmp
                ret[:,i] = tmp
            print "ret", ret.shape, # 
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

        # FIXME: check return type of func at init time and set step function

    @decStep()
    def step(self, x = None):
        """FuncBlock2.step

        Function block step function
        """
        # print "%s.step inputs[%d] = %s" % (self.cname, self.cnt, self.inputs)
        # self.inputs is a dict with values [array]
        # assert self.inputs.has_key('x'), "%s.inputs expected to have key 'x' with function input vector. Check config."

        self.debug_print("step[%d]: x = %s", (self.cnt, self.inputs,))

        # assumes func to be smp_graphs aware and map the input/output onto the inner function 
        f_val = self.func(self.inputs)
        if type(f_val) is dict:
            for k, v in f_val.items():
                setattr(self, k, v)
        else:
            for outk, outv in self.outputs.items():
                # print "k, v", outk, outv, f_val
                setattr(self, outk, f_val)
            self.y = f_val
            self.debug_print("step[%d]: y = %s", (self.cnt, self.y,))
            
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
    - loop: the loop specification. either a list of tuples or a function returning tuples. Tuples have the form ('param', value) and param is a configuration parameter of the inner loopblock.
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
    def __init__(self, conf = {}, paren = None, top = None):
        self.defaults['loop'] = [1]
        self.defaults['loopblock'] = {}

        # force loopmode set
        self.loopmode = 'parallel'
        
        assert conf['params'].has_key('loop'), "Come on, looping without specification is dumb"

        # parent Block2 init takes care of constructing self.nxgraph
        Block2.__init__(self, conf = conf, paren = paren, top = top)

        # done
        
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

        self.cnt += 1

class SeqLoopBlock2(Block2):
    """SeqLoopBlock2 class

    A sequential loop block: dynamic instantiation of Blocks within loop iterations

    FIXME: clean up primitive / decstep issues, in/out rewriting, etc
    FIXME: work with nxgraph
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
        results = self.loop(self, i, f_obj)
        return results
            
    @decStep()
    def step(self, x = None):
        """SeqLoopBlock2.step"""
        self.debug_print("%s.step:\n\tx = %s,\n\tbus = %s,\n\tinputs = %s,\n\toutputs = %s",
                             (self.__class__.__name__,self.outputs.keys(),
                                  self.bus, self.inputs, self.outputs))

        def f_obj(lparams):
            """instantiate the loopblock and run it"""
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
                
            for k, v in self.loopblock['params'].items():
                # print "SeqLoopBlock2.step.f_obj loopblock params", k, v # , lparams[0]

                if k == 'id':
                    loopblock_params[k] = "%s_%d" % (self.id, i)
                elif k == lparams[0]:
                    loopblock_params[k] = lparams[1]
                else:
                    loopblock_params[k] = v

            self.debug_print(
                "%s.step.f_obj:\n    loopblock_params = %s",
                (self.cname, loopblock_params))
            
            # create dynamic conf, beware the copy (!!!)
            loopblock_conf = {'block': self.loopblock['block'], 'params': copy.deepcopy(loopblock_params)}
            
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
            for j in range(1, self.dynblock.numsteps+1):
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
            for outk, outv in self.dynblock.outputs.items():
                d[outk] = getattr(self.dynblock, outk)
            # print "j", j, "d", d
            return d
        
        def f_obj_hpo(params):
            # print "f_obj_hpo: params", params
            x = np.array([params]).T
            # XXX
            lparams = ('inputs', {'x': {'val': x}}) # , x.shape, self.outputs['x']
            # print "%s.step.f_obj_hpo lparams = {%s}" % (self.cname, lparams)
            f_obj(lparams)

            # now we have self.dynblock
            assert hasattr(self, 'dynblock')

            # compute the loss for hpo from dynblock outputs
            loss = 0
            for outk in self.outputs.keys():
                # omit input values / FIXME
                if outk in self.dynblock.inputs.keys(): continue
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
            f_obj_ = f_obj
        else:
            f_obj_ = f_obj_hpo

        # loop the loop
        then = time.time()
        # print "%s-%s.step[%d]" % (self.cname, self.id, self.cnt)
        # loopblock loop
        for i in range(self.numsteps/self.loopblocksize):
            # print "%s-%s.step[%d] loop iter %d" % (self.cname, self.id, self.cnt, i,)
            sys.stdout.flush()
            then = time.time()

            # run the loop, if it's a func loop: need input function from config
            results = self.f_loop(i, f_obj_)
            # print "results", results
            self.debug_print("SeqLoopBlock2 f_loop results[%d] = %s", (i, results))

            # FIXME: WORKS for loop example hpo, model sweeps,
            #        BREAKS for real experiment with measure output
            # # copy dict to self attrs
            # if results is not None:
            #     for k, v in results.items():
            #         self.debug_print("SeqLoopBlock2.step loop %d result k = %s, v = %s", (i, k, v))
            #         setattr(self, k, v)
                    
            # dynblock = results['dynblock']
            # lparams = results['lparams']
            
            # print "%s.steploop[%d], %s, %s" % (self.cname, i, lparams, self.loopblock['params'])
            # for k,v in dynblock.outputs.items():
            #     print "dynout", getattr(dynblock, k)

            for outk in self.outputs.keys():
                # print "SeqLoopBlock2.step[%d] loop iter %d, outk = %s, dynblock outk = %s" % (self.cnt, i, outk, self.dynblock.outputs.keys(), )
                outvar = getattr(self, outk)
                # print "SeqLoopBlock2.step[%d] loop iter %d, outk = %s, outvar = %s" % (self.cnt, i, outk, outvar, )
                
                # func: need output function from config
                # FIXME: handle loopblock blocksizes greater than one
                # self.__dict__[outk][:,[i]] = np.mean(getattr(dynblock, outk), axis = 1, keepdims = True)
                
                # FIXME: which breaks more?
                # outslice = slice(i*self.dynblock.blocksize, (i+1)*self.dynblock.blocksize)

                # FIXME: which breaks more?
                outslice = slice(i*self.dynblock.outputs[outk]['shape'][-1], (i+1)*self.dynblock.outputs[outk]['shape'][-1])
                
                # print "self.   block", self.outputs[outk]
                # print "self.dynblock", self.dynblock.outputs[outk], getattr(self.dynblock, outk).shape #, self.dynblock.file
                self.debug_print(
                    "%s.step self.%s = %s, outslice = %s",
                    (self.cname, outk, getattr(self, outk).shape, outslice, ))
                    
                # print "    dynblock = %s.%s" % (self.dynblock.cname)
                outvar[:,outslice] = getattr(self.dynblock, outk).copy()
                # print "dynblock-%s outslice = %s, outvar = %s/%s%s, dynblock.out[%s] = %s" %(self.dynblock.id, outslice, outvar.shape, outvar[...,:].shape, outvar[...,outslice].shape, outk, getattr(self.dynblock, outk).shape)
        sys.stdout.write('\n')

        confgraph_full = {'block': Block2, 'params': {'id': self.id, 'graph': self.confgraph}}

        # print "%s-%s.step conf['params']['id'] = %s" % (self.cname, self.id, confgraph_full['params']['id'])
        self.nxgraph = nxgraph_from_smp_graph(confgraph_full)
        
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

    Integrator block: add input 'k_i' in input.keys to current value of output state 'Ik_i'

    Params: inputs ['x'], outputs ['Ix'], leakrate [1.0]
    """
    defaults = {
        # 'leak': 0.0,
        'inputs': {'x': {'shape': (1,1), 'val': np.zeros((1,1))}},
        'outputs': {},
        }
        
    @decInit()
    def __init__(self, conf = {}, paren = None, top = None):
        # default out
        if not conf['params'].has_key('outputs'):
            conf['params']['outputs'] = {}

        # create output states
        for k, v in conf['params']['inputs'].items():
            print "%s.init inkeys %s" % (self.__class__.__name__, k)
            print "IBlock2 conf['params'] keys", conf['params'].keys()
            print "IBlock2 conf['params']['outputs'] keys = %s" % (conf['params']['outputs'].keys(), ) # ["I%s" % k]
            outk = 'I%s' % (k, )
            busk = conf['params']['inputs'][k]
            # print "IBlock2 outk = %s, busk = %s" % (outk, busk, )
            if conf['params']['outputs'].has_key(outk):
                # conf['params']['outputs'][outk] = {'shape': top.bus[busk['bus']].shape} # ['val'].shape]}
                pass
            else:
                conf['params']['outputs'][outk] = {'shape': v['shape']} # {'shape': top.bus[busk['bus']].shape} # ['val'].shape]}
            
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)

        # print "IBlock2 outputs", self.outputs
        
        if hasattr(self, 'leak'):# and self.leak > 0.0:
            self.step = self.step_leak
        else:
            self.step = self.step_all

    @decStep()
    def step_leak(self, x = None):
        for i in range(self.blocksize):
            for ink in self.inputs.keys():
                outk = "I%s" % ink
                tmp_ = getattr(self, outk)
                tmp_[:,i] = ((1 - self.leak) * tmp_[:,i-1]) + (self.inputs[ink][0][:,i] * self.d)
                setattr(self, outk, tmp_)

    @decStep()
    def step_all(self, x = None):
        for ink, inv in self.inputs.items():
            outk = 'I%s' % ink
            # input integral / cumsum
            Iin = np.cumsum(inv['val'], axis = 1) # * self.d
            # print getattr(self, outk)[:,[-1]].shape, self.inputs[ink][0].shape, Iin.shape
            # single step
            # setattr(self, outk, getattr(self, outk)[:,[-1]] + Iin)
            # setattr(self, outk, getattr(self, outk) + (self.inputs[ink][0] * 1.0))
            # multi step / batch
            setattr(self, outk, Iin)
            print "IBlock2.step[%d] self.%s = %s / %s" % (self.cnt, outk, getattr(self, outk).shape, self.outputs[outk]['shape'])

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
        if not conf['params'].has_key('outputs'):
            conf['params']['outputs'] = {}
            
        for ink in conf['params']['inputs'].keys():
            # get input shape
            inshape = top.bus[conf['params']['inputs'][ink]['bus']].shape
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
        for ink in self.inputs.keys():
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
            print "dBlock2.step", self.id, self.cnt, getattr(self, outk)[:,[-1]].shape, self.inputs[ink]['val'].shape, din.shape
            setattr(self, outk, din)
            # store current input
            setattr(self, ink_, self.inputs[ink]['val'].copy())
            # print "dBlock2.step", self.id, self.cnt, getattr(self, outk), din

class DelayBlock2(PrimBlock2):
    """DelayBlock2 class

    Delay block: delay signal with internal ringbuffer delay line.

    Params: inputs, delay in steps / shift
    """
    @decInit() # outputs from inputs block decorator
    def __init__(self, conf = {}, paren = None, top = None):
        """DelayBlock2 init"""
        params = conf['params']
        if not params.has_key('outputs'):
            params['outputs'] = {}

        delays_ = {}
            
        for ink, inv in params['inputs'].items():
            # get input shape
            # assert top.bus.has_key(params['inputs'][ink]['bus']), "DelayBlock2 needs existing bus item at %s to infer delay shape" % (params['inputs'][ink]['bus'], )
            if top.bus.has_key(params['inputs'][ink]['bus']):
                inshape = top.bus[params['inputs'][ink]['bus']].shape
            else:
                inshape = inv['shape']
            # alloc delay block
            if params.has_key('delays'):
                delays_ = params['delays'][ink]
                # setattr(self, "%s_" % ink, np.zeros((inshape[0], inshape[1] + params['delays'][ink])))
            else:
                delays_[ink] = inv['delay']
                # setattr(self, "%s_" % ink, np.zeros((inshape[0], inshape[1] + inv['delay'])))
            setattr(self, "%s_" % ink, np.zeros((inshape[0], inshape[1] + delays_[ink])))
                
            # set output members
            params['outputs']["d%s" % ink] = {'shape': inshape}

        params['delays'] = delays_
            
        # base block init
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)

    @decStep()
    def step(self, x = None):
        """DelayBlock2 step"""
        for ink in self.inputs.keys():
            # output key
            outk = "d%s" % ink
            # input from last block
            ink_ = "%s_" % ink
            # stack last and current block input
            # tmp_   = np.hstack((inv_, self.inputs[ink][0]))
            # slice -(blocksize + 1) until now
            inv_ = getattr(self, ink_)
            sl = slice(self.delays[ink], self.delays[ink]+self.blocksize)
            # print "DelayBlock2: ink", ink, "sl", sl, "inv_", inv_.shape, "input", self.inputs[ink]['val'].shape
            inv_[...,sl] = self.inputs[ink]['val'].copy()
            # print "DelayBlock2: ink", ink, inv_[...,sl].shape
            
            # compute the diff in the input
            # din = np.diff(tmp_[:,tmp_sl], axis = 1) # * self.d
            # which should be same shape is input
            # assert din.shape == self.inputs[ink][0].shape
            setattr(self, outk, inv_[...,slice(0, self.blocksize)])
            # print "DelayBlock2 outk %s shape" %(outk,), self.inputs[ink]['val'].shape, getattr(self, ink_).shape, getattr(self, outk) # [...,[-1]].shape, self.inputs[ink]['val'].shape #, din.shape
            # store current input
            setattr(self, ink_, np.roll(inv_, shift = -self.blocksize, axis = 1))
                        
class SliceBlock2(PrimBlock2):
    """SliceBlock2

    Cut slices from the input tensor

    FIXME: make slicing a general function of block i/o
    FIXME: generic ndim slicing?
    """
    def __init__(self, conf = {}, paren = None, top = None):
        params = conf['params']
        if not params.has_key('outputs'):
            params['outputs'] = {}
            
        for k, v in params['inputs'].items():
            slicespec = params['slices'][k]
            # print slicespec
            for slk, slv in slicespec.items():
                # print "%s.init inkeys %s, slicekey = %s" % (self.__class__.__name__, k, slk)
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
        for ink in self.inputs.keys():
            # print "%s-%s[%d] ink = %s, inv = %s" % (self.cname, self.id, self.cnt,
            #                                             ink, self.inputs[ink])
            slicespec = self.slices[ink]
            for slk, slv in slicespec.items():
                outk = "%s_%s" % (ink, slk)
                setattr(self, outk, self.inputs[ink]['val'][slv])
                # print "%s-%s.step[%d] outk = %s, outsh = %s, out = %s" % (self.cname, self.id, self.cnt, outk, getattr(self, outk).shape, getattr(self, outk))

class StackBlock2(PrimBlock2):
    """StackBlock2 class

    Stack block can combine input slices into a single output item
    """
    def __init__(self, conf = {}, paren = None, top = None):
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)

    @decStep()
    def step(self, x = None):
        st = [inv['val'] for ink, inv in self.inputs.items()]
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
        assert self.inputs['c']['val'].shape[:-1] == self.x.shape[:-1], "ConstBlock2 input / output shapes must agree: %s == %s?" % (self.inputs['c']['val'].shape[:-1], self.x.shape[:-1])
        
        # replicate column vector
        # if self.x.shape[1] == 1: # this was wrong
        if self.inputs['c']['val'].shape[1] == 1:
            self.x = np.tile(self.inputs['c']['val'], self.blocksize) # FIXME as that good? only works for single column vector
        else:
            self.x = self.inputs['c']['val'].copy() # FIXME as that good? only works for single column vector

class CountBlock2(PrimBlock2):
    """CountBlock2 class

    Count block: output is a counter
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
        self.outk = self.outputs.keys()[0]
        # init cnt_ of blocksize
        # self.cnt_ = np.zeros(self.outputs[self.outk]['shape'] + (self.blocksize,))
        self.cnt_ = np.zeros(self.outputs[self.outk]['shape'])
        # print self.inputs
        # FIXME: modulo / cout range with reset/overflow

        print "\n%s endofinit bus = %s\n" % (self.cname, self.bus.keys())
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
        for k, v in self.outputs.items():
            x_ = np.random.uniform(-1e-1, 1e-1, v['shape'])
            if hasattr(self, 'lo') and hasattr(self, 'hi'):
                x_ = np.random.uniform(self.lo, self.hi, v['shape'])
            setattr(self, k, x_.copy())
        # self.x = np.random.uniform(
        #     self.inputs['lo']['val'], self.inputs['hi']['val'],
        #     self.outputs['x']['shape'])
        
    @decStep()
    def step(self, x = None):
        self.debug_print("%s.step:\n\tx = %s,\n\tbus = %s,\n\tinputs = %s,\n\toutputs = %s",
                         (self.__class__.__name__,self.outputs.keys(), self.bus, self.inputs, self.outputs))

        # FIXME: relation rate / blocksize, remember cnt from last step, check difference > rate etc
        if self.cnt % self.rate == 0:
            # FIXME: take care of rate/blocksize issue
            for k, v in self.outputs.items():
                # x = np.random.uniform(self.inputs['lo'][0][:,[-1]], self.inputs['hi'][0][:,[-1]], (self.outputs[k][0]))
                # print 'lo', self.inputs['lo']['val'], '\nhi', self.inputs['hi']['val'], '\noutput', v['bshape']
                x = np.random.uniform(self.inputs['lo']['val'], self.inputs['hi']['val'], size = v['shape'])
                setattr(self, k, x)
            # print "self.x", self.x
        
        # # loop over outputs dict and copy them to a slot in the bus
        # for k, v in self.outputs.items():
        #     buskey = "%s/%s" % (self.id, k)
        #     self.bus[buskey] = getattr(self, k)
        # self.bus[self.id] = self.x
        return None
