"""smp_graphs - smp sensorimotor experiments as computation graphs

block: basic block of computation

2017 Oswald Berthold
"""

import uuid, sys, time, copy
from collections import OrderedDict, MutableMapping
import itertools
from functools import partial

import networkx as nx

import numpy as np

from hyperopt import STATUS_OK, STATUS_FAIL

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from smp_base.plot import makefig, set_interactive

import smp_graphs.logging as log
from smp_graphs.utils import print_dict, xproduct, myt
from smp_graphs.common import conf_header, conf_footer
from smp_graphs.common import get_config_raw, get_config_raw_from_string
from smp_graphs.common import set_attr_from_dict

from smp_graphs.graph import nxgraph_from_smp_graph

BLOCKSIZE_MAX = 10000

################################################################################
# utils, TODO move to utils.py
def ordereddict_insert(ordereddict = None, insertionpoint = None, itemstoadd = []):
    """self rolled ordered dict insertion
    from http://stackoverflow.com/questions/29250479/insert-into-ordereddict-behind-key-foo-inplace
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
    """A dictionary that applies an arbitrary key-altering
       function before accessing the keys"""

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

    def plot(self, ax = None):
        assert ax is not None
        xspacing = 10
        yspacing = 10
        yscaling = 0.66
        ax.text(10, 0, "Bus (%s)" % ("topblock"))
        ax.grid(0)
        ax.set_xlim((0, 100))
        ax.set_ylim((-100, 0))
        # ax.plot(np.random.uniform(-5, 5, 100), "ko", alpha = 0.1)
        i = 0
        for k, v in self.store.items():
            # print "k = %s, v = %s" % (k, v)
            ypos = -10 # -(i+1)*yspacing
            xpos = (i+1)*xspacing
            ax.text(xpos, ypos, "{0: <8}\n{1: <12}".format(k, v.shape), family = 'monospace')
            # elem shape
            ax.add_patch(
                patches.Rectangle(
                    # (30, ypos - (v.shape[0]/2.0) - (yspacing / 3.0)),   # (x,y)
                    (xpos+2, ypos-2),   # (x,y)
                    v.shape[0],          # width
                    -1 * yscaling,          # height
                    fill = False,
                    # hatch = "|",
                    hatch = "-",
                )
            )
            # blockshape
            ax.add_patch(
                patches.Rectangle(
                    # (30, ypos - (v.shape[0]/2.0) - (yspacing / 3.0)),   # (x,y)
                    (xpos+2, ypos-8),   # (x,y)
                    v.shape[0],          # width
                    -v.shape[1] * yscaling,          # height
                    fill = False,
                    # hatch = "|",
                    hatch = "-",
                )
            )
            i+=1
        plt.draw()
        plt.pause(1e-6)
        
################################################################################
# Block decorator init
class decInit():
    """!@brief Block.init wrapper"""
    def __call__(self, f):
        def wrap(xself, *args, **kwargs):
            f(xself, *args, **kwargs)
        return wrap

################################################################################
# Block decorator step
class decStep():
    """!@brief Block.step wrapper"""
    def __call__(self, f):
        def wrap(xself, *args, **kwargs):
            if True:
                sname  = self.__class__.__name__
                esname = xself.cname
                esid   = xself.id
                escnt  = xself.cnt
                # loop over block's inputs
                for k, v in xself.inputs.items():
                    # check sanity
                    assert v['val'].shape == v['shape'], "real and desired input shapes need to agree %s != %s" % (v['val'].shape, v['shape'])
                    # copy bus inputs to input buffer
                    if v.has_key('bus'): # input item has a bus associated in v['bus']
                        # if extended input buffer, rotate the data with each step
                        if xself.ibuf > 1:
                            # input blocksize
                            blocksize_input = xself.bus[v['bus']].shape[-1]
                            # input block border 
                            if (xself.cnt % blocksize_input) == 0: # (blocksize_input - 1):
                                # shift by input blocksize along self.blocksize axis
                                # axis = len(xself.bus[v['bus']].shape) - 1
                                axis = len(v['shape']) - 1
                                v['val'] = np.roll(v['val'], shift = -blocksize_input, axis = axis)
                                # print "%s.decStep v[val]" % (xself.cname), v['val'].shape, "v.sh", v['shape'], "axis", axis, "v", v['val']
                                
                                # set inputs [last-inputbs:last] if input blocksize reached
                                # # debugging in to out copy
                                # print "%s-%s.%s[%d] bus[%s] = %s" % (esname, esid,
                                #                                          sname,
                                #                                          escnt,
                                #                                          v[2],
                                #                                          xself.bus[v[2]])
                                
                                sl = slice(-blocksize_input, None)
                                # print xself.cname, "sl", sl, "bus.shape", xself.bus[v['bus']].shape, v['val'].shape
                                v['val'][...,-blocksize_input:] = xself.bus[v['bus']].copy() # np.fliplr(xself.bus[v[2]])
                                # xself.inputs[k][0][:,-1,np.newaxis] = xself.bus[v[2]]                                
                        else: # ibuf = 1
                            v['val'][...,[0]] = xself.bus[v['bus']]
                            
                    # copy input to output if inkey k is in outkeys
                    if k in xself.outputs.keys():
                        # outvar = v[2].split("/")[-1]
                        # print "%s.stepwrap split %s from %s" % (xself.cname, outvar, v[2])
                        setattr(xself, k, v['val'].copy()) # to copy or not to copy
                        # esk = getattr(xself, k)
                        
                        # # debug in to out copy
                        # print "%s.%s[%d]  self.%s = %s" % (esname, sname, escnt, k, esk)
                        # print "%s.%s[%d] outkeys = %s" % (esname, sname, escnt, xself.outputs.keys())
                        # if k in xself.outputs.keys():
                        #     print "%s.%s[%d]:   outk = %s" % (esname, sname, escnt, k)
                        #     print "%s.%s[%d]:    ink = %s" % (esname, sname, escnt, k)
                        #     # xself.outputs[k] = xself.inputs[k][0]

            # call the function on blocksize boundaries
            # FIXME: might not be the best idea to control that on the wrapper level as some
            #        blocks might need to be called every step nonetheless?
            if (xself.cnt % xself.blocksize) == 0: # or (xself.cnt % xself.rate) == 0:
                f_out = f(xself, None)

                # copy output to bus
                for k, v in xself.outputs.items():
                    buskey = "%s/%s" % (xself.id, k)
                    # print "copy %s.outputs[%s] = %s to bus[%s], bs = %d" % (xself.id, k, getattr(xself, k), buskey, xself.blocksize)
                    assert xself.bus[v['buskey']].shape == v['bshape'], "real and desired output shapes need to agree, %s != %s" % (xself.bus[v['buskey']].shape, v['shape'])
                    xself.bus[v['buskey']] = getattr(xself, k)
                    # print "xself.bus[v['buskey'] = %s]" % (v['buskey'], ) , xself.bus[v['buskey']]
            else:
                f_out = None
            
            # count calls
            xself.cnt += 1 # should be min_blocksize
            # xself.ibufidx = xself.cnt % xself.ibufsize
            
            return f_out
        # return the new func
        return wrap

################################################################################
# Base block class
class Block2(object):
    """!@brief Basic block class
    """
    defaults = {
        'id': None,
        'debug': False,
        'topblock': False,
        'ibuf': 1, # input  buffer size
        'obuf': 1, # output buffer size
        'cnt': 1,
        'blocksize': 1, # this is gonna be phased out
        'inputs': {}, # internal port, scalar / vector/ bus key, [slice]
        'outputs': {}, # name, dim
        'logging': True, # normal logging
        'rate': 1, # execution rate rel. to cnt
        'ros': False, # no ROS yet
    }

    def __init__(self, conf = {}, paren = None, top = None, blockid = None):
        # general stuff
        self.conf = conf
        self.paren = paren
        self.top = top
        self.cname = self.__class__.__name__
        
        # load defaults
        set_attr_from_dict(self, self.defaults)
                    
        # fetch existing configuration arguments
        if type(self.conf) == dict and self.conf.has_key('params'):
            set_attr_from_dict(self, self.conf['params'])
        else:
            print "What could it be? Look at %s" % (self.conf)

        # check id
        assert hasattr(self, 'id')
        # if id not is None:
        #     self.id = blockid

        # input buffer vs. blocksize: input buffer is sliding, blocksize is jumping
        if self.blocksize > self.ibuf:
            self.ibuf = self.blocksize

        ################################################################################
        # 1 general top block stuff: init bus, set top to self, init logging
        #   all the rest should be the same as for hier from file, hier from dict, loop, loop_seq
        if self.topblock:
            # fix the random seed
            np.random.seed(self.randseed)
                
            self.top = self
            self.bus = Bus()
            # initialize pandas based hdf5 logging
            log.log_pd_init(self.conf)

            # write initial configuration to dummy table attribute in hdf5
            log.log_pd_store_config_initial(print_dict(self.conf))

            # # dump the execution graph configuration to a file
            # finalconf = self.dump_final_config()
            # # this needs more work
            # log.log_pd_store_config_final(finalconf)
                
            # print "init top graph", print_dict(self.top.graph)
        else:
            # get bus from topblock
            self.bus = self.top.bus
            
        ################################################################################
        # 2 copy the config dict to exec graph if hierarchical
        if hasattr(self, 'graph') or hasattr(self, 'subgraph') \
          or (hasattr(self, 'loopblock') and len(self.loopblock) != 0): # composite block made up of other blocks FIXME: loop, loop_seq

            # print "has all these attrs %s-%s" % (self.cname, self.id)
            # for k,v in self.__dict__.items():
            #     print "%s-%s k = %s, v = %s" % (self.cname, self.id, k, v)

            # subgraph prepare
            if hasattr(self, 'subgraph'):
                self.init_subgraph()
            
            self.nxgraph = nxgraph_from_smp_graph(self.conf)
            
            # for n in self.nxgraph.nodes():
            #     print "%s-%s g.node[%s] = %s" % (self.cname, self.id, n, self.nxgraph.node[n])
        
            # 2.1 init_pass_1: instantiate blocks and init outputs, descending into hierarchy
            self.init_graph_pass_1()
                                        
            # 2.2 init_pass_2: init inputs, again descending into hierarchy
            self.init_graph_pass_2()
                    
        ################################################################################
        # 3 initialize a primitive block
        else:
            self.init_primitive()
            
    def init_primitive(self):
        """initialize primitive block"""
        # initialize block output
        self.init_outputs()
            
        # initialize block logging
        self.init_logging()
            

    def init_subgraph(self):
        subconf = get_config_raw(self.subgraph, 'conf') # 'graph')
        assert subconf is not None
        # make sure subordinate number of steps is less than top level numsteps
        assert subconf['params']['numsteps'] <= self.top.numsteps, "enclosed numsteps = %d greater than top level numsteps = %d" % (subconf['params']['numsteps'], self.top.numsteps)

        self.conf['params']['graph'] = subconf['params']['graph']
                    
    def init_graph_pass_1(self):
        """!@brief initialize this block's graph by instantiating all graph nodes"""
        # if we're coming from non topblock init
        # self.graph = self.conf['params']['graph']

        print "{0: <20}.init_graph_pass_1 graph.keys = {1}".format(self.cname[:20], self.nxgraph.nodes())
        
        # pass 1 init
        # for k, v in self.graph.items():
        # for n in self.nxgraph.nodes_iter():
        for i in range(self.nxgraph.number_of_nodes()):
            v = self.nxgraph.node[i]
            # print "%s.init_graph_pass_1 node = %s" % (self.cname, v)
            k = v['params']['id']
            # v = n['params']
            # self.debug_print("__init__: pass 1\nk = %s,\nv = %s", (k, print_dict(v)))

            # debug timing
            print "{0: <20}.init pass 1 k = {1: >5}, v = {2: >20}".format(self.__class__.__name__[:20], k, v['block'].__name__)
            then = time.time()

            # print v['block_']
            
            # actual instantiation
            # self.graph[k]['block'] = self.graph[k]['block'](conf = v, paren = self, top = self.top)
            v['block_'] = v['block'](conf = v, paren = self, top = self.top)

            # print "%s init self.top.graph = %s" % (self.cname, self.top.graph.keys())
            
            # complete time measurement
            print "{0: <20}.init pass 1 k = {1: >5}, v = {2: >20}".format(self.__class__.__name__[:20], k, v['block_'].cname), "took %f s" % (time.time() - then)
            
            # print "%s self.graph[k]['block'] = %s" % (self.graph[k]['block'].__class__.__name__, self.graph[k]['block'].bus)
        # done pass 1 init

    def init_graph_pass_2(self):
        # print "bbbhhh", self.graph.keys()
        # pass 2 init
        # for k, v in self.graph.items():
        for i in range(self.nxgraph.number_of_nodes()):
            v = self.nxgraph.node[i]
            k = v['params']['id']
            
            self.debug_print("__init__: pass 2\nk = %s,\nv = %s", (k, print_dict(v)))
            # print "%s.init pass 2 k = %s, v = %s" % (self.__class__.__name__, k, v['block'].cname)
            print "{0: <20}.init pass 2 k = {1: >5}, v = {2: >20}".format(self.__class__.__name__[:20], k, v['block_'].cname),
            then = time.time()
            # self.graph[k]['block'].init_pass_2()
            v['block_'].init_pass_2()
            print "took %f s" % (time.time() - then)

        # for k, v in self.graph.items():
        #     v['block'].step()
            
    def init_outputs(self):
        # print "%s.init_outputs: inputs = %s" % (self.cname, self.inputs)
        # create outputs
        # format: variable: [shape]
        # new format: outkey = str: outval = {val: value, shape: shape, dst: destination, ...}
        for k, v in self.outputs.items():
            # print "%s.init_outputs: outk = %s, outv = %s" % (self.cname, k, v)
            assert type(v) is dict, "Old config of %s output %s with type %s, %s" % (self.id, k, type(v), v)
            # create new shape tuple by appending the blocksize to original dimensions
            v['bshape']  = v['shape'] + (self.blocksize,)
            # print "v.bshape", v['bshape']

            # compute buskey from id and variable name
            v['buskey'] = "%s/%s" % (self.id, k)

            # logging by output item
            if not v.has_key('logging'):
                v['logging'] = True
                
            # set self attribute to that shape
            setattr(self, k, np.zeros(v['bshape']))
            
            # print "%s.init_outputs: %s.bus[%s] = %s" % (self.cname, self.id, v['buskey'], getattr(self, k).shape)
            self.bus[v['buskey']] = getattr(self, k)
            # self.bus.setval(v['buskey'], getattr(self, k))

            # output item initialized
            v['init'] = True

    def init_logging(self):
        # initialize block logging
        if not self.logging: return

        # assume output's initialized        
        for k, v in self.outputs.items():
            if (not v.has_key('init')) or (not v['init']) or (not v['logging']): continue

            tbl_columns_dims = "_".join(["%d" for axis in v['shape']])
            tbl_columns = [tbl_columns_dims % tup for tup in xproduct(itertools.product, v['shape'])]
            # print "tbl_columns", tbl_columns

            # initialize the log table for this block
            log.log_pd_init_block(
                tbl_name    = v['buskey'], # "%s/%s" % (self.id, k),
                tbl_dim     = np.prod(v['shape']), # flattened dim without blocksize
                tbl_columns = tbl_columns,
                numsteps    = self.top.numsteps,
                blocksize   = self.blocksize,
            )
                
        # # FIXME: make one min_blocksize bus group for each node output
        # for outkey, outparams in self.nodes[nk].outputs.items():
        #     nodeoutkey = "%s/%s" % (nk, outkey)
        #     print "bus %s, outkey %s, odim = %d" % (nk, nodeoutkey, outparams[0])
        #     self.bus[nodeoutkey] = np.zeros((self.nodes[nk].odim, 1))
                                
    def init_pass_2(self):
        """second init pass which needs to be done after all outputs have been initialized"""
        if not self.topblock:
            # create inputs by mapping from constants or bus
            # that's actually for pass 2 to enable recurrent connections
            # old format: variable: [buffered const/array, shape, bus]
            # new format: variable: {'val': buffered const/array, 'shape': shape, 'src': bus|const|generator?}
            for k, v in self.inputs.items():
                self.debug_print("__init__: pass 2\n    in_k = %s,\n    in_v = %s", (k, v))
                assert len(v) > 0
                assert type(v) is dict, "input value %s in block %s/%s must be a dict but it is a %s, probably old config" % (k, self.cname, self.id, type(v))

                # set input from bus
                if v.has_key('bus'):
                    if v.has_key('shape'):
                        # init input buffer from configuration shape
                        # print "input config shape = %s" % (v['shape'][:-1],)
                        if len(v['shape']) == 1:
                            vshape = (v['shape'][0], self.ibuf,)
                        else:
                            vshape = v['shape'][:-1] + (self.ibuf,)
                        if not self.bus.has_key(v['bus']):
                            v['val'] = np.zeros(vshape) # ibuf >= blocksize
                            self.bus[v['bus']] = v['val']
                            inbus = self.bus[v['bus']]
                        else:
                            inbus = self.bus[v['bus']]
                            v['val'] = inbus
                            # v['val'][...,0:inbus.shape[-1]] = inbus
                        print "v['val'].shape", v['val'].shape
                        
                    elif not v.has_key('shape'):
                        # check if key exists or not. if it doesn't, that means this is a block inside dynamical graph construction
                        assert self.bus.has_key(v['bus']), "Requested bus item %s is not in buskeys %s" % (v['bus'], self.bus.keys())
                    
                        # enforce bus blocksize smaller than local blocksize, tackle later
                        assert self.bus[v['bus']].shape[-1] <= self.blocksize, "input block size needs to be less than or equal self blocksize in %s/%s\ncheck blocksize param" % (self.cname, self.id)
                        # get shortcut
                        inbus = self.bus[v['bus']]
                        # print "init_pass_2 inbus.sh = %s" % (inbus.shape,)
                        # if no shape given, take busdim times input buffer size
                        v['val'] = np.zeros(inbus.shape[:-1] + (self.ibuf,)) # ibuf >= blocksize inbus.copy()
                                        
                    v['shape'] = v['val'].shape # self.bus[v['bus']].shape
                    
                    self.debug_print("%s.init_pass_2: %s, bus[%s] = %s, input = %s", (self.cname, self.id, v['bus'], self.bus[v['bus']].shape, v['val'].shape))
                # elif type(v[0]) is str:
                #     # it's a string but no valid buskey, init zeros(1,1)?
                #     if v[0].endswith('.h5'):
                #         setattr(self, k, v[0])
                else:
                    assert v.has_key('val'), "Input spec needs either a 'bus' or 'val' entry in %s" % (v.keys())
                    # expand scalar to vector
                    if np.isscalar(v['val']):
                        # print "isscalar", v['val']
                        # check for shape info
                        # if len(v['val']) == 1: # one-element vector
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
    def step(self, x = None):
        """Base block step function

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

        if self.topblock:
            # then log all
            # for k, v in self.graph.items(): # self.bus.items():
            for i in range(self.nxgraph.number_of_nodes()):
                v = self.nxgraph.node[i]
                k = v['params']['id']
                # return if block doesn't want logging
                if not v['block_'].logging: continue
                # if (self.cnt % v['block'].blocksize) != (v['block'].blocksize - 1): continue
                if (self.cnt % v['block_'].blocksize) > 0: continue
                # print debug foo
                self.debug_print("step: node k = %s, v = %s", (k, v))
                # do logging for all of the node's output variables
                for k_o, v_o in v['block_'].outputs.items():
                    buskey = "%s/%s" % (v['block_'].id, k_o)
                    # print "%s step outk = %s, outv = %s, bus.sh = %s" % (self.cname, k_o, v_o, self.bus[buskey].shape)
                    log.log_pd(tbl_name = buskey, data = self.bus[buskey])

            # store log
            if (self.cnt) % 500 == 0 or self.cnt == (self.numsteps - 1):
                print "storing log @iter %04d" % (self.cnt)
                log.log_pd_store()
                
        # need to to count ourselves
        self.cnt += 1

    def get_config(self):
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

class FuncBlock2(Block2):
    """!@brief Function block: wrap the function given by the configuration in params['func'] in a block"""
    def __init__(self, conf = {}, paren = None, top = None):
        Block2.__init__(self, conf = conf, paren = paren, top = top)

        self.check_attrs(['func'])

        # FIXME: check return type of func at init time and set step function

    @decStep()
    def step(self, x = None):
        # print "%s.step inputs[%d] = %s" % (self.cname, self.cnt, self.inputs)
        # self.inputs is a dict with values [array]
        assert self.inputs.has_key('x'), "%s.inputs expected to have key 'x' with function input vector. Check config."

        self.debug_print("step[%d]: x = %s", (self.cnt, self.inputs,))
        f_val = self.func(self.inputs)
        if type(f_val) is dict:
            for k, v in f_val.items():
                setattr(self, k, v)
        else:
            self.y = f_val
        self.debug_print("step[%d]: y = %s", (self.cnt, self.y,))
            
class LoopBlock2(Block2):
    """!@brief Loop block: dynamically create block variations according to some specificiations of variation

    Two loop modes in this framework:
     - parallel mode (LoopBlock2): modify the graph structure and all block variations at the same time
     - sequential mode (SeqLoopBlock2): modify execution to run each variation one after the other
     """
    def __init__(self, conf = {}, paren = None, top = None):
        self.defaults['loop'] = [1]
        # self.defaults['loopmode'] = 'sequential'
        self.defaults['loopblock'] = {}
        Block2.__init__(self, conf = conf, paren = paren, top = top)

        # loopblocks = []
        # # loop the loop
        # for i, lparams in enumerate(self.loop):
        #     print "lparams", lparams, "self.loopblock['params']", self.loopblock['params']
            
        #     # copy params
        #     loopblock_params = {}
        #     for k, v in self.loopblock['params'].items():
        #         if k == 'id':
        #             loopblock_params[k] = "%s_%d" % (self.id, i+1)
        #         # FIXME: only works for first item
        #         elif k == lparams[0]:
        #             loopblock_params[k] = lparams[1]
        #         else:
        #             loopblock_params[k] = v

        #     # create dynamic conf
        #     loopblock_conf = {'block': self.loopblock['block'], 'params': loopblock_params}
        #     # instantiate block
        #     dynblock = self.loopblock['block'](conf = loopblock_conf,
        #                                        paren = self.paren, top = self.top)
        #     # print "loopblock dynblock = %s" % (dynblock.cname, )

        #     # get config and store it
        #     dynblockconf = dynblock.get_config()
        #     dynblockconf[1]['block'] = dynblock

        #     # append to list of dynamic blocks
        #     loopblocks.append(dynblockconf)
        #     # print print_dict(self.top.graph)

        # # # debug loopblocks in LoopBlock2 init
        # # for item in loopblocks:
        # #     print "%s.__init__ loopblocks = %s: %s" % (self.__class__.__name__, item[0], print_dict(item[1]))

        # # print "loopblocks", self.id, loopblocks
                
        # # FIXME: this good?
        # # insert dynamic blocks into existing ordered dict
        # # print "topgraph", print_dict(self.top.graph)
        # print "%s-%s.init swong topgrah = %s" % (self.cname, self.id, self.top.graph.keys()) # , self.graph.keys()
        # ordereddict_insert(ordereddict = self.top.graph, insertionpoint = '%s' % self.id, itemstoadd = loopblocks)
        # print "%s-%s.init swong topgrah = %s" % (self.cname, self.id, self.top.graph.keys()) # , self.graph.keys()
        # for k, v in self.top.graph.items():
        #     print "k", k, "v", v['block']

        # print "top graph", print_dict(self.top.graph)
        # print "top graph", self.top.graph.keys()
        # print "top graph", print_dict(self.top.graph[self.top.graph.keys()[0]])

        # replace loopblock block entry in original config, propagated back to the top / global namespace
        # self.loopblock['block'] = Block2.__class__.__name__
                   
    def step(self, x = None):
        """loopblock2.step: loop over self.nxgraph items and step them"""
        # pass
        for i in range(self.nxgraph.number_of_nodes()):
            # print "node %d" % (i,)
            v = self.nxgraph.node[i]
            k = v['params']['id']

            # print "k, v", k, v
            
            # for k, v in self.graph.items():
            v['block_'].step()

            # buskey = "%s/x" % v['block_'].id
            # buskey2 = v['block_'].outputs['x']['buskey']
            # print "constblock", k, buskey, buskey2, v['block_'].x.flatten()
            # print self.bus[buskey], self.bus.keys()
            # print self.top.bus[buskey], self.top.bus.keys()
            # print self.bus[v['params']]

class SeqLoopBlock2(Block2):
    """!@brief Sequential loop block"""
    def __init__(self, conf = {}, paren = None, top = None):
        self.defaults['loop'] = [1]
        # self.defaults['loopmode'] = 'sequential'
        self.defaults['loopblock'] = {}
        Block2.__init__(self, conf = conf, paren = paren, top = top)

        self.init_primitive()
        
        # check 'loop' parameter type and set the loop function
        if type(self.loop) is list: # it's a list
            self.f_loop = self.f_loop_list
            # assert len(self.loop) == (self.numsteps/self.blocksize), "%s step numsteps / blocksize (%s/%s = %s) needs to be equal the loop length (%d)" % (self.cname, self.numsteps, self.blocksize, self.numsteps/self.blocksize, len(self.loop))
        else: # it's a func
            self.f_loop = self.f_loop_func

    # loop function for self.loop = list 
    def f_loop_list(self, i, f_obj):
        results = f_obj(self.loop[i])
        return results

    # loop function for self.loop = func
    def f_loop_func(self, i, f_obj):
        results = self.loop(self, i, f_obj)
        return results
            
    @decStep()
    def step(self, x = None):

        def f_obj(lparams):
            """instantiate the loopblock and run it"""
            # print "f_obj lparams", lparams
            # copy params
            loopblock_params = {}
            for k, v in self.loopblock['params'].items():
                if k == 'id':
                    loopblock_params[k] = "%s_%d" % (self.id, i+1)
                elif k == lparams[0]:
                    loopblock_params[k] = lparams[1]
                else:
                    loopblock_params[k] = v

            # print "%s.step.f_obj: loopblock_params = %s" % (self.cname, loopblock_params)
            # create dynamic conf
            loopblock_conf = {'block': self.loopblock['block'], 'params': loopblock_params}
            # instantiate block
            self.dynblock = self.loopblock['block'](conf = loopblock_conf,
                                               paren = self.paren, top = self.top)
            # second pass
            self.dynblock.init_pass_2()

            # run the block
            for j in range(self.dynblock.blocksize):
                # print "%s trying %s.step[%d]" % (self.cname, dynblock.cname, j)
                self.dynblock.step()
        
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

        if type(self.loop) is list:
            f_obj_ = f_obj
        else:
            f_obj_ = f_obj_hpo

        then = time.time()
        # loop the loop
        print "%s iter#" % (self.cname,),
        for i in range(self.numsteps/self.loopblocksize):
            print "%d" % (i,),
            sys.stdout.flush()
            then = time.time()
            # dynblock = obj()
            # func: need input function from config
            results = self.f_loop(i, f_obj_)#_hpo)
            if results is not None:
                for k, v in results.items():
                    setattr(self, k, v)
            # dynblock = results['dynblock']
            # lparams = results['lparams']
            
            # print "%s.steploop[%d], %s, %s" % (self.cname, i, lparams, self.loopblock['params'])
            # for k,v in dynblock.outputs.items():
            #     print "dynout", getattr(dynblock, k)

            for outk in self.outputs.keys():
                # outvar = getattr(self, outk)
                # outvar[:,[i] = bla
                
                # func: need output function from config
                # FIXME: handle loopblock blocksizes greater than one
                # self.__dict__[outk][:,[i]] = np.mean(getattr(dynblock, outk), axis = 1, keepdims = True)
                outslice = slice(i*self.dynblock.blocksize, (i+1)*self.dynblock.blocksize)
                # print "self.dynblock", self.dynblock.outputs[outk], getattr(self.dynblock, outk).shape, self.dynblock.file
                # print "%s.step self.%s = %s, outslice = %s" % (self.cname, outk, self.__dict__[outk].shape, outslice, )
                # self.__dict__[outk][:,[i]] = getattr(self.dynblock, outk)
                self.__dict__[outk][:,outslice] = getattr(self.dynblock, outk)
        sys.stdout.write('\n')

        # # hack for checking hpo minimum
        # if hasattr(self, 'hp_bests'):
        #     print "%s.step: bests = %s, %s" % (self.cname, self.hp_bests[-1], f_obj_hpo(tuple([self.hp_bests[-1][k] for k in sorted(self.hp_bests[-1])])))
    
class PrimBlock2(Block2):
    """!@brief Base class for primitive blocks"""
    def __init__(self, conf = {}, paren = None, top = None):
        Block2.__init__(self, conf = conf, paren = paren, top = top)

    @decStep()
    def step(self, x = None):
        """PrimBlock2.step: step the block, decorated, blocksize boundaries"""
        # print "primblock step id %s, v = %s" % (self.id, self.x)
        pass
    
class IBlock2(PrimBlock2):
    """!@brief Integrator block: integrate input and write to output

params: inputs, outputs, leakrate    
    """
    @decInit()
    def __init__(self, conf = {}, paren = None, top = None):
        for k in conf['params']['inputs'].keys():
            # print "%s.init inkeys %s" % (self.__class__.__name__, k)
            conf['params']['outputs']["I%s" % k] = [top.bus[conf['params']['inputs'][k][0]].shape]
            
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)

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
        for ink in self.inputs.keys():
            outk = "I%s" % ink
            # input integral / cumsum
            Iin = np.cumsum(self.inputs[ink][0], axis = 1) * self.d
            # print getattr(self, outk)[:,[-1]].shape, self.inputs[ink][0].shape, Iin.shape
            setattr(self, outk, getattr(self, outk)[:,[-1]] + Iin)
            # setattr(self, outk, getattr(self, outk) + (self.inputs[ink][0] * 1.0))
            # print getattr(self, outk).shape

class dBlock2(PrimBlock2):
    """!@brief Differentiator block: compute differences of input and write to output

params: inputs, outputs, leakrate / smoothrate?
    """
    @decInit()
    def __init__(self, conf = {}, paren = None, top = None):
        """dBlock2 init"""
        for ink in conf['params']['inputs'].keys():
            # get input shape
            inshape = top.bus[conf['params']['inputs'][ink]['bus']].shape
            # inshape = conf['params']['inputs'][ink]['shape']
            print "inshape", inshape
            # alloc copy of previous input block 
            setattr(self, "%s_" % ink, np.zeros(inshape))
            # set output members
            conf['params']['outputs']["d%s" % ink] = {'shape': inshape[:-1]}
            
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
            din = np.diff(tmp_[:,tmp_sl], axis = 1) # * self.d
            # which should be same shape is input
            assert din.shape == self.inputs[ink]['val'].shape
            # print getattr(self, outk)[:,[-1]].shape, self.inputs[ink][0].shape, din.shape
            setattr(self, outk, din)
            # store current input
            setattr(self, ink_, self.inputs[ink]['val'].copy())

class DelayBlock2(PrimBlock2):
    """!@brief Delay block: delay shift input by n steps

params: inputs, delay in steps / shift
    """
    @decInit() # outputs from inputs block decorator
    def __init__(self, conf = {}, paren = None, top = None):
        """DelayBlock2 init"""
        params = conf['params']
        if not params.has_key('outputs'):
            params['outputs'] = {}
            
        for ink in params['inputs'].keys():
            # get input shape
            inshape = top.bus[params['inputs'][ink]['bus']].shape
            # print "DelayBlock2 inshape", inshape
            # alloc delay block
            # print ink, params['delays'][ink]
            setattr(self, "%s_" % ink, np.zeros((inshape[0], inshape[1] + params['delays'][ink])))
            # set output members
            params['outputs']["d%s" % ink] = {'shape': inshape[:-1]}
            
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
            # print "sl", sl
            inv_[:,sl] = self.inputs[ink]['val'].copy()
            
            # compute the diff in the input
            # din = np.diff(tmp_[:,tmp_sl], axis = 1) # * self.d
            # which should be same shape is input
            # assert din.shape == self.inputs[ink][0].shape
            # print getattr(self, outk)[:,[-1]].shape, self.inputs[ink][0].shape, din.shape
            setattr(self, outk, inv_[:,slice(0, self.blocksize)])
            # store current input
            setattr(self, ink_, np.roll(inv_, shift = -self.blocksize, axis = 1))
                        
class SliceBlock2(PrimBlock2):
    """!@brief Slice block can cut slices from the input
    """
    def __init__(self, conf = {}, paren = None, top = None):
        params = conf['params']
        if not params.has_key('outputs'):
            params['outputs'] = {}
            
        for k in params['inputs'].keys():
            slicespec = params['slices'][k]
            # print slicespec
            for slk, slv in slicespec.items():
                # print "%s.init inkeys %s" % (self.__class__.__name__, k)
                outk = "%s_%s" % (k, slk)
                if type(slv) is slice:
                    params['outputs'][outk] = {'shape': (slv.stop - slv.start, )} # top.bus[params['inputs'][k][0]].shape
                elif type(slv) is list:
                    params['outputs'][outk] = {'shape': (len(slv), )} # top.bus[params['inputs'][k][0]].shape
                elif type(slv) is tuple:
                    params['outputs'][outk] = {'shape': (slv[1] - slv[0], )} # top.bus[params['inputs'][k][0]].shape
            
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)

    @decStep()
    def step(self, x = None):
        for ink in self.inputs.keys():
            slicespec = self.slices[ink]
            for slk, slv in slicespec.items():
                outk = "%s_%s" % (ink, slk)
                setattr(self, outk, self.inputs[ink]['val'][slv])

class StackBlock2(PrimBlock2):
    """!@brief Stack block can combine input slices into a single output item
    """
    def __init__(self, conf = {}, paren = None, top = None):
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)

    @decStep()
    def step(self, x = None):
        st = [inv[0] for ink, inv in self.inputs.items()]
        # print "Stack st = %s" % ( len(st))
        self.y = np.vstack(st)
                    
class ConstBlock2(PrimBlock2):
    """!@brief Constant block: output is a constant vector
    """
    def __init__(self, conf = {}, paren = None, top = None):
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)
        
        # either column vector to be replicated or blocksize already
        assert self.x.shape[-1] in [1, self.blocksize]
        assert self.inputs['c']['val'].shape[:-1] == self.x.shape[:-1], "ConstBlock2 input / output shapes must agree: %s == %s?" % (self.inputs['c']['val'].shape[:-1], self.x.shape[:-1])
        
        # replicate column vector
        # if self.x.shape[1] == 1: # this was wrong
        if self.inputs['c']['val'].shape[1] == 1:
            self.x = np.tile(self.inputs['c']['val'], self.blocksize) # FIXME as that good? only works for single column vector
        else:
            self.x = self.inputs['c']['val'].copy() # FIXME as that good? only works for single column vector

class CountBlock2(PrimBlock2):
    """!@brief Count block: output is just the count
    """
    @decInit()
    def __init__(self, conf = {}, paren = None, top = None):
        # defaults
        self.scale  = 1
        self.offset = 0
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)
        # single output key
        self.outk = self.outputs.keys()[0]
        # init cnt_ of blocksize
        self.cnt_ = np.zeros(self.outputs[self.outk]['shape'] + (self.blocksize,))
        # print self.inputs
        # FIXME: modulo / cout range with reset/overflow
        
    @decStep()
    def step(self, x = None):
        """CountBlock step: if blocksize is 1 just copy the counter, if bs > 1 set cnt_ to range"""
        if self.blocksize > 1:
            self.cnt_[:,-self.blocksize:] = np.arange(self.cnt - self.blocksize, self.cnt).reshape(self.outputs[self.outk][0])
        else:
            self.cnt_[:,0] = self.cnt
        # FIXME: make that a for output items loop
        setattr(self, self.outk, (self.cnt_ * self.scale) + self.offset)

class UniformRandomBlock2(PrimBlock2):
    """!@brief Uniform random numbers: output is uniform random vector
    """
    @decInit()
    def __init__(self, conf = {}, paren = None, top = None):
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)
        
        # self.lo = 0
        # self.hi = 1
        # self.x = np.random.uniform(self.lo, self.hi, (self.odim, 1))
        
    @decStep()
    def step(self, x = None):
        self.debug_print("%s.step:\n\tx = %s,\n\tbus = %s,\n\tinputs = %s,\n\toutputs = %s",
                         (self.__class__.__name__,self.outputs.keys(), self.bus, self.inputs, self.outputs))
        # self.hi = x['hi']
        if self.cnt % self.rate == 0:
            # FIXME: take care of rate/blocksize issue
            for k, v in self.outputs.items():
                # x = np.random.uniform(self.inputs['lo'][0][:,[-1]], self.inputs['hi'][0][:,[-1]], (self.outputs[k][0]))
                # print 'lo', self.inputs['lo']['val'], '\nhi', self.inputs['hi']['val'], '\noutput', v['bshape']
                x = np.random.uniform(self.inputs['lo']['val'], self.inputs['hi']['val'], size = v['bshape'])
                setattr(self, k, x.copy())
            # print "self.x", self.x
        
        # # loop over outputs dict and copy them to a slot in the bus
        # for k, v in self.outputs.items():
        #     buskey = "%s/%s" % (self.id, k)
        #     self.bus[buskey] = getattr(self, k)
        # self.bus[self.id] = self.x
        return None
