"""smp_graphs - smp sensorimotor experiments as computation graphs

block: basic block of computation

2017 Oswald Berthold
"""

import uuid, sys
from collections import OrderedDict
import pickle

import numpy as np

import smp_graphs.logging as log
from smp_graphs.utils import print_dict

BLOCKSIZE_MAX = 10000

# utils
def ordereddict_insert(ordereddict = None, insertionpoint = None, itemstoadd = []):
    """self rolled ordered dict insertion
    from http://stackoverflow.com/questions/29250479/insert-into-ordereddict-behind-key-foo-inplace
    """
    assert ordereddict is not None
    new_ordered_dict = ordereddict.__class__()
    for key, value in ordereddict.items():
        new_ordered_dict[key] = value
        if key == insertionpoint:
            for item in itemstoadd:
                keytoadd, valuetoadd = item
                new_ordered_dict[keytoadd] = valuetoadd
    ordereddict.clear()
    ordereddict.update(new_ordered_dict)
    return ordereddict
    


################################################################################
# v2 blocks

# some decorators
class decInit():
    def __call__(self, f):
        def wrap(exec_self, *args, **kwargs):
            f(exec_self, *args, **kwargs)
        return wrap
            

class decStep():
    """step decorator"""
    def __call__(self, f):
        def wrap(exec_self, *args, **kwargs):
            # # get any input or set to default
            # if len(args) > 0:
            #     x = np.atleast_2d(args[0])
            # elif kwargs.has_key('x'):
            #     x = np.atleast_2d(kwargs['x'])
            # else:
            #     x = None

            # input argument above might be obsolete due to busses
            # get the input from the bus
            if True:
                # stack = [exec_self.bus[k] for k in exec_self.inputs]
                # print "stack", stack
                # x = np.vstack(stack)
                # print "x", x
                x = {}
                for k, v in exec_self.inputs.items():
                        
                    # # print type(v)
                    # if np.isscalar(v):
                    #     print "%s.decstep scalar" % (exec_self.__class__.__name__), k, v, getattr(exec_self, k)
                    #     x[k] = getattr(exec_self, k) # v
                    # elif type(v) is list:
                    #     print "%s.decstep list" % (exec_self.__class__.__name__), k, v
                    #     stack = [exec_self.bus[bk] for bk in v]
                    #     # x[k] = stack
                    #     x[k] = np.vstack(stack)
                    # elif type(v) is dict:
                    #     print "%s.decstep dict" % (exec_self.__class__.__name__), k, v
                    #     stack = [exec_self.bus[bk][bv[0]:bv[1]]
                    #                  for bk,bv in v.items()]
                    #     # x[k] = stack
                    #     x[k] = np.vstack(stack)
                    # elif type(v) is np.ndarray:
                    #     print "%s.decstep array" % (exec_self.__class__.__name__), k, v
                    #     x[k] = v

                    # print "%s.decstep x[%s] = %s" % (exec_self.__class__.__name__, k, x[k])

                    # print "decStep: x.shape = %s" % (x.shape,)
            
                    # write to input buffer
                    # if x.shape == (exec_self.idim, 1):
                    # for k, v 
                    if v[2] is not None:
                        if exec_self.ibuf > 1:
                            exec_self.inputs[k][0] = np.roll(exec_self.inputs[k][0], shift = -1, axis = 1)
                        # print "shapes", "ibuf", \
                        #   exec_self.bufs['ibuf'][k][:,-1,np.newaxis].shape,\
                        #   'x[k]', x[k].shape
                        exec_self.inputs[k][0][:,-1,np.newaxis] = exec_self.bus[v[2]]

            # call the function
            f_out = f(exec_self, x)

            # copy output to bus
            for k, v in exec_self.outputs.items():
                buskey = "%s/%s" % (exec_self.id, k)
                exec_self.bus[buskey] = getattr(exec_self, k)
            
            # count calls
            exec_self.cnt += 1
            # exec_self.ibufidx = exec_self.cnt % exec_self.ibufsize
            
            return f_out
        # return the new func
        return wrap

################################################################################
# base blocks
class Block2(object):
    defaults = {
        'id': None,
        'debug': False,
        'topblock': False,
        'ibuf': 1,
        'cnt': 0,
        'blocksize': 1,
        'inputs': {}, # internal port, scalar / vector/ bus key, [slice]
        'outputs': {}, # name, dim
        'logging': True, # normal logging
        # 'idim': None,
        # 'odim': None,
        # # 'obufsize': 1,
        # # 'savedata': True,
        # 'ros': False,
    }
        
    def __init__(self, conf = {}, paren = None, top = None):
        self.conf = conf
        self.paren = paren
        self.top = top
        self.cname = self.__class__.__name__
        
        # load defaults
        for k,v in self.defaults.items():
            self.__dict__[k] = v
                    
        # fetch existing configuration arguments
        if type(self.conf) == dict:
            for k,v in self.conf['params'].items():
                # self.__dict__[k] = v
                setattr(self, k, v)

        # input buffer vs. blocksize: input buffer is sliding, blocksize is jumping
        if self.blocksize > self.ibuf:
            self.ibuf = self.blocksize

        if self.topblock:
            # pass 1: complete config with runtime info
            # init global messaging bus
            self.bus = {}

            log.log_pd_init(self.conf)

            # init pass 1: complete the graph by expanding dynamic variables and initializing the outputs to get the bus def
            self.init_graph_pass_1()

            # init pass 2:
            self.init_graph_pass_2()
            
            self.debug_print("self.bus = %s", (print_dict(self.bus),))
                            
        else:
            # pass 1: complete config with runtime info
            # get bus
            self.bus = self.top.bus

            self.init_outputs()
            
            # TODO: init logging
            self.init_logging()

    def init_graph_pass_1(self):
        self.graph = self.conf['params']['graph']
        # pass 1 init
        for k, v in self.graph.items():
            self.debug_print("__init__: pass 1\nk = %s,\nv = %s", (k, print_dict(v)))
            self.graph[k]['block'] = self.graph[k]['block'](conf = v, paren = self, top = self)
            print "%s self.graph[k]['block'] = %s" % (self.graph[k]['block'].__class__.__name__, self.graph[k]['block'].bus)
        # done pass 1 init

    def init_graph_pass_2(self):
        # pass 2 init
        for k, v in self.graph.items():
            self.debug_print("__init__: pass 2\nk = %s,\nv = %s", (k, print_dict(v)))
            self.graph[k]['block'].init_pass_2()

    def init_outputs(self):
        # create outputs
        # format: variable: [shape]
        for k, v in self.outputs.items():
            # create self attribute for output item
            setattr(self, k, np.zeros(v[0]))
            buskey = "%s/%s" % (self.id, k)
            self.bus[buskey] = getattr(self, k)

    def init_logging(self):
        # initialize block logging
        if not self.logging: return
        
        for k, v in self.outputs.items():
            log.log_pd_init_block(
                tbl_name = "%s/%s" % (self.id, k),
                tbl_dim = v[0], # odim
                tbl_columns = ["%s_%d" % (k, col) for col in range(v[0][0])],
                numsteps = self.top.numsteps
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
            # format: variable: [buffered const/array, shape, bus, localbuf]
            for k, v in self.inputs.items():
                self.debug_print("__init__: pass 2\nk = %s,\nv = %s", (k, print_dict(v)))
                assert len(v) > 0
                # set input from bus
                if type(v[0]) is str:
                    tmp = [None for i in range(3)]
                    tmp[0] = self.bus[v[0]]
                    tmp[1] = tmp[0].shape
                    tmp[2] = v[0]
                    self.inputs[k] = tmp
                else:
                    # expand scalar to vector
                    if np.isscalar(self.inputs[k][0]):
                        # check for shape info
                        if len(self.inputs[k]) == 1:
                            self.inputs[k].append((1,1))
                        # create ones multiplied by constant
                        self.inputs[k][0] = np.ones(self.inputs[k][1]) * self.inputs[k][0]
                    else:
                        # write array shape back into config
                        self.inputs[k] += [self.inputs[k][0].shape]
                    self.inputs[k].append(None)
                # add input buffer
                self.inputs[k][0] = np.hstack((np.zeros((self.inputs[k][1][0], self.ibuf-1)), self.inputs[k][0]))
            
    def debug_print(self, fmtstring, data):
        """only print if debug is enabled for this block"""
        fmtstring = "\n%s." + fmtstring
        data = (self.cname,) + data
        if self.debug:
            print fmtstring % data

    # undecorated step, need to count ourselves
    def step(self, x = None):
        if self.topblock:
            for k, v in self.graph.items():
                v['block'].step()

            # do logging for all bus items
            # for k, v in self.bus.items():
            for k, v in self.graph.items():
                if not v['block'].logging: continue
                
                self.debug_print("%s.step: bus k = %s, v = %s", (self.__class__.__name__, k, v))
                # print "%s id" % (self.__class__.__name__), self.id, "k", k, v.shape, v
                for k_o, v_o in v['block'].outputs.items():
                    buskey = "%s/%s" % (v['block'].id, k_o)
                    log.log_pd(tbl_name = buskey, data = self.bus[buskey])

            # store log
            if (self.cnt+1) % 100 == 0:
                print "storing log @iter %04d" % (self.cnt)
                log.log_pd_store()

            self.cnt += 1

    def get_config(self):
        params = {}
        for k, v in self.__dict__.items():
            # FIXME: include bus, top, paren?
            if k not in ['conf', 'bus', 'top', 'paren']:
                params[k] = v
            
        conf = ('%s' % self.id,
                {
                    'block': self.__class__,
                    'params': params
                })
        return conf

class LoopBlock2(Block2):
    """Loop block: dynamically create block variations according to some specificiations of variation

    two modes:
     - parallel mode: modify the graph structure and all block variations at the same time
     - sequential mode: modify execution to run each variation in one after the other
     """

    def __init__(self, conf = {}, paren = None, top = None):
        self.defaults['loop'] = [1]
        self.defaults['loopmode'] = 'sequential'
        self.defaults['loopblock'] = {}
        Block2.__init__(self, conf = conf, paren = paren, top = top)

        loopblocks = []
        for i, lparams in enumerate(self.loop):
            # print lparams, self.loopblock['params']
            
            # copy params
            loopblock_params = {}
            for k, v in self.loopblock['params'].items():
                if k == 'id':
                    loopblock_params[k] = "%s-%d" % (self.id, i+1)
                elif k == lparams[0]:
                    loopblock_params[k] = lparams[1]
                else:
                    loopblock_params[k] = v

            loopblock_conf = {'block': self.loopblock['block'], 'params': loopblock_params}
            dynblock = self.loopblock['block'](conf = loopblock_conf,
                                               paren = self.paren, top = self.top)
            
            # dynblockparams = {}
            # for k, v in self.loopblock['params']
            
            dynblockconf = dynblock.get_config()
            dynblockconf[1]['block'] = dynblock
            
            loopblocks.append(dynblockconf)
            # print print_dict(self.top.graph)
            
        for item in loopblocks:
            print "%s.__init__ loopblocks = %s: %s" % (self.__class__.__name__, item[0], print_dict(item[1]))

        # FIXME: this good?
        ordereddict_insert(ordereddict = self.top.graph, insertionpoint = '%s' % self.id, itemstoadd = loopblocks)
            
    def step(self, x = None):
        """loop block does nothing for now"""
        pass

class PrimBlock2(Block2):
    def __init__(self, conf = {}, paren = None, top = None):
        Block2.__init__(self, conf = conf, paren = paren, top = top)

class ConstBlock2(PrimBlock2):
    @decInit()
    def __init__(self, conf = {}, paren = None, top = None):
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)
        self.x = self.inputs['c'][0]

    @decStep()
    def step(self, x = None):
        self.debug_print("%s.step: x = %s, bus = %s", (self.__class__.__name__, self.x, self.bus))

        # self.x = self.in        
        # # loop over outputs dict and copy them to a slot in the bus
        # for k, v in self.outputs.items():
        #     buskey = "%s/%s" % (self.id, k)
        #     self.bus[buskey] = getattr(self, k)
            
        return self.x
    
class UniformRandomBlock2(PrimBlock2):
    @decInit()
    def __init__(self, conf = {}, paren = None, top = None):
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)
        
        # self.lo = 0
        # self.hi = 1
        # self.x = np.random.uniform(self.lo, self.hi, (self.odim, 1))

    @decStep()
    def step(self, x = None):
        self.debug_print("%s.step:\n\tx = %s,\n\tbus = %s,\n\tinputs = %s,\n\toutputs = %s", (self.__class__.__name__, self.x, self.bus, self.inputs,
                                                                    self.outputs))
        # self.hi = x['hi']
        self.x = np.random.uniform(self.inputs['lo'][0][:,[-1]], self.inputs['hi'][0][:,[-1]], (self.outputs['x'][0]))
        # loop over outputs dict and copy them to a slot in the bus
        for k, v in self.outputs.items():
            buskey = "%s/%s" % (self.id, k)
            self.bus[buskey] = getattr(self, k)
        # self.bus[self.id] = self.x
        return self.x
        
# # some decorators
# class decInit():
#     def __call__(self, f):
#         def wrap(exec_self, *args, **kwargs):
#             f(exec_self, *args, **kwargs)
#             # write dynamic changes back to config
#             if type(kwargs['conf']) == dict: # and kwargs['conf'].has_key('graph'):
#                 for k,v in kwargs['conf'].items():
#                     # print "%s k = %s, v = %s" % (exec_self.__class__.__name__, k, v)
#                     kwargs['conf'][k] = exec_self.__dict__[k]
#                     # print "xxx", k, v
#                 # print "%s, conf = %s" % (exec_self.__class__.__name__, kwargs['conf'])
#         return wrap
            

# class decStep():
#     """step decorator"""
#     def __call__(self, f):
#         def wrap(exec_self, *args, **kwargs):
#             # get any input or set to default
#             if len(args) > 0:
#                 x = np.atleast_2d(args[0])
#             elif kwargs.has_key('x'):
#                 x = np.atleast_2d(kwargs['x'])
#             else:
#                 x = None

#             # input argument above might be obsolete due to busses
#             # get the input from the bus
#             if exec_self.idim is not None:
#                 # stack = [exec_self.bus[k] for k in exec_self.inputs]
#                 # print "stack", stack
#                 # x = np.vstack(stack)
#                 # print "x", x
#                 x = {}
#                 for k, v in exec_self.inputs.items():
#                     # print type(v)
#                     if np.isscalar(v):
#                         print "%s.decstep scalar" % (exec_self.__class__.__name__), k, v, getattr(exec_self, k)
#                         x[k] = getattr(exec_self, k) # v
#                     elif type(v) is list:
#                         print "%s.decstep list" % (exec_self.__class__.__name__), k, v
#                         stack = [exec_self.bus[bk] for bk in v]
#                         # x[k] = stack
#                         x[k] = np.vstack(stack)
#                     elif type(v) is dict:
#                         print "%s.decstep dict" % (exec_self.__class__.__name__), k, v
#                         stack = [exec_self.bus[bk][bv[0]:bv[1]]
#                                      for bk,bv in v.items()]
#                         # x[k] = stack
#                         x[k] = np.vstack(stack)
#                     elif type(v) is np.ndarray:
#                         print "%s.decstep array" % (exec_self.__class__.__name__), k, v
#                         x[k] = v

#                     print "%s.decstep x[%s] = %s" % (exec_self.__class__.__name__, k, x[k])

#                     # print "decStep: x.shape = %s" % (x.shape,)
            
#                     # write to input buffer
#                     # if x.shape == (exec_self.idim, 1):
#                     # for k, v 
#                     if exec_self.ibufsize > 1:
#                         exec_self.bufs['ibuf'][k] = np.roll(exec_self.bufs['ibuf'][k], shift = -1, axis = 1)
#                     print "shapes", "ibuf", \
#                       exec_self.bufs['ibuf'][k][:,-1,np.newaxis].shape,\
#                       'x[k]', x[k].shape
#                     exec_self.bufs['ibuf'][k][:,-1,np.newaxis] = x[k]

#             # call the function
#             f_out = f(exec_self, x)
            
#             # count calls
#             exec_self.cnt += 1
#             exec_self.ibufidx = exec_self.cnt % exec_self.ibufsize
            
#             return f_out
#         # return the new func
#         return wrap
    
class Block(object):
    """smp_graphs block base class

handles both primitive and composite blocks
 - a primitive block directly implements a computation in it's step function
 - a composite block consists of a list of other blocks, e.g. the top level experiment, or a robot
    """
    
    defaults = {
        'id': None,
        'idim': None,
        'odim': None,
        'ibufsize': 1,
        'blocksize': 1,
        # 'obufsize': 1,
        'logging': True, # normal logging
        # 'savedata': True,
        'topblock': False,
        'ros': False,
        'debug': False,
        'inputs': {}, # internal port, scalar / vector/ bus key, [slice]
        'outputs': {'x': [1]} # name, dim
    }

    # @decInit()
    def __init__(self, block = None, conf = None, bus = None):
        # fetch default block config
        for k,v in self.defaults.items():
            self.__dict__[k] = v
            
        self.debug_print("%s.__init__: conf = %s", (self.__class__.__name__, conf))
        # fetch configuration arguments if block is primitive and conf is a dict 
        if type(conf) == dict:
            for k,v in conf.items():
                self.__dict__[k] = v

        # auto-generate id if None supplied
        if self.id is None:
            # self.id = self.__class__.__name__ + "_%s" % uuid.uuid4()
            self.id = self.__class__.__name__ + "_%s", (uuid.uuid1().int>>64)

        # count steps
        self.cnt = 0

        # minimum blocksize downstairs
        self.blocksize_min = BLOCKSIZE_MAX

        # global bus (SMT)
        self.bus = bus
        # block's nodes
        self.nodes = None

        # print "init", conf
        
        # composite node: init sub-nodes
        if type(conf) is dict and conf.has_key('graph'):
            
            # topblock is special, connecting experiment with the outside
            if self.topblock:
                # initialize global signal bus
                self.bus = {}
                # initialize global logging
                log.log_pd_init(conf)
            
            # pass 1 compute all missing numbers
            for i, node in enumerate(conf['graph'].items()):
                nk, nv = node
                # print "key: %s, val = %s" % (nk, nv)
                self.debug_print("node[%d] = %s(%s)", (i, nv['block'], nv['params']))
                # s = "%s(%s)", (node_val["block"], node_val["params"])
                # self.nodes[node_key] = node_val['block'](node_val['params'])
                assert nk == nv['params']['id']
                # nodekey =  # self.id # 'n%04d' % i
                self.debug_print("nodekey = %s", (nk))
                # self.nodes[nodekey] = Block(block = nv['block'], conf = nv['params'])
                # blocksize check over entire graph
                if nv['blocksize'] < self.blocksize_min:
                    self.blocksize_min = self.nodes[nk].blocksize

                
                    
            # print "%s.init: conf = %s"  %(self.__class__.__name__, conf)

            # done, all block added to top block, now reiterate
            # for i, node in enumerate(self.nodes.items()):
            #     print "%s.init nodes.item = %s" % (self.__class__.__name__, node)
            #     nk, nv = node
            
            # node dictionary
            self.nodes = OrderedDict()
            
            # pass 2 instantiate blocks in config
            for i, node in enumerate(conf['graph'].items()):
                nk, nv = node

                # create node
                self.nodes[nk] = nv['block'](
                    block = nv['block'],
                    conf = nv['params'],
                    bus = self.bus)

                # initialize block logging
                for k, v in self.nodes[nk].outputs.items():
                    log.log_pd_init_block(
                        tbl_name = "%s/%s" % (self.nodes[nk].id, k),
                        tbl_dim = v[0], # odim
                        tbl_columns = ["out_%d" % col for col in range(v[0])],
                                        numsteps = self.numsteps)
                
                # FIXME: make one min_blocksize bus group for each node output
                for outkey, outparams in self.nodes[nk].outputs.items():
                    nodeoutkey = "%s/%s" % (nk, outkey)
                    print "bus %s, outkey %s, odim = %d" % (nk, nodeoutkey, outparams[0])
                    self.bus[nodeoutkey] = np.zeros((self.nodes[nk].odim, 1))
                    
                # initialize local buffers
                self.nodes[nk].init_bufs()
            
        # atomic block
        elif type(conf) is dict and block is not None:
            self.debug_print("block is %s, nothing to do", (block))
            # self.nodes['n0000'] = self
            # self.step = step_funcs[block]            

    def init_bufs(self):
        self.bufs = {}
        print "%s self.bus" % (self.__class__.__name__), self.bus
        # check for blocksize argument, ibuf needs to be at least of size blocksize
        self.ibufsize = max(self.ibufsize, self.blocksize)
        self.ibufidx = self.cnt
        # current index
        if not self.idim is None:
            self.bufs['ibuf'] = {}
            for k,v in self.inputs.items():
                # self.bufs['ibuf'] = np.zeros((self.idim, self.ibufsize))
                idim = 0
                if np.isscalar(v):
                    idim = 1
                elif type(v) is list:
                    for bk in v:
                        idim += self.bus[bk].shape[0]
                elif type(v) is dict:
                    idim = 0
                    for bk,bv in v.items():
                        idim += self.bus[bk][bv[0]:bv[1]].shape[0]
                elif type(v) is np.ndarray:
                    idim = v.shape[0]

                print "init_bufs", k, v, idim
                setattr(self, k, np.zeros((idim, self.ibufsize)))
                self.bufs['ibuf'][k] = getattr(self, k)
            # self.bufs['ibuf'][k] = np.zeros((self.idim, self.ibufsize))
            # self.bufs["obuf"] = np.zeros((self.odim, self.obufsize))
            
    @decStep()
    def step(self, x = {}):
        self.debug_print("%s-%s.step: x = %s", (self.__class__.__name__, self.id, x))
        # iterate all nodes and step them
        if self.nodes is not None:
            for k,v in self.nodes.items():
                x_ = v.step(x = x)
        else:
            # default action: copy input to output / identity
            x_ = x

        if self.topblock:
            # do logging for all bus items
            for k, v in self.bus.items():
                self.debug_print("%s.step: bus k = %s, v = %s, %s", (self.__class__.__name__, k, v.shape))
                # print "%s id" % (self.__class__.__name__), self.id, "k", k, v.shape, v
                log.log_pd(nodeid = k, data = v)

        # store log
        if (self.cnt+1) % 100 == 0:
            log.log_pd_store()

        return(x_)

    def save(self, filename):
        """save this block into a pickle"""
        pass

    def load(self, filename):
        """load block from a pickle"""
        pass

    def debug_print(self, fmtstring, data):
        """only print if debug is enabled for this block"""
        if self.debug:
            print fmtstring % data

################################################################################
# Simple blocks for testing

class ConstBlock(Block):
    @decInit()
    def __init__(self, block = None, conf = None, bus = None):
        Block.__init__(self, block = block, conf = conf, bus = bus)
        self.x = np.ones((self.odim, 1)) * self.const

    @decStep()
    def step(self, x = None):
        self.debug_print("%s.step: x = %s, bus = %s", (self.__class__.__name__, x, self.bus))
        # loop over outputs dict and copy them to a slot in the bus
        for k, v in self.outputs.items():
            buskey = "%s/%s" % (self.id, k)
            self.bus[buskey] = getattr(self, k)
        return self.x

class UniformRandomBlock(Block):
    @decInit()
    def __init__(self, block = None, conf = None, bus = None):
        Block.__init__(self, block = block, conf = conf, bus = bus)
        # self.lo = 0
        # self.hi = 1
        self.x = np.random.uniform(self.lo, self.hi, (self.odim, 1))

    @decStep()
    def step(self, x = None):
        self.debug_print("%s.step: x = %s, bus = %s, inputs = %s", (self.__class__.__name__, x, self.bus, self.inputs))
        self.hi = x['hi']
        self.x = np.random.uniform(self.lo, self.hi, (self.odim, 1))
        # loop over outputs dict and copy them to a slot in the bus
        for k, v in self.outputs.items():
            buskey = "%s/%s" % (self.id, k)
            self.bus[buskey] = getattr(self, k)
        # self.bus[self.id] = self.x
        return self.x

# File reading

def read_puppy_hk_pickles(lfile, key = None):
    """read pickled log dicts from andi's puppy experiments"""
    d = pickle.load(open(lfile, 'rb'))
    # print "d.keys", d.keys()
    # data = d["y"][:,0,0] # , d["y"]
    rate = 20
    offset = 0
    # data = np.atleast_2d(data).T
    # print "wavdata", data.shape
    data = d
    x = d['x']
    # special treatment for x with addtional dimension
    data['x'] = x[:,:,0]
    # print "x.shape", data['x'].shape
    return (data, rate, offset)
    
class FileBlock(Block):
    @decInit()
    def __init__(self, block = None, conf = None, bus = None):
        Block.__init__(self, block = block, conf = conf, bus = bus)
        # multiple files: concat? block manipulation blocks?
        self.file = []
        # auto odim
        # if self.odim == 'auto':
        lfile = conf['file'][0]
        # puppy homeokinesis (andi)
        if lfile.startswith('data/pickles_puppy') and lfile.endswith('.pickle'):
            (self.data, self.rate, self.offset) = read_puppy_hk_pickles(lfile)
            # setattr(self, 'x', self.data['x'])
            # setattr(self, 'x', self.data['x'])

        # init states
        for key, v in self.outputs.items(): # ['x', 'y']:
            # print "key", self.data[key]
            setattr(self, key, np.zeros((self.data[key].shape[1], 1)))
            # self.x = np.zeros((self.odim, 1))
        
        # set odim from file
        self.odim = self.x.shape[1] # None # self.data.shape[1]

    @decStep()
    def step(self, x = None):
        self.debug_print("%s.step: x = %s, bus = %s", (self.__class__.__name__, x, self.bus))
        # self.x = np.atleast_2d(self.data[[self.cnt]]).T #?
        self.debug_print("self.x = %s", (self.x))
        for k, v in self.outputs.items():
            buskey = "%s/%s" % (self.id, k)
            # self.bus[buskey] = getattr(self, k)
            self.bus[buskey] = self.data[k][[self.cnt]].T
        # self.bus[buskey] = self.x
        return self.x
