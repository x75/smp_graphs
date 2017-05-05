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

################################################################################
# utils, TODO move to utils.py
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
    
def read_puppy_hk_pickles(lfile, key = None):
    """read pickled log dicts from andi's puppy experiments"""
    d = pickle.load(open(lfile, 'rb'))
    # print "d.keys", d.keys()
    # data = d["y"][:,0,0] # , d["y"]
    rate = 20
    offset = 0
    # data = np.atleast_2d(data).T
    # print "wavdata", data.shape
    # special treatment for x,y with addtional dimension
    data = d
    x = d['x']
    data['x'] = x[:,:,0]
    y = d['y']
    data['y'] = y[:,:,0]
    # print "x.shape", data['x'].shape
    return (data, rate, offset)

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
            if True:
                # dummy input
                x = {}
                # loop over block's inputs
                for k, v in exec_self.inputs.items():
                    # copy bus inputs to input buffer
                    if v[2] is not None:
                        # if extended input buffer, rotate the data with each step
                        if exec_self.ibuf > 1:
                            # input blocksize
                            blocksize_input = exec_self.bus[v[2]].shape[1]
                            # input block border 
                            if (exec_self.cnt % blocksize_input) == 0: # (blocksize_input - 1):
                                # shift by input blocksize
                                exec_self.inputs[k][0] = np.roll(exec_self.inputs[k][0], shift = -blocksize_input, axis = 1)
                                # print "decinit", exec_self.inputs[k][0].shape
                                
                                # set inputs [last-inputbs:last] if input blocksize reached
                                sl = slice(-blocksize_input, None)
                                exec_self.inputs[k][0][:,sl] = exec_self.bus[v[2]] # np.fliplr(exec_self.bus[v[2]])
                                # exec_self.inputs[k][0][:,-1,np.newaxis] = exec_self.bus[v[2]]
                        else: # ibuf = 1
                            exec_self.inputs[k][0][:,[0]] = exec_self.bus[v[2]]
                            

            # call the function depending on the blocksize
            if (exec_self.cnt % exec_self.blocksize) == 0: # (exec_self.blocksize - 1):
                f_out = f(exec_self, x)

                # copy output to bus
                for k, v in exec_self.outputs.items():
                    buskey = "%s/%s" % (exec_self.id, k)
                    # print "getattr", getattr(exec_self, k)
                    exec_self.bus[buskey] = getattr(exec_self, k)
            else:
                f_out = None
            
            # count calls
            exec_self.cnt += 1 # should be min_blocksize
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
        'cnt': 1,
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
            log.log_pd_store_config_initial(print_dict(self.conf))

            # init pass 1: complete the graph by expanding dynamic variables and initializing the outputs to get the bus def
            self.init_graph_pass_1()

            # self.debug_print("init_1: buskeys = %s", (self.bus.keys(),)) # (print_dict(self.bus),))
            # for k,v in self.bus.items():
            #     self.debug_print("init_1: self.bus[%s].shape = %s", (k, v.shape)) # (print_dict(self.bus),))

            # init pass 2:
            self.init_graph_pass_2()

            # for k,v in self.bus.items():
            #     self.debug_print("init_2: self.bus[%s].shape = %s", (k, v.shape)) # (print_dict(self.bus),))

            self.dump_final_config()
            
            # log.log_pd_dump_config()
                                                    
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
            # self.debug_print("__init__: pass 1\nk = %s,\nv = %s", (k, print_dict(v)))
            self.graph[k]['block'] = self.graph[k]['block'](conf = v, paren = self, top = self)
            # print "%s self.graph[k]['block'] = %s" % (self.graph[k]['block'].__class__.__name__, self.graph[k]['block'].bus)
        # done pass 1 init

    def init_graph_pass_2(self):
        # pass 2 init
        for k, v in self.graph.items():
            # self.debug_print("__init__: pass 2\nk = %s,\nv = %s", (k, print_dict(v)))
            self.graph[k]['block'].init_pass_2()

    def init_outputs(self):
        # create outputs
        # format: variable: [shape]
        for k, v in self.outputs.items():
            # alloc dim x blocksize buf
            self.outputs[k][0] = (v[0][0], self.blocksize)
            # create self attribute for output item
            setattr(self, k, np.zeros(self.outputs[k][0]))
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
            # format: variable: [buffered const/array, shape, bus]
            for k, v in self.inputs.items():
                self.debug_print("__init__: pass 2\n    k = %s,\n    v = %s", (k, v))
                assert len(v) > 0
                # set input from bus
                if type(v[0]) is str:
                    # enforce bus blocksize smaller than local blocksize, tackle later
                    print "%s" % self.cname, self.bus.keys()
                    assert self.bus[v[0]].shape[1] <= self.blocksize
                    # create input tuple
                    tmp = [None for i in range(3)]
                    # get the bus
                    inbus = self.bus[v[0]]
                    # alloc the inbuf
                    tmp[0] = np.zeros((inbus.shape[0], self.ibuf)) # ibuf >= blocksize
                    # splice buf into inbuf
                    tmp[0][:,0:inbus.shape[1]] = inbus
                    # store shape fwiw
                    tmp[1] = tmp[0].shape
                    # store buskey
                    tmp[2] = v[0]
                    # assign tuple
                    self.inputs[k] = tmp # 
                    print "%s.init_pass_2: %s, bus[%s] = %s, input = %s" % (self.cname, self.id, v[0], self.bus[v[0]].shape, self.inputs[k][0].shape)
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
                # stack??
                # self.inputs[k][0] = np.hstack((np.zeros((self.inputs[k][1][0], self.ibuf-1)), self.inputs[k][0]))
                self.debug_print("init_pass_2 %s in_k.shape = %s", (self.id, self.inputs[k][0].shape))
            
    def debug_print(self, fmtstring, data):
        """only print if debug is enabled for this block"""
        fmtstring = "\n%s[%d]." + fmtstring
        data = (self.cname,self.cnt) + data
        if self.debug:
            print fmtstring % data

    # undecorated step, need to count ourselves
    def step(self, x = None):
        if self.topblock:
            for k, v in self.graph.items():
                v['block'].step()

            # do logging for all nodes, was bus items
            for k, v in self.graph.items(): # self.bus.items():
                # return if block doesn't want logging
                if not v['block'].logging: continue
                # if (self.cnt % v['block'].blocksize) != (v['block'].blocksize - 1): continue
                if (self.cnt % v['block'].blocksize) > 0: continue
                # print debug foo
                self.debug_print("step: node k = %s, v = %s", (self.__class__.__name__, k, v))
                # do logging for all of the node's output variables
                for k_o, v_o in v['block'].outputs.items():
                    buskey = "%s/%s" % (v['block'].id, k_o)
                    log.log_pd(tbl_name = buskey, data = self.bus[buskey])

            # store log
            if (self.cnt) % 100 == 0:
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
        confstr = print_dict(pdict = finalconf[1]['params'])
        confstr_ = "conf = {'block': 'Block2', 'params': %s}" % (confstr, )
        f.write(confstr_)
        f.flush()
        # print "%s.dump_final_config wrote config, closing file %s" % (self.cname, dump_final_config_file,)
        f.close()

        # log.log_pd_store_config_final(confstr_)
    
class LoopBlock2(Block2):
    """Loop block: dynamically create block variations according to some specificiations of variation

    two modes:
     - parallel mode: modify the graph structure and all block variations at the same time
     - sequential mode: modify execution to run each variation one after the other
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
                    loopblock_params[k] = "%s_%d" % (self.id, i+1)
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

        # replace loopblock block entry
        self.loopblock['block'] = Block2.__class__.__name__
                   
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
        # self.debug_print("%s.step: x = %s, bus = %s", (self.__class__.__name__, self.x, self.bus))

        # self.x = self.in        
        # # loop over outputs dict and copy them to a slot in the bus
        # for k, v in self.outputs.items():
        #     buskey = "%s/%s" % (self.id, k)
        #     self.bus[buskey] = getattr(self, k)
            
        return self.x

class CountBlock2(PrimBlock2):
    @decInit()
    def __init__(self, conf = {}, paren = None, top = None):
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)

    @decStep()
    def step(self, x = None):
        pass
        
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
        
        # # loop over outputs dict and copy them to a slot in the bus
        # for k, v in self.outputs.items():
        #     buskey = "%s/%s" % (self.id, k)
        #     self.bus[buskey] = getattr(self, k)
        # self.bus[self.id] = self.x
        return self.x

import pandas as pd
class FileBlock2(Block2):
    @decInit()
    def __init__(self, conf = {}, paren = None, top = None):
        # ad hoc default
        if not conf['params'].has_key('type'): conf['params']['type'] = 'puppy'
        # multiple files: concat? block manipulation blocks?
        self.file = []
        # auto odim
        # if self.odim == 'auto':
        # print conf
        lfile = conf['params']['file'][0]
        # puppy homeokinesis (andi)
        if conf['params']['type'] == 'puppy':
            (self.data, self.rate, self.offset) = read_puppy_hk_pickles(lfile)
            # setattr(self, 'x', self.data['x'])
            # setattr(self, 'x', self.data['x'])
            self.step = self.step_puppy
        # selflogconfs
        elif conf['params']['type'] == 'selflogconf':
            self.store = pd.HDFStore(lfile)
            self.step = self.step_selflogconf
        elif conf['params']['type'] == 'selflog':
            self.store = pd.HDFStore(lfile)
            # clean up dummy entry
            del conf['params']['outputs']['log']
            # loop log tables
            for k in self.store.keys():
                print "%s" % self.__class__.__name__, k, self.store[k].shape
                conf['params']['outputs'][k] = [self.store[k].T.shape]
                conf['params']['blocksize'] = self.store[k].shape[0]
            self.step = self.step_selflog

        # init states
        for k, v in conf['params']['outputs'].items(): # ['x', 'y']:
            # print "key", self.data[key]
            # setattr(self, k, np.zeros((self.data[k].shape[1], 1)))
            # print "v[0]", v[0]
            if v[0] is None:
                # self.outputs[k][0] = (self.data[k].shape[1], 1)
                # print "data", self.data[k].shape[1]
                # print "out dim", conf['params']['outputs'][k]
                conf['params']['outputs'][k][0] = (self.data[k].shape[1], conf['params']['blocksize'])
                # print "out dim", conf['params']['outputs'][k]
            # self.x = np.zeros((self.odim, 1))
        
        Block2.__init__(self, conf = conf, paren = paren, top = top)
        
        # set odim from file
        # self.odim = self.x.shape[1] # None # self.data.shape[1]

    @decStep()
    def step(self, x = None):
        pass
    
    @decStep()
    def step_puppy(self, x = None):
        self.debug_print("%s.step: x = %s, bus = %s", (self.__class__.__name__, x, self.bus))
        # self.x = np.atleast_2d(self.data[[self.cnt]]).T #?
        self.debug_print("self.x = %s", (self.x,))
        if (self.cnt % self.blocksize) == 0: # (self.blocksize - 1):
            for k, v in self.outputs.items():                
                sl = slice(self.cnt-self.blocksize, self.cnt)
                setattr(self, k, self.data[k][sl].T)
                # setattr(self, k, self.data[k][[self.cnt]].T)
            
    @decStep()
    def step_selflog(self, x = None):
        if (self.cnt % self.blocksize) == 0:
            for k, v in self.outputs.items():
                # if k.startswith('conf'):
                print "step: cnt = %d key = %s, log.sh = %s" % (self.cnt, k, self.store[k].shape)
                # print self.store[k].values
                setattr(self, k, self.store[k][self.cnt-self.blocksize:self.cnt].values.T)
                # print self.store[k][self.cnt-self.blocksize:self.cnt].values.T
                # print k
        # for k in self.__dict__.keys(): #self.bus.keys():
            # k = k.replace("/", "_")
            # print "k", k
                        
    @decStep()
    def step_selflogconf(self, x = None):
        self.debug_print("%s.step: x = %s, bus = %s", (self.__class__.__name__, x, self.bus))
        if (self.cnt % self.blocksize) == 0:
            for k, v in self.outputs.items():
                if k.startswith('conf'):
                    print "%s = %s" % (k, self.store.get_storer(k).attrs.conf,)
