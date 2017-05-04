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
            log.log_pd_store_config_initial(print_dict(self.conf))

            # init pass 1: complete the graph by expanding dynamic variables and initializing the outputs to get the bus def
            self.init_graph_pass_1()

            # init pass 2:
            self.init_graph_pass_2()
            
            self.debug_print("self.bus = %s", (print_dict(self.bus),))

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
                    'block': self.cname,
                    'params': params
                })
        return conf

    def dump_final_config(self):
        finalconf = self.get_config()
        print type(finalconf), finalconf
        # pickle.dump(finalconf, open("data/%s_conf.pkl" % (self.id, ), "wb"))
        dump_final_config_file = "data/%s.conf" % (self.id)
        f = open(dump_final_config_file, "w")
        # confstr = repr(finalconf[1]['params'])
        confstr = print_dict(pdict = finalconf[1]['params'])
        confstr_ = "conf = {'block': 'Block2', 'params': %s}" % (confstr, )
        f.write(confstr_)
        f.flush()
        print "%s.dump_final_config wrote config, closing file %s" % (self.cname, dump_final_config_file,)
        f.close()

        log.log_pd_store_config_final(confstr_)
    
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
        
        # # loop over outputs dict and copy them to a slot in the bus
        # for k, v in self.outputs.items():
        #     buskey = "%s/%s" % (self.id, k)
        #     self.bus[buskey] = getattr(self, k)
        # self.bus[self.id] = self.x
        return self.x

class FileBlock2(Block2):
    @decInit()
    def __init__(self, conf = {}, paren = None, top = None):
        # multiple files: concat? block manipulation blocks?
        self.file = []
        # auto odim
        # if self.odim == 'auto':
        # print conf
        lfile = conf['params']['file'][0]
        # puppy homeokinesis (andi)
        if lfile.startswith('data/pickles_puppy') and lfile.endswith('.pickle'):
            (self.data, self.rate, self.offset) = read_puppy_hk_pickles(lfile)
            # setattr(self, 'x', self.data['x'])
            # setattr(self, 'x', self.data['x'])

        # init states
        # for k, v in self.outputs.items(): # ['x', 'y']:
        for k, v in conf['params']['outputs'].items(): # ['x', 'y']:
            # print "key", self.data[key]
            # setattr(self, k, np.zeros((self.data[k].shape[1], 1)))
            # print "v[0]", v[0]
            if v[0] is None:
                # self.outputs[k][0] = (self.data[k].shape[1], 1)
                print "data", self.data[k].shape[1]
                print "out dim", conf['params']['outputs'][k]
                conf['params']['outputs'][k][0] = (self.data[k].shape[1], 1)
                print "out dim", conf['params']['outputs'][k]
            # self.x = np.zeros((self.odim, 1))
        
        Block2.__init__(self, conf = conf, paren = paren, top = top)
        
        # set odim from file
        # self.odim = self.x.shape[1] # None # self.data.shape[1]

    @decStep()
    def step(self, x = None):
        self.debug_print("%s.step: x = %s, bus = %s", (self.__class__.__name__, x, self.bus))
        # self.x = np.atleast_2d(self.data[[self.cnt]]).T #?
        self.debug_print("self.x = %s", (self.x,))
        for k, v in self.outputs.items():
            setattr(self, k, self.data[k][[self.cnt]].T)
            
        return self.x
