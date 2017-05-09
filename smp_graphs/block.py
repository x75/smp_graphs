"""smp_graphs - smp sensorimotor experiments as computation graphs

block: basic block of computation

2017 Oswald Berthold
"""

import uuid, sys
from collections import OrderedDict
import pickle

import numpy as np

from hyperopt import STATUS_OK, STATUS_FAIL

import pandas as pd

import smp_graphs.logging as log
from smp_graphs.utils import print_dict
from smp_graphs.common import conf_header, conf_footer
from smp_graphs.common import get_config_raw, get_config_raw_from_string
from smp_graphs.common import set_attr_from_dict

BLOCKSIZE_MAX = 10000

################################################################################
# utils, TODO move to utils.py
def ordereddict_insert(ordereddict = None, insertionpoint = None, itemstoadd = []):
    """self rolled ordered dict insertion
    from http://stackoverflow.com/questions/29250479/insert-into-ordereddict-behind-key-foo-inplace
    """
    assert ordereddict is not None
    assert insertionpoint in ordereddict.keys()
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
# Block decorator init
class decInit():
    """!@brief Block.init wrapper"""
    def __call__(self, f):
        def wrap(exec_self, *args, **kwargs):
            f(exec_self, *args, **kwargs)
        return wrap

################################################################################
# Block decorator step
class decStep():
    """!@brief Block.step wrapper"""
    def __call__(self, f):
        def wrap(exec_self, *args, **kwargs):
            if True:
                sname  = self.__class__.__name__
                esname = exec_self.cname
                esid   = exec_self.id
                escnt  = exec_self.cnt
                # loop over block's inputs
                for k, v in exec_self.inputs.items():
                    # copy bus inputs to input buffer
                    if v[2] is not None: # input item has a bus associated in v[2]
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
                                # # debugging in to out copy
                                # print "%s-%s.%s[%d] bus[%s] = %s" % (esname, esid,
                                #                                          sname,
                                #                                          escnt,
                                #                                          v[2],
                                #                                          exec_self.bus[v[2]].shape)
                                
                                sl = slice(-blocksize_input, None)
                                exec_self.inputs[k][0][:,sl] = exec_self.bus[v[2]] # np.fliplr(exec_self.bus[v[2]])
                                # exec_self.inputs[k][0][:,-1,np.newaxis] = exec_self.bus[v[2]]                                
                        else: # ibuf = 1
                            exec_self.inputs[k][0][:,[0]] = exec_self.bus[v[2]]
                            
                    # copy input to output if inkey k is in outkeys
                    if k in exec_self.outputs.keys():
                        # outvar = v[2].split("/")[-1]
                        # print "%s.stepwrap split %s from %s" % (exec_self.cname, outvar, v[2])
                        setattr(exec_self, k, v[0])
                        esk = getattr(exec_self, k)
                        
                        # # debug in to out copy
                        # print "%s.%s[%d]  self.%s = %s" % (esname, sname, escnt, k, esk)
                        # print "%s.%s[%d] outkeys = %s" % (esname, sname, escnt, exec_self.outputs.keys())
                        # if k in exec_self.outputs.keys():
                        #     print "%s.%s[%d]:   outk = %s" % (esname, sname, escnt, k)
                        #     print "%s.%s[%d]:    ink = %s" % (esname, sname, escnt, k)
                        #     # exec_self.outputs[k] = exec_self.inputs[k][0]

            # call the function on blocksize boundaries
            # FIXME: might not be the best idea to control that on the wrapper level as some
            #        blocks might need to be called every step nonetheless?
            if (exec_self.cnt % exec_self.blocksize) == 0:
                f_out = f(exec_self, None)

                # copy output to bus
                for k, v in exec_self.outputs.items():
                    buskey = "%s/%s" % (exec_self.id, k)
                    # print "copy %s.outputs[%s] = %s to bus[%s], bs = %d" % (exec_self.id, k, np.mean(getattr(exec_self, k)), buskey, exec_self.blocksize)
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
# Base block class
class Block2(object):
    """!@brief Basic block class
    """
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

    def load_defaults(self):
        for k,v in self.defaults.items():
            self.__dict__[k] = v
                
    def __init__(self, conf = {}, paren = None, top = None):
        self.conf = conf
        self.paren = paren
        self.top = top
        self.cname = self.__class__.__name__
        
        # load defaults
        # self.load_defaults()
        set_attr_from_dict(self, self.defaults)
                    
        # fetch existing configuration arguments
        if type(self.conf) == dict:
            set_attr_from_dict(self, self.conf['params'])
        else:
            print "What could it be? Look at %s" % (self.conf)

        # check
        assert hasattr(self, 'id')

        # input buffer vs. blocksize: input buffer is sliding, blocksize is jumping
        if self.blocksize > self.ibuf:
            self.ibuf = self.blocksize

        """pass 1: complete config with runtime info"""
        if hasattr(self, 'graph') or hasattr(self, 'subgraph'):
            # the topblock, there can only be one
            if self.topblock:
                # initialize the global messaging bus
                self.bus = {}

                # set the top reference
                self.top = self

                # initialize pandas based hdf5 logging
                log.log_pd_init(self.conf)

                # write initial configuration to dummy table attribute in hdf5
                log.log_pd_store_config_initial(print_dict(self.conf))
            # non topblock configuration driven blocks
            else:
                self.bus = self.top.bus
                # hierarchical block with config from a file
                if hasattr(self, 'subgraph'):
                    # FIXME: call that graph
                    subconf = get_config_raw(self.subgraph, 'conf') # 'graph')
                    assert subconf is not None
                    # make sure subordinate number of steps is less than top level numsteps
                    assert subconf['params']['numsteps'] <= self.top.numsteps, "enclosed numsteps = %d greater than top level numsteps = %d" % (subconf['params']['numsteps'], self.top.numsteps)

                    self.conf['params']['graph'] = subconf['params']['graph']
                    # print "subgraph %s" % (self.id), self.conf['params']['graph']

                    # insert subgraph into top graph
                    ordereddict_insert(ordereddict = self.top.graph, insertionpoint = '%s' % self.id, itemstoadd = subconf['params']['graph'])
                    # print "topgraph", print_dict(self.top.graph)
                                        
                    # # pass 1
                    # self.init_graph_pass_1()
                    # # pass 2
                    # self.init_graph_pass_2()

                # hierarchical block with config right inside block: what does it buy you if you ditch hierarchy during exe graph init?
                elif hasattr(self, 'graph'):
                    if not hasattr(self, 'numsteps'):
                        self.numsteps = self.top.numsteps
                    # subconf = get_config_raw_from_string(self.subgraph, 'conf')
                    # print "graph", self.graph, self.conf['params']['graph']
                    # print "graph same", self.graph is self.conf['params']['graph']
                    # self.init_graph_pass_1()
                    # self.init_graph_pass_2()
                    
            # common for all configuration driven blocks
                
            # init pass 1: construct the exec graph by expanding dynamic variables
            # and initializing the outputs to get the bus definitions
            self.init_graph_pass_1()

            # debug bus init
            # self.debug_print("init_1: buskeys = %s", (self.bus.keys(),)) # (print_dict(self.bus),))
            # for k,v in self.bus.items():
            #     self.debug_print("init_1: self.bus[%s].shape = %s", (k, v.shape)) # (print_dict(self.bus),))

            # init pass 2:
            self.init_graph_pass_2()

            # debug bus init
            # for k,v in self.bus.items():
            #     self.debug_print("init_2: self.bus[%s].shape = %s", (k, v.shape)) # (print_dict(self.bus),))

            # dump the exec graph configuration to a file
            if self.topblock:
                finalconf = self.dump_final_config()
                # this needs more work
                log.log_pd_store_config_final(finalconf)
                
            # print "init top graph", self.top.graph.keys(), self.top.graph['bhier']

        # primitive block
        else:
            """pass 1: complete config with runtime info"""
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
            self.graph[k]['block'] = self.graph[k]['block'](conf = v, paren = self, top = self.top)
            # print "%s self.graph[k]['block'] = %s" % (self.graph[k]['block'].__class__.__name__, self.graph[k]['block'].bus)
        # done pass 1 init

    def init_graph_pass_2(self):
        # pass 2 init
        for k, v in self.graph.items():
            # self.debug_print("__init__: pass 2\nk = %s,\nv = %s", (k, print_dict(v)))
            self.graph[k]['block'].init_pass_2()

    def init_outputs(self):
        # print "%s.init_outputs: inputs = %s" % (self.cname, self.inputs)
        # create outputs
        # format: variable: [shape]
        for k, v in self.outputs.items():
            # print "%s.init_outputs: outk = %s, outv = %s" % (self.cname, k, v)
            # if self.inputs.has_key(k):
            #     print "%s init_outputs ink = %s, inv = %s" % (self.cname, k, self.inputs[k])
            # alloc dim x blocksize buf
            self.outputs[k][0] = (v[0][0], self.blocksize)
            # create self attribute for output item, FIXME: blocksize?
            setattr(self, k, np.zeros(v[0])) # self.outputs[k][0]
            buskey = "%s/%s" % (self.id, k)
            # print "%s.init_outputs: bus[%s] = %s" % (self.cname, buskey, getattr(self, k).shape)
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
                self.debug_print("__init__: pass 2\n    in_k = %s,\n    in_v = %s", (k, v))
                assert len(v) > 0
                # set input from bus
                if type(v[0]) is str:
                    assert self.bus.has_key(v[0]), "Bus item %s doesn't not in %s" % (v[0], self.bus.keys())
                    # enforce bus blocksize smaller than local blocksize, tackle later
                    # print "%s" % self.cname, self.bus.keys()
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
        """Base block step function

        if topblock iterate graph and step each node block, reiterate graph and do the logging for each node, store the log every n steps
        """
        # if self.topblock or hasattr(self, 'subgraph'):
        # if hasattr(self, 'graph') or hasattr(self, 'subgraph'):
        # mode 1 for handling hierarchical blocks: graph is flattened during init, only topblock iterates nodes
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
                self.debug_print("step: node k = %s, v = %s", (k, v))
                # do logging for all of the node's output variables
                for k_o, v_o in v['block'].outputs.items():
                    buskey = "%s/%s" % (v['block'].id, k_o)
                    log.log_pd(tbl_name = buskey, data = self.bus[buskey])

        if self.topblock:
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

    @decStep()
    def step(self, x = None):
        # print "%s.step inputs[%d] = %s" % (self.cname, self.cnt, self.inputs)
        # self.inputs is a dict with values [array]
        assert self.inputs.has_key('x'), "%s.inputs expected to have key 'x' with function input vector. Check config."

        self.debug_print("step[%d]: x = %s", (self.cnt, self.inputs,))
        self.y = self.func(self.inputs)
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

        loopblocks = []
        # loop the loop
        for i, lparams in enumerate(self.loop):
            # print "lparams", lparams, "self.loopblock['params']", self.loopblock['params']
            
            # copy params
            loopblock_params = {}
            for k, v in self.loopblock['params'].items():
                if k == 'id':
                    loopblock_params[k] = "%s_%d" % (self.id, i+1)
                elif k == lparams[0]:
                    loopblock_params[k] = lparams[1]
                else:
                    loopblock_params[k] = v

            # create dynamic conf
            loopblock_conf = {'block': self.loopblock['block'], 'params': loopblock_params}
            # instantiate block
            dynblock = self.loopblock['block'](conf = loopblock_conf,
                                               paren = self.paren, top = self.top)

            # get config and store it
            dynblockconf = dynblock.get_config()
            dynblockconf[1]['block'] = dynblock

            # append to list of dynamic blocks
            loopblocks.append(dynblockconf)
            # print print_dict(self.top.graph)

        # # debug loopblocks in LoopBlock2 init
        # for item in loopblocks:
        #     print "%s.__init__ loopblocks = %s: %s" % (self.__class__.__name__, item[0], print_dict(item[1]))

        # print "loopblocks", self.id, loopblocks
                
        # FIXME: this good?
        # insert dynamic blocks into existing ordered dict
        # print "topgraph", print_dict(self.top.graph)
        ordereddict_insert(ordereddict = self.top.graph, insertionpoint = '%s' % self.id, itemstoadd = loopblocks)

        # print "top graph", print_dict(self.top.graph)
        # print "top graph", self.top.graph.keys()
        # print "top graph", print_dict(self.top.graph[self.top.graph.keys()[0]])

        # replace loopblock block entry in original config, propagated back to the top / global namespace
        self.loopblock['block'] = Block2.__class__.__name__
                   
    def step(self, x = None):
        """loop block does nothing for now"""
        pass

class SeqLoopBlock2(Block2):
    """!@brief Sequential loop block"""
    def __init__(self, conf = {}, paren = None, top = None):
        self.defaults['loop'] = [1]
        # self.defaults['loopmode'] = 'sequential'
        self.defaults['loopblock'] = {}
        Block2.__init__(self, conf = conf, paren = paren, top = top)

        if type(self.loop) is list: # it's a list
            self.f_loop = self.f_loop_list
            assert len(self.loop) == self.blocksize, "%s step blocksize (%d) needs to be equal the loop length (%d)" % (self.cname, self.blocksize, len(self.loop))
        else: # it's a func
            self.f_loop = self.f_loop_func

    def f_loop_list(self, i, f_obj):
        results = f_obj(self.loop[i])
        return results

    def f_loop_func(self, i, f_obj):
        results = self.loop(self, i, f_obj)
        return results
            
    @decStep()
    def step(self, x = None):

        def f_obj(lparams):
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
            dynblock = self.loopblock['block'](conf = loopblock_conf,
                                               paren = self.paren, top = self.top)
            # second pass
            dynblock.init_pass_2()

            for j in range(dynblock.blocksize):
                # print "%s trying %s.step[%d]" % (self.cname, dynblock.cname, j)
                dynblock.step()

            loss = 0
            for outk in self.outputs.keys():
                if outk in dynblock.inputs.keys(): continue
                # print "outk", outk, getattr(dynblock, outk)
                loss += np.mean(getattr(dynblock, outk), axis = 1, keepdims = True)

            # print "loss", loss
                
            rundata = {
                'loss': loss[0,0], # compute_complexity(Xs)
                'status': STATUS_OK, # compute_complexity(Xs)
                'dynblock': dynblock,
                'lparams': lparams,
                # "M": M, # n.networks["slow"]["M"], # n.M
                # "timeseries": Xs.copy(),
                # "loss": np.var(Xs),
            }
                
            return rundata
        
        def f_obj_hpo(params):
            # print "f_obj_hpo: params", params
            x = np.array([params]).T
            # XXX
            lparams = ('inputs', {'x': [x]}) # , x.shape, self.outputs['x']
            # print "%s.step.f_obj_hpo lparams = {%s}" % (self.cname, lparams)
            rundata = f_obj(lparams)
            return rundata
        # {'loss': , 'status': STATUS_OK, 'dynblock': None, 'lparams': lparams}

        if type(self.loop) is list:
            f_obj_ = f_obj
        else:
            f_obj_ = f_obj_hpo
            
        # loop the loop
        for i in range(self.blocksize):
            # print "%s iter# %d" % (self.cname, i)
            # dynblock = obj()
            # func: need input function from config
            results = self.f_loop(i, f_obj_)#_hpo)
            dynblock = results['dynblock']
            lparams = results['lparams']
            
            # print "%s.steploop[%d], %s, %s" % (self.cname, i, lparams, self.loopblock['params'])
            # for k,v in dynblock.outputs.items():
            #     print "dynout", getattr(dynblock, k)

            for outk in self.outputs.keys():
                # outvar = getattr(self, outk)
                # outvar[:,[i] = bla
                
                # func: need output function from config
                # self.__dict__[outk][:,[i]] = np.mean(getattr(dynblock, outk), axis = 1, keepdims = True)
                self.__dict__[outk][:,[i]] = np.mean(getattr(dynblock, outk), axis = 1, keepdims = True)
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

class ConstBlock2(PrimBlock2):
    """!@brief Constant block: output is a constant vector
    """
    def __init__(self, conf = {}, paren = None, top = None):
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)
        
        # either column vector to be replicated or blocksize already
        assert self.x.shape[1] in [1, self.blocksize]

        # replicate column vector
        # if self.x.shape[1] == 1: # this was wrong
        if self.inputs['c'][0].shape[1] == 1:
            self.x = np.tile(self.inputs['c'][0], self.blocksize) # FIXME as that good? only works for single column vector
        else:
            self.x = self.inputs['c'][0].copy() # FIXME as that good? only works for single column vector

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
        self.cnt_ = np.zeros((self.outputs[self.outk][0][0], self.blocksize))
        # print self.inputs
        
    @decStep()
    def step(self, x = None):
        """CountBlock step: if blocksize is 1 just copy the counter, if bs > 1 set cnt_ to range"""
        if self.blocksize > 1:
            self.cnt_[:,-self.blocksize:] = np.arange(self.cnt - self.blocksize, self.cnt).reshape(self.outputs[self.outk][0])
        else:
            self.cnt_[:,0] = self.cnt
        # FIXME: make that a for output items loop
        setattr(self, self.outputs.keys()[0], (self.cnt_ * self.scale) + self.offset)

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

# FIXME: this should go into systems    
class FileBlock2(Block2):
    """!@brief File block: read some log or data file and output blocksize lines each step"""
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
                    print "%s = %s\n" % (k, self.store.get_storer(k).attrs.conf,)
