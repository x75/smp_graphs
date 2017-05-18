
import os

import numpy as np
import pandas as pd

from smp_graphs.common import read_puppy_hk_pickles
from smp_graphs.block  import Block2, decInit, decStep

# FIXME: this should go into systems    
class FileBlock2(Block2):
    """!@brief File block: read some log or data file and output blocksize lines each step"""
    @decInit()
    def __init__(self, conf = {}, paren = None, top = None):
        # ad hoc default
        # if not conf['params'].has_key('type'): conf['params']['type'] = 'puppy'
        assert conf['params'].has_key('type'), "Type required for FileBlock2"
        # multiple files: concat? block manipulation blocks?
        self.file = []
        # auto odim
        # if self.odim == 'auto':
        # print conf
        lfile = conf['params']['file'][0]
        ############################################################
        # puppy homeokinesis (andi)
        if conf['params']['type'] == 'puppy':
            del conf['params']['outputs']['log']
            (self.data, self.rate, self.offset) = read_puppy_hk_pickles(lfile)
            # print "self.data", self.data.keys()
            # for k,v in self.data.items():
            #     if type(v) is np.ndarray:
            #         print "puppylog", k, lfile, v.shape, np.mean(v), np.var(v), v
            # setattr(self, 'x', self.data['x'])
            # setattr(self, 'x', self.data['x'])
            # conf['params']['outputs'][k_] = [v.shape]
            # conf['params']['blocksize'] = v.shape[0]
            # assert filedata eq blocksize
            self.step = self.step_puppy
        ############################################################
        # sphero res learner
        elif conf['params']['type'] == 'sphero_res_learner':
            self.data = np.load(lfile)
            print "fileblock self.data sphero", self.data.keys()
            del conf['params']['outputs']['log']
            # for k in self.data.keys():
            for k in ['x', 'zn', 't']:
                sl = slice(None)
                k_ = k
                if k is 'zn':
                    k_ = 'y'
                elif k == 't': # target
                    sl = slice(0, 1)
                else:
                    pass

                conf['params']['outputs'][k_] = [self.data[k][:,sl].T.shape]
            self.step = self.step_sphero_res_learner
            print "fileblock sphero_res_learner conf", conf['params']['outputs']
        ############################################################
        # test data format 1
        elif conf['params']['type'] == 'testdata1':
            self.data = np.load(lfile)
            print "fileblock self.data testdata1", self.data.keys()
            del conf['params']['outputs']['log']
            for k in self.data.keys():
            # for k in ['x', 'zn', 't']:
                sl = slice(None)
                k_ = k
                # if k is 'zn':
                #     k_ = 'y'
                # elif k == 't': # target
                #     sl = slice(0, 1)
                # else:
                #     pass

                conf['params']['outputs'][k_] = [self.data[k][:,sl].shape]
            self.step = self.step_testdata1
            print "fileblock testdata1 conf", conf['params']['outputs']
            
        ############################################################
        # selflogconfs
        elif conf['params']['type'] == 'selflogconf':
            self.store = pd.HDFStore(lfile)
            self.step = self.step_selflogconf
        ############################################################
        # selflog
        elif conf['params']['type'] == 'selflog':
            assert os.path.exists(lfile), "logfile %s not found" % (lfile, )
            self.store = pd.HDFStore(lfile)
            # clean up dummy entry
            del conf['params']['outputs']['log']
            # loop over log tables and create output for each table
            conf['params']['storekeys'] = {}
            for k in self.store.keys():
                # process key from selflog format: remove the block id in the beginning of the key
                k_ = k.lstrip("/")
                if not k_.startswith('conf'):
                    k_ = "/".join(k_.split("/")[1:])
                print "%s" % self.__class__.__name__, k, self.store[k].shape
                conf['params']['outputs'][k_] = [self.store[k].T.shape]
                conf['params']['blocksize'] = self.store[k].shape[0]
                # map output key to log table key
                conf['params']['storekeys'][k_] = k
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
        self.debug_print("%s.step: x = %s, bus = %s", (self.__class__.__name__, self.x, self.bus))
        # self.x = np.atleast_2d(self.data[[self.cnt]]).T #?
        # self.debug_print("self.x = %s", (self.x,))
        if (self.cnt % self.blocksize) == 0: # (self.blocksize - 1):
            for k, v in self.outputs.items():                
                sl = slice(self.cnt-self.blocksize, self.cnt)
                setattr(self, k, self.data[k][sl].T)
                # setattr(self, k, self.data[k][[self.cnt]].T)
            
    @decStep()
    def step_sphero_res_learner(self, x = None):
        self.debug_print("%s.step: x = %s, bus = %s", (self.__class__.__name__, x, self.bus))
        # self.x = np.atleast_2d(self.data[[self.cnt]]).T #?
        self.debug_print("self.x = %s", (self.x,))
        if (self.cnt % self.blocksize) == 0: # (self.blocksize - 1):
            for k, v in self.outputs.items():
                sl = slice(self.cnt-self.blocksize, self.cnt)
                sl2 = slice(None)
                k_ = k
                if k is 'y':
                    k_ = 'zn'
                elif k == 't':
                    sl2 = slice(0, 1)
                else:
                    pass
                setattr(self, k, self.data[k_][sl,sl2].T)
                # setattr(self, k, self.data[k][[self.cnt]].T)
                
    @decStep()
    def step_testdata1(self, x = None):
        self.debug_print("%s.step: x = %s, bus = %s", (self.__class__.__name__, x, self.bus))
        # self.x = np.atleast_2d(self.data[[self.cnt]]).T #?
        self.debug_print("self.x = %s", (self.x,))
        if (self.cnt % self.blocksize) == 0: # (self.blocksize - 1):
            for k, v in self.outputs.items():
                k_ = k
                sl = slice(self.cnt-self.blocksize, self.cnt)
                sl2 = slice(None)
                setattr(self, k, self.data[k_][sl,sl2])
                
    @decStep()
    def step_selflog(self, x = None):
        if (self.cnt % self.blocksize) == 0:
            for k, v in self.outputs.items():
                # if k.startswith('conf'):
                storek = self.storekeys[k]
                print "step: cnt = %d key = %s, log.sh = %s" % (self.cnt, k, self.store[storek].shape)
                # print self.store[k].values
                setattr(self, k, self.store[storek][self.cnt-self.blocksize:self.cnt].values.T)
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
