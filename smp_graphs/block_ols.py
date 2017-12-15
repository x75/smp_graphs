
import os

import numpy as np
import pandas as pd

from smp_base.common import get_module_logger

from smp_graphs.common import read_puppy_hk_pickles
from smp_graphs.block  import Block2, decInit, decStep
from smp_graphs.block  import PrimBlock2

import logging
logger = get_module_logger(modulename = 'block_ols', loglevel = logging.DEBUG)

class FileBlock2(Block2):
    """!@brief File block: read some log or data file and output blocksize lines each step"""
    defaults = {
        'block_group': 'data',
    }
    @decInit()
    def __init__(self, conf = {}, paren = None, top = None):
        # ad hoc default
        # if not conf['params'].has_key('type'): conf['params']['type'] = 'puppy'
        assert 'file' in conf['params'] and len(conf['params']['file']) > 0, "FileBlock2 requires a 'file' parameter"
        assert 'type' in conf['params'], "FileBlock2 requires a file 'type' parameter"
        # multiple files: concat? block manipulation blocks?
        self.file = []
        self.nesting_indent = "        "
        self.id = "bla"
        # auto odim
        # if self.odim == 'auto':
        # print conf
        if type(conf['params']['file']) is dict:
            lfile = conf['params']['file']['filename']
        else:
            lfile = conf['params']['file'][0]
            # make it a dict
            conf['params']['file'] = {'filename': lfile}
            
        
        ############################################################
        # puppy homeokinesis (andi)
        if 'filetype' in conf['params']['file']:
            filetype = conf['params']['file']['filetype']
        else:
            filetype = conf['params']['type']
        logger.info("loading %s-type data from %s" % (filetype, lfile))

        self.init_load_file(filetype, lfile, conf)
        # FIXME: perform quick check of data
        
            
        # init states
        for k, v in list(conf['params']['outputs'].items()): # ['x', 'y']:
            # print "key", self.data[k]
            # setattr(self, k, np.zeros((self.data[k].shape[1], 1)))
            # print "v", v
            assert type(v) is dict, "FileBlock2 outputs need a configuration dictionary, not %s" % (type(v),)
            assert 'shape' in v, "FileBlock2 outputs need a 'shape' key"
            if v['shape'] is None:
                # self.outputs[k][0] = (self.data[k].shape[1], 1)
                # print "data", self.data[k].shape[1]
                # print "out dim", conf['params']['outputs'][k]
                conf['params']['outputs'][k]['shape'] = (self.data[k].shape[1], conf['params']['blocksize'])
                print("out dim", conf['params']['outputs'][k])
            # self.x = np.zeros((self.odim, 1))
        
        Block2.__init__(self, conf = conf, paren = paren, top = top)
        
        # set odim from file
        # self.odim = self.x.shape[1] # None # self.data.shape[1]

    def init_load_file(self, filetype, lfile, conf):
        if filetype == 'puppy':
            offset = 0
            if 'log' in conf['params']['outputs']:
                del conf['params']['outputs']['log']
            (self.data, self.rate, self.offset, length) = read_puppy_hk_pickles(lfile)

            logger.info("init_load_file loaded puppy data, outputs = %s" % (conf['params']['outputs']))
            logger.debug("init_load_file loaded puppy data = %s" % (self.data['x'], ))
            
            logger.debug("init_load_file loaded puppy data, data['x'] = %s" % (self.data['x'].shape, ))
            logger.debug("init_load_file loaded puppy data, data['y'] = %s" % (self.data['y'].shape, ))
             
            # check offset
            if 'offset' in conf['params']['file']:
                offset = conf['params']['file']['offset']
                
            # check length
            length = None # 0
            if 'length' in conf['params']['file']:
                length = conf['params']['file']['length']

            # reslice
            if length is not None:
                sl = slice(offset, offset + length)
            else:
                sl = slice(offset, length)
            # for k in ['x', 'y']:
            #     self.data[k] = self.data[k][sl]
                
            logger.debug("init_load_file slicing data with %s from offset = %s, length = %s" % (sl, offset, length))
                    
            for k in ['x', 'y']:
                self.data[k] = self.data[k][sl]
                # setattr(self, 'x', self.data['x'])
                logger.debug("    self.data = %s %s/%s" % (
                    self.data[k].shape,
                    np.mean(self.data[k], axis = 0),
                    np.var(self.data[k], axis = 0)))
                # for k,v in self.data.items():
                #     if type(v) is np.ndarray:
                #         print "puppylog", k, lfile, v.shape, np.mean(v), np.var(v), v
                # setattr(self, 'x', self.data['x'])
                # print "fileblock 'puppy' data.keys()", self.data.keys(), conf['params']['blocksize']

                if 'shape' not in conf['params']['outputs'][k]:
                    conf['params']['outputs'][k] = {'shape': self.data[k].T.shape} # [:-1]
                # conf['params']['outputs']['y'] = {'shape': self.data['y'].T.shape} # [:-1]
            
            # conf['params']['outputs'][k_] = [v.shape]
            # conf['params']['blocksize'] = v.shape[0]
            # assert filedata eq blocksize
            self.step = self.step_puppy
        ############################################################
        # sphero res learner
        elif filetype == 'sphero_res_learner':
            self.data = np.load(lfile)
            print("fileblock self.data sphero", list(self.data.keys()))
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

                conf['params']['outputs'][k_] = {'shape': self.data[k][:,sl].T.shape}
            self.step = self.step_sphero_res_learner
            print("fileblock sphero_res_learner conf", conf['params']['outputs'])
        ############################################################
        # test data format 1
        elif filetype == 'testdata1':
            self.data = np.load(lfile)
            print("fileblock self.data testdata1", list(self.data.keys()))
            del conf['params']['outputs']['log']
            for k in list(self.data.keys()):
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
            print("fileblock testdata1 conf", conf['params']['outputs'])
            
        ############################################################
        # selflogconfs
        elif filetype == 'selflogconf':
            assert os.path.exists(lfile), "logfile %s not found" % (lfile, )
            self.store = pd.HDFStore(lfile)
            self.step = self.step_selflogconf
        ############################################################
        # selflog
        elif filetype == 'selflog':
            # print "FileBlock2 selflog", conf['params']['blocksize']
            assert os.path.exists(lfile), "logfile %s not found" % (lfile, )
            self.store = pd.HDFStore(lfile)
            # clean up dummy entry
            del conf['params']['outputs']['log']
            # loop over log tables and create output for each table
            conf['params']['storekeys'] = {}
            for k in list(self.store.keys()):
                # process key from selflog format: remove the block id in the beginning of the key
                # print "FileBlock2 selflog", conf['params']['blocksize']
                k_ = k.lstrip("/")
                if not k_.startswith('conf'):
                    n_ = k_.split("/")[0]
                    # FIXME: hack
                    if n_ == 'pre_l1': continue
                    k_ = "/".join(k_.split("/")[1:])
                    # assert conf['params']['blocksize'] == self.store[k].shape[0], "numsteps (%d) needs to be set to numsteps (%s) in the file %s" % (conf['params']['blocksize'], self.store[k].shape, lfile)
                    
                    print("%s.init store_key = %s, raw key = %s, shape = %s" % (self.__class__.__name__, k, k_, self.store[k].shape))
                    # conf['params']['outputs'][k_] = {'shape': self.store[k].T.shape[:-1]}
                    if 'blocksize' in conf['params']:
                        conf['params']['outputs'][k_] = {'shape': self.store[k].T.shape[:-1] + (conf['params']['blocksize'],)}
                    else:
                        conf['params']['outputs'][k_] = {'shape': self.store[k].T.shape}
                # print "out shape", k_, conf['params']['outputs'][k_]
                # conf['params']['blocksize'] = self.store[k].shape[0]
                # map output key to log table key
                conf['params']['storekeys'][k_] = k
            self.step = self.step_selflog
        ############################################################
        # wav file
        elif filetype == 'wav':
            import scipy.io.wavfile as wavfile
            rate, data = wavfile.read(lfile)
            sl = slice(conf['params']['file']['offset'], conf['params']['file']['offset'] + conf['params']['file']['length'])
            self.data = {'x': data[sl]}
            print("data", data.shape, self.data['x'].shape)
            self.step = self.step_wav
            
        elif filetype == 'mp3':
            # FIXME:
            self.top._warning("Implement loading mp3 files without essentia")
            
            # # default samplerate
            # if not conf['params']['file'].has_key('samplerate'):
            #     conf['params']['file']['samplerate'] = 44100

            # # load data
            # loader = estd.MonoLoader(filename = lfile, sampleRate = conf['params']['file']['samplerate'])
            # data = loader.compute()

            # # if not length is given, create random slice of 60 sec minimal length if file length allows
            # if not conf['params']['file'].has_key('length') or conf['params']['file']['length'] is None or conf['params']['file']['length'] == 0:
            #     conf['params']['file']['length'] = min(
            #         data.shape[0],
            #         np.random.randint(
            #             conf['params']['file']['samplerate'] * 60,
            #             data.shape[0] - conf['params']['file']['offset']))

            # # compute slice
            # sl = slice(conf['params']['file']['offset'], conf['params']['file']['offset'] + conf['params']['file']['length'])
            # print "%sFileBlock2-%s fileypte mp3 sl = %s" % (self.nesting_indent, self.id, sl, )
            # # select data
            # self.data = {'x': data[sl]} # , 'y': data[sl]}
            # print "%sFileBlock2-%s data = %s, self.data['x'] = %s" % (self.nesting_indent, self.id, data.shape, self.data['x'].shape)
            # # set step callback
            # self.step = self.step_wav

    @decStep()
    def step(self, x = None):
        pass

    @decStep()
    def step_wav(self, x = None):
        # print "step_wav[%d]" % (self.cnt, )
        if (self.cnt % self.blocksize) == 0: # (self.blocksize - 1):
            for k, v in list(self.outputs.items()):                
                sl = slice(self.cnt-self.blocksize, self.cnt)
                # print "self.blocksize", self.blocksize
                # print "step_wav[%d].outputs[%s][%s] = %s (%s)" % (self.cnt, k, sl, self.data[k][sl].T.shape, self.file)
                outv_ = self.data[k][sl].T
                if outv_.shape[-1] < self.blocksize:
                    padwidth = self.blocksize - outv_.shape[-1]
                    outv_ = np.pad(outv_, pad_width = (padwidth, 0), mode = 'constant')
                setattr(self, k, outv_)
                # setattr(self, k, self.data[k][[self.cnt]].T)
        
    @decStep()
    def step_puppy(self, x = None):
        self.debug_print("%s.step_puppy: x = %s, bus = %s", (self.__class__.__name__, self.x, self.bus))
        # self.x = np.atleast_2d(self.data[[self.cnt]]).T #?
        # self.debug_print("self.x = %s", (self.x,))
        if (self.cnt % self.blocksize) == 0: # (self.blocksize - 1):
            for k, v in list(self.outputs.items()):                
                # sl = slice(self.cnt-self.blocksize, self.cnt)
                windowlen = v['shape'][-1]
                if self.cnt < windowlen: continue
                sl = slice(self.cnt-windowlen, self.cnt)
                # print "    step_puppy: self.cnt", self.cnt, "bs", self.blocksize, "win", windowlen, "sl", sl, "k", k, self.data[k][sl].T.shape
                setattr(self, k, self.data[k][sl].T)
                # setattr(self, k, self.data[k][[self.cnt]].T)
            
    @decStep()
    def step_sphero_res_learner(self, x = None):
        self.debug_print("%s.step: x = %s, bus = %s", (self.__class__.__name__, x, self.bus))
        # self.x = np.atleast_2d(self.data[[self.cnt]]).T #?
        self.debug_print("self.x = %s", (self.x,))
        if (self.cnt % self.blocksize) == 0: # (self.blocksize - 1):
            for k, v in list(self.outputs.items()):
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
            for k, v in list(self.outputs.items()):
                k_ = k
                sl = slice(self.cnt-self.blocksize, self.cnt)
                sl2 = slice(None)
                setattr(self, k, self.data[k_][sl,sl2])
                
    @decStep()
    def step_selflog(self, x = None):
        if (self.cnt % self.blocksize) == 0:
            for k, v in list(self.outputs.items()):
                # if k.startswith('conf'):
                storek = self.storekeys[k]
                # print "%s-%s.step[%d] key = %s, logdata.sh = %s" % (self.cname, self.id, self.cnt, k, self.store[storek].shape)
                # print self.store[k].values
                setattr(self, k, self.store[storek][self.cnt-self.blocksize:self.cnt].values.T)
                # print self.store[k][self.cnt-self.blocksize:self.cnt].values.T
                # print "%s-%s.step[%d] self.%s = %s" % (self.cname, self.id, self.cnt, k, getattr(self, k).shape)
        # for k in self.__dict__.keys(): #self.bus.keys():
            # k = k.replace("/", "_")
            # print "k", k
                        
    @decStep()
    def step_selflogconf(self, x = None):
        self.debug_print("%s.step: x = %s, bus = %s", (self.__class__.__name__, x, self.bus))
        if (self.cnt % self.blocksize) == 0:
            print(list(self.store.keys()))
            for k, v in list(self.outputs.items()):
                print("trying k = %s, v = %s" % (k, v))
                if k.startswith('conf'):
                    print("%s = %s\n" % (k, self.store.get_storer(k).attrs.conf,))


class SequencerBlock2(PrimBlock2):
    """SequencerBlock2

    Emit a predefined sequence of values, usually slowly changing constants
    specified by dict sequences of key (time): value: dict
    """
    defaults = {
        'block_group': 'data',
    }
    @decInit()
    def __init__(self, conf = {}, paren = None, top = None):

        self.sequences = conf['params']['sequences']
        conf['params']['outputs'] = {}

        for k, v in list(self.sequences.items()):
            if 0 in v['events']:
                val = v['events'][0]
            else:
                val = np.zeros(v['shape'])
            setattr(self, k, val)
            conf['params']['outputs'][k] = {'shape': v['shape']}
            
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)
        
    @decStep()
    def step(self, x = None):
        for k, v in list(self.sequences.items()):
            # print "k", k, "v", self.id
            if self.cnt in list(v['events'].keys()):
                val = np.ones(v['shape']) * v['events'][self.cnt]
                print("Sequencer out = %s changed to %s @%d" % (k, val, self.cnt))
                setattr(self, k, val)
    

# class sweepBlock2(PrimBlock2):
#     """sweepBlock2

#     generate a sweep of an n-dimensional space (meshgrid wrapper)
#     """

#     @decInit()
#     def __init__(self, conf = {}, paren = None, top = None):
#         Block2.__init__(self, conf = conf, paren = paren, top = top)

#     @decStep()
#     def step(self, x = None):
#         pass
