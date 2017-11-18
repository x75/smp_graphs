"""smp_graphs logging module

.. moduleauthor:: 2017 Oswald Berthold

Log variables are global in smp_graphs. There are two logging mechanisms
 - :func:`log_tb` logs to hdf5 using pytables, snatched from :mod:`smpblocks`
 - :func:`log_pd` logs to hdf5 using pandas, originally from :mod:`smq`

.. warning::
 - Merge this with python's own :mod:`logging`
 - Merge with more general output layer idea

"""

import numpy as np
import tables as tb

from logging import INFO as logging_INFO
from logging import CRITICAL as logging_CRITICAL
from smp_base.common import get_module_logger

loglevel_DEFAULT = logging_CRITICAL
logger = get_module_logger(modulename = 'logging', loglevel = loglevel_DEFAULT)

# declare global h5 file handle
h5file = 0
loginit = True
lognodes = {}

################################################################################
# raw tables logging
def log_tb_init(config):
    """log_tb: log to tables

    log init function: global, called from topblock init
    """
    # global handles
    global h5file, loginit, lognodes
    # experiment signature
    # experiment = "%s" % (config["id"])
    # filename
    tblfile = '%s_log_tb.h5' % (config['params']['datafile_expr'], )
    # tblfile = "data/%s.h5" % (experiment)
    # h5 file
    h5file  = tb.open_file(tblfile, mode = "w", title = "%s" % (experiment))
    root = h5file.root
    storage_version = "v2"

    # create VLArray for storing the graph configuration
    conf_array = h5file.create_vlarray(root, 'conf', tb.VLStringAtom(),
                               "Variable Length Config String")
    conf_array.append(str(config))
    # FIXME: log git commit

    # create topblock array
    # a = tb.Float64Atom()
    # lognodes[parent.id] = h5file.create_earray(root, "%s_obuf" % (parent.id), a, (parent.odim, 0))
    # # create arrays for each node's data
    # for nodek, nodev in nodes.items():
    #     print("init_log: node", nodek)
    #     a = tb.Float64Atom()
    #     # node_fieldkeys = 
    #     # for node_fieldkey in ["%s_%03d" % (nodek, i) for i in range(nodev.odim)]:
    #     #     tdef[node_fieldkey] = tb.Float32Col()
    #     # tdef[nodek] = tb.Float32Col(shape=(nodev.odim, 1))
    #     # return tdef
    #     lognodes[nodev.id] = h5file.create_earray(root, "%s_obuf" % (nodev.id), a, (nodev.odim, 0))
    # loginit = True # this doesn't work yet persistently, why?
    loginit = True
    # print("initlog done")

def log_tb_init_block(tbl_name, tbl_dim, tbl_columns = None):
    """log_tb: log to tables

log init function: local to each block / node in the graph
"""
    # print("%s.log_tb_init_block" % ("logging"), tbl_name, tbl_dim)
    # global handles
    global loginit, h5file, lognodes
    # global init done?
    if loginit:
        # prepare data structure
        a = tb.Float64Atom()
        lognodes[tbl_name] = h5file.create_earray(h5file.root, "item_%s_data" % (tbl_name), a, (tbl_dim, 0))
        
def log_tb(nodeid, data):
    """log_tb: log to tables

Global logging method, like in python logging: take the data, write it
into an associated array

Arguments:
 - nodeid: node id used as key in storage dict
 -   data: data as a dim x 1 numpy vector
"""
    # FIXME: make a dummy log method an overwrite it on loginit with the real function body?
    # print("id: %s, data: %s" % (id, str(data)[:60]))
    # print("log")
    # print("lognode", lognodes[nodeid])
    # print("data shape", data.shape)
    lognodes[nodeid].append(data)

################################################################################
# pandas tables based logging
import pandas as pd

log_store = 0
log_lognodes_idx = {}
log_lognodes     = {}
log_logarray     = {}
log_blocksize    = {}
log_lognodes_blockidx = {}

def log_pd_init(config):
    """log_pd_init

    global init for logging to hdf5 tables via pandas

    Arguments:
        - config: the global configuration dictionary
    """
    global log_store
    log_store_filename = '%s_log_pd.h5' % (config['params']['datafile_expr'], )
    log_store = pd.HDFStore(log_store_filename)
    # experiment = "%s" % (config['params']['id'])
    # log_store = pd.HDFStore("data/%s_pd.h5" % (experiment))
    # print "logging.log_pd_init log_store.root = %s" % (log_store.root)

def log_pd_store_config_initial(conf):
    """log_pd_store_config_initial

    store the initial config in a separate table '/conf'

    Arguments:
        - conf: configuration dictionary
    """
    # # table attribute hack
    # df_conf = pd.DataFrame([[0]])
    # log_store['conf'] = df_conf
    # print "logging.log_pd_store_config_initial conf = %s" % conf
    # log_store.get_storer('conf').attrs.conf = conf

    # get h5 handle from pandas    
    h5file = log_store._handle
    # create VLArray for storing the graph configuration
    conf_array = h5file.create_vlarray(log_store.root, 'conf', tb.VLStringAtom(),
                               "Variable Length Config String")
    # store it
    conf_array.append(str(conf))


def log_pd_store_config_final(conf):
    # store the final config for this run via the node attribute hack
    # df_conf = pd.DataFrame([[0]])
    # log_store['conf_final'] = df_conf
    # log_store.get_storer('conf_final').attrs.conf = "" # (conf)
    h5file = log_store._handle
    conf_array = h5file.create_vlarray(log_store.root, 'conf_final', tb.VLStringAtom(),
                               "Variable Length Config String")
    conf_array.append(str(conf))
        
def log_pd_init_block(tbl_name, tbl_dim, tbl_columns = None, numsteps=100, blocksize = 1):
    """log_pd_init_block

    log to tables via pandas, local node init

    Arguments:
        - tbl_name: node id storage key
        - tbl_dim: busitem output shape for preallocation
        - tbl_columns: node component labels
        - numsteps: experiment number of steps for preallocation
        - blocksize: blocksize
    """
    global log_store, log_lognodes, log_lognodes_idx, log_blocksize, log_lognodes_blockidx
    # np.prod(tbl_dim[:-1]), # flattened dim without blocksize

    # print "logging.log_pd_init_block: adding %s to log_lognodes with columns %s" % (tbl_name, tbl_columns)
    log_lognodes[tbl_name] = pd.DataFrame(columns=tbl_columns, index = range(numsteps), dtype=float)
    # log_logarray[tbl_name] = np.zeros((len(tbl_columns), log_blocksize))
    # log_logarray[tbl_name] = np.zeros((len(tbl_columns), numsteps))
    log_logarray[tbl_name] = np.zeros((np.prod(tbl_dim[:-1]), numsteps))
    # print "log_tables.shape", log_lognodes[tbl_name].shape
    log_lognodes_idx[tbl_name] = 0
    log_lognodes_blockidx[tbl_name] = 0
    # logging blocksize, FIXME: enforce max and integer multiple relation
    # log_blocksize[tbl_name]    = max(50, blocksize)
    if blocksize == 1:
        log_blocksize[tbl_name]    = 50
    else:
        log_blocksize[tbl_name]    = blocksize
        
    # store original shape in table attribute
    # FIXME: doesn't seem to work?
    # FIXME: this doesnt work because pandas doesnt propagate the table attributes when it grows the memory?
    log_store[tbl_name] = log_lognodes[tbl_name]
    
    # print "log_pd_init_block: log_store.get_storer(tbl_name).attrs.shape", log_store.get_storer(tbl_name).attrs.shape
    # print "log_pd_init_block: log_store.get_storer(tbl_name).attrs.numsteps", log_store.get_storer(tbl_name).attrs.numsteps

def log_pd_init_block_attr(tbl_name, tbl_dim, numsteps, blocksize):
    global log_store
    # log_store.get_storer(tbl_name).attrs.datashape = tbl_dim[:-1]
    # log_store.get_storer(tbl_name).attrs.numsteps = numsteps
    # log_store.get_storer(tbl_name).attrs.blocksize = blocksize

    # pytables
    h5file = log_store._handle
    # print "listnodes[%s] = %s" % (tbl_name, h5file.list_nodes(log_store.root))
    # print "node[%s] = %s" % (tbl_name, h5file.get_node(log_store.root, tbl_name))
    node = h5file.get_node(h5file.root, tbl_name)
    h5file.set_node_attr(node, 'datashape', tbl_dim)
    h5file.set_node_attr(node, 'numsteps', numsteps)
    h5file.set_node_attr(node, 'blocksize', blocksize)
    # print "node[%s].attr = %s" % (tbl_name, h5file.get_node_attr(log_store.root, 'datashape', tbl_name), )
    # print "node[%s].attr = %s" % (tbl_name, h5file.get_node_attr(log_store.root, tbl_name, 'numsteps'), )
    # print "node[%s].attr = %s" % (tbl_name, h5file.get_node_attr(log_store.root, tbl_name, 'blocksize'), )
    h5file.flush()

def log_pd_deinit():
    global log_store
    log_store.close()
        
def log_pd_store():
    # store logs, FIXME incrementally
    global log_lognodes, log_store
    for k,v in log_lognodes.items():
        # if 'b4/' in k:
        if np.any(np.isnan(v)):
            logger.warning("storing nan values, table k = %s with data type = %s" % (k, type(v)))
        log_store[k] = v

def log_pd(tbl_name, data):
    """log_pd: log to tables via pandas, local node logging

Arguments:
    tbl_name: node id storage key
      data: the data as a dim x 1 numpy vector
"""
    global log_lognodes, log_lognodes_idx, log_blocksize, log_logarray, log_lognodes_blockidx
    assert log_lognodes_idx.has_key(tbl_name), "Logtable key = %s is not in keys = %s" % (tbl_name, log_lognodes_idx.keys())
    # print "log_pd tbl_name = %s, data.shape = %s" % (tbl_name, data.flatten().shape), log_lognodes_idx[tbl_name]
    # infer blocksize from data
    blocksize = data.shape[-1]
    assert blocksize > 0, \
      "logging.log_pd: table = %s's blocksize > 1 false with blocksize = %d\n    probably wrong output of source block %s" \
      % (tbl_name, blocksize, tbl_name, )
      
    # get last index
    cloc = log_lognodes_idx[tbl_name]

    # 20171106 first data point compute vs. log
    # start count mangling
    # if cloc == 0 and blocksize == 1: # we will see cloc == 1 only with blocksize 1 anyway?
    #     cloc_ = 0
    #     # log_lognodes_idx[tbl_name] = 1
    #     # bsinc = blocksize - 1
    # else:
    #     cloc_ = cloc
        
    # if cloc == 0:
    #     cloc += 1
    #     # sl1 += 1
    #     log_lognodes_idx[tbl_name] = 1
    #     # sl1 = 1
    #     # sl2 = cloc + blocksize - 1
    #     # if blocksize > 1:

    sl1 = cloc
    sl2 = cloc + blocksize
    # sl2 -= sl2 % blocksize

    # else:
    # print "cloc", cloc
    # print "log_lognodes[tbl_name].loc[cloc].shape = %s, data.shape = %s" % (log_lognodes[tbl_name].loc[cloc].shape, data.shape)
    # using flatten to remove last axis, FIXME for block based logging
    # print "logging.log_pd: data.shape", data.shape, cloc, cloc + blocksize - 1, "bs", blocksize
    # print log_logarray[tbl_name][:,[-1]].shape, data.shape

    # log_logarray[tbl_name][:,[-1]] = data
    # np.roll(log_logarray[tbl_name], shift = -1, axis = 1)

    # always copy current data into array
    sl = slice(sl1, sl2)
    # print "logging: sl", sl
    # assert len(data.shape) == 2, "data of %s is multidimensional array with shape %s, not fully supported yet" % (tbl_name, data.shape)

    # print "%s log_pd sl = %s, data.shape = %s" % (tbl_name, sl, data.shape)
    # if cloc == 1 and blocksize > 1:
    #     log_logarray[tbl_name][:,sl] = data[:,1:].copy()
    # else:
    # print "logging: tbl_name", tbl_name, "log_pd data.shape", data.shape, "blocksize", blocksize
    tmplogdata = data.copy().reshape((-1, blocksize))
    # print "tmplogdata.sh", tmplogdata.shape

    assert log_logarray[tbl_name][:,sl].shape == tmplogdata.shape, \
      "logging.log_pd: Block output %s's shape %s doesn't agree with logging shape %s, sl1 = %s, sl2 = %s" % (
          tbl_name, log_logarray[tbl_name][:,sl].shape, tmplogdata.shape, sl1, sl2)

    # if 'b4/' in tbl_name:
    #     print "logging b4", tbl_name, sl, tmplogdata, log_logarray[tbl_name].shape
    
    log_logarray[tbl_name][:,sl] = tmplogdata # to copy or not to copy?

    # if 'b4/' in tbl_name:
    #     print "logging b4", tbl_name, log_logarray[tbl_name]

    # if logging blocksize aligns with count
    # if cloc % log_blocksize[tbl_name] == 0:
    # 20171106 first data point compute vs. log
    # 20171115 just cloc after step decorator pre-increment fix
    if (cloc + 0) % log_blocksize[tbl_name] == 0:
        dloc = log_lognodes_blockidx[tbl_name]
        pdsl = slice(dloc, dloc + log_blocksize[tbl_name] - 1)
        sl = slice(dloc, dloc + log_blocksize[tbl_name])
        # print "logging.log_pd: tbl_name = %s, log.shape at sl = %s" % (tbl_name, sl), log_lognodes[tbl_name].loc[pdsl].shape, log_logarray[tbl_name][:,sl].T.shape
        # log_lognodes[tbl_name].loc[sl] = data.T # data.flatten()
        log_lognodes[tbl_name].loc[pdsl] = log_logarray[tbl_name][:,sl].T # data.flatten()
        log_lognodes_blockidx[tbl_name] += log_blocksize[tbl_name]
    # log_lognodes[tbl_name].loc[0] = 1
    # print "log_lognodes[tbl_name]", log_lognodes[tbl_name], log_lognodes[tbl_name].loc[cloc]
    log_lognodes_idx[tbl_name] += blocksize
    
    # if 'b4/' in tbl_name:
    #     print "logging b4 post log_blocksize", tbl_name, log_logarray[tbl_name]

def log_pd_dump_config(h5store, storekey = None):
    assert h5store is not None
    assert storekey is not None
    store = pd.HDFStore(h5store)
    try:
        ret = store.get_storer(storekey).attrs.conf
    except AttributeError:
        print "key %s doesn't exist" % (storekey)
        ret = None
    store.close()
    return ret
    
# def log_pd_dump_log(h5store = None):
#     assert h5store is not None
#     store = pd.HDFStore(h5store)

#     print "initial config = %s" % store.get_storer('conf').attrs.conf
