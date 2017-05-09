"""smp_graphs logging module

2017 Oswald Berthold

log_tb snatched from smpblocks
log_pd snatched from smq
"""

import numpy as np
import tables as tb

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
    experiment = "%s" % (config["id"])
    # filename
    tblfile = "data/%s.h5" % (experiment)
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
# pandas mediated tables logging
import pandas as pd

log_blocksize = 50
log_store = 0
log_lognodes = {}
log_logarray = {}
log_lognodes_idx = {}

def log_pd_init(config):
    """log_pd: log to tables via pandas, global init

Arguments:
    config: the global configuration dictionary
"""
    global log_store
    experiment = "%s" % (config['params']['id'])
    log_store = pd.HDFStore("data/%s_pd.h5" % (experiment))

def log_pd_store_config_initial(conf):
    # store the initial config for this run via the node attribute hack
    df_conf = pd.DataFrame([[0]])
    log_store['conf'] = df_conf
    log_store.get_storer('conf').attrs.conf = conf

def log_pd_store_config_final(conf):
    # store the final config for this run via the node attribute hack
    df_conf = pd.DataFrame([[0]])
    log_store['conf_final'] = df_conf
    log_store.get_storer('conf_final').attrs.conf = "" # (conf)
        
def log_pd_init_block(tbl_name, tbl_dim, tbl_columns = None, numsteps=100):
    """log_pd: log to tables via pandas, local node init

Arguments:
    tbl_name: node id storage key
     tbl_dim: node output dim for preallocation
 tbl_columns: node component labels
    numsteps: experiment number of steps for preallocation
"""
    global log_store, log_lognodes
    # print "logging.log_pd_init_block: adding %s to log_lognodes with columns %s" % (tbl_name, tbl_columns)
    log_lognodes[tbl_name] = pd.DataFrame(columns=tbl_columns, index = range(numsteps), dtype=float)
    log_logarray[tbl_name] = np.zeros((len(tbl_columns), log_blocksize))
    # print "log_tables.shape", log_lognodes[tbl_name].shape
    log_lognodes_idx[tbl_name] = 0

def log_pd_store():
    # store logs, FIXME incrementally
    global log_lognodes, log_store
    for k,v in log_lognodes.items():
        # print "storing table k = %s with data type = %s" % (k, v)
        log_store[k] = v

def log_pd(tbl_name, data):
    """log_pd: log to tables via pandas, local node logging

Arguments:
    tbl_name: node id storage key
      data: the data as a dim x 1 numpy vector
"""
    global log_lognodes, log_lognodes_idx, log_blocksize
    # print "data.shape", data.flatten().shape, log_lognodes_idx[tbl_name]
    # infer blocksize from data
    blocksize = data.shape[1]
    # get last index
    cloc = log_lognodes_idx[tbl_name]
    # print "cloc", cloc
    # print "log_lognodes[tbl_name].loc[cloc].shape = %s, data.shape = %s" % (log_lognodes[tbl_name].loc[cloc].shape, data.shape)
    # using flatten to remove last axis, FIXME for block based logging
    # print "logging.log_pd: data.shape", data.shape, cloc, cloc + blocksize - 1, "bs", blocksize
    # print log_logarray[tbl_name][:,[-1]].shape, data.shape
    log_logarray[tbl_name][:,[-1]] = data
    np.roll(log_logarray[tbl_name], shift = -1, axis = 1)

    if cloc % log_blocksize == 0:
        # sl = slice(cloc, cloc + blocksize - 1)
        sl = slice(cloc, cloc + log_blocksize - 1)
        # print "logging.log_pd: log.shape at sl", sl, log_lognodes[tbl_name].loc[sl].shape
        # log_lognodes[tbl_name].loc[sl] = data.T # data.flatten()
        log_lognodes[tbl_name].loc[sl] = log_logarray[tbl_name].T # data.flatten()
    # log_lognodes[tbl_name].loc[0] = 1
    # print "log_lognodes[tbl_name]", log_lognodes[tbl_name], log_lognodes[tbl_name].loc[cloc]
    log_lognodes_idx[tbl_name] += blocksize

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
