"""smp_graphs logging module

log_tb snatched from smpblocks
log_pd snatched from smq
"""

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

log_store = 0
log_lognodes = {}
log_lognodes_idx = {}

def log_pd_init(config):
    """log_pd: log to tables via pandas, global init

Arguments:
    config: the global configuration dictionary
"""
    global log_store
    experiment = "%s" % (config["id"])
    log_store = pd.HDFStore("data/%s_pd.h5" % (experiment))

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
    log_lognodes_idx[tbl_name] = 0

def log_pd_store():
    # store logs, FIXME incrementally
    global log_lognodes, log_store
    for k,v in log_lognodes.items():
        # print "storing table k = %s with data type = %s" % (k, v)
        log_store[k] = v
    
def log_pd(nodeid, data):
    """log_pd: log to tables via pandas, local node logging

Arguments:
    nodeid: node id storage key
      data: the data as a dim x 1 numpy vector
"""
    global log_lognodes, log_lognodes_idx
    # print "data.shape", data.flatten().shape, log_lognodes_idx[nodeid]
    cloc = log_lognodes_idx[nodeid]
    log_lognodes[nodeid].loc[cloc] = data.flatten()
    # log_lognodes[nodeid].loc[0] = 1
    # print "log_lognodes[nodeid]", log_lognodes[nodeid], log_lognodes[nodeid].loc[cloc]
    log_lognodes_idx[nodeid] += 1
    
