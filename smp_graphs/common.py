
import traceback
import sys
import pickle

################################################################################
# static config templates
conf_header = """from smp_graphs.experiment import make_expr_id

from collections import OrderedDict
from functools import partial

from smp_graphs.block import Block2, ConstBlock2, CountBlock2, UniformRandomBlock2
from smp_graphs.block import FuncBlock2, LoopBlock2
from smp_graphs.block_ols import FileBlock2
from smp_graphs.block_plot import PlotBlock2

from smp_base.plot import timeseries, histogram, rp_timeseries_embedding

import numpy as np

debug = False
showplot = True
saveplot = False
randseed = 0
recurrent = False
"""

conf_footer = """# top block
conf = {
    'block': Block2,
    'params': {
        'id': make_expr_id(),
        'debug': debug,
        'topblock': True,
        'numsteps': numsteps,
        'graph': graph,
        'showplot': showplot,
        'saveplot': saveplot,
        'randseed': randseed,
        'recurrent': recurrent,
    }
}
"""


def get_config_raw(conf, confvar = 'conf'):
    """open config file, read it and call get_config_raw_from_string on that string"""
    # open and read config file containing a dictionary spec of the graph
    try:
        s_ = open(conf, "r").read()
    except Exception, e:
        print e
        sys.exit(1)

    return get_config_raw_from_string(s_, confvar = confvar)

def get_config_raw_from_string(conf, confvar = 'conf'):
    """compile the 'conf' string and return the resulting 'confvar' variable"""
    # prepend / append header and footer
    lineoffset = 0
    lineoffset += conf_header.count('\n')
    lineoffset += conf_footer.count('\n')
    s   = "%s\n%s\n%s" % (conf_header, conf, conf_footer)

    # load config by compiling and running the configuration code

    # make sure to catch syntax errors in the config for debugging
    try:
        code = compile(s, "<string>", "exec")
    except Exception, e:
        print "\n%s" % (e.text)
        print "Compilation of %s failed with %s in %s at line %d" % (conf, e.msg, e.filename, e.lineno)
        sys.exit(1)

    # prepare variables
    global_vars = {}
    local_vars  = {}
    
    # run the code
    try:
        exec(code, global_vars, local_vars)
    except Exception, e:
        # FIXME: how to get more stack context?
        traceback.print_exc(limit = 10)
        # print traceback
        print "Error running config code: %s" % (repr(e),)
        print "Probably causes:\n    missing parentheses or comma\n    dict key followed by , instead of :\nin config"
        sys.exit(1)

    # return resulting variable
    return local_vars[confvar]

def set_attr_from_dict(obj, dictionary):
    """set object attribute k = v from a dictionary's k, v for all dict items"""
    for k,v in dictionary.items():
        setattr(obj, k, v)

def dict_search_recursive(d, k):
    """dict_search_recursive

    search for the presence of the key k recursively over a nested smp_graph config dicts
    """
    # print "d", d, "k", k
    print "#" * 80
    print "searching k = %s " % (k,),
    if d.has_key(k):
        print "found k = %s, params = %s" % (k, d[k]['params'].keys())
        return d[k]
    else:
        print "d.keys()", d.keys()
        for k_, v_ in d.items():
            # if v_[
            if v_['params'].has_key('graph'): #  or v_['params'].has_key('subgraph'):
                print "k_", k_, "v_", v_['params'].keys()
                return dict_search_recursive(v_['params']['graph'], k)
    # None found
    return None

def dict_replace_idstr_recursive(d, cid, xid):
    """dict_replace_idstr_recursive

    replace all id references in the dict with id + parent information
    """
    assert d.has_key('params')
    # assert d['params'].has_key('id')
    
    if cid is not None:
        d['params']['id'] = "%s_%s" % (cid, xid)

    if d['params'].has_key('graph'):
        for k, v in d['params']['graph'].items():
            # v['params']['id'] = k
            v = dict_replace_idstr_recursive(d = v, cid = k, xid = xid)
            d['params']['graph'][k] = v

    return d

def dict_replace_nodekeys(d, xid, idmap = {}):
    # idmap = {}
    for k, v in d.items():
        # print "dict_replace_nodekeys: k = %s, v = %s, idmap = %s" % (k, v.keys(), idmap)
        k_ = "%s_%s" % (k, xid)
        # fix key
        d[k_] = d.pop(k) # FIXME: does this confuse .items()
        # print "k_", k_, "v_", d[k_].keys()
        # fix block id
        d[k_]['params']['id'] = k_
        idmap[k] = k_
        
        # descend
        if d[k_]['params'].has_key('graph'):
            d[k_]['params']['graph'], idmap = dict_replace_nodekeys(d[k_]['params']['graph'], xid, idmap)

    # ascend
    return d, idmap

def dict_replace_nodekeyrefs(d, xid, idmap):
    for k_, v in d.items():
        # fix bus references
        if d[k_]['params'].has_key('inputs'):
            for ink, inv in [(ink, inv) for ink, inv in d[k_]['params']['inputs'].items() if inv.has_key('bus')]:
                # if inv.has_key('bus'):
                invbuss = inv['bus'].split("/")
                if invbuss[0] in idmap.keys():
                    # print "ink, inv", ink, inv
                    # print "idmap", idmap
                    buskey = "%s/%s" % (idmap[invbuss[0]], invbuss[1])
                    print "dict_replace_nodekeyrefs: replacing %s with %s in node %s" % (inv['bus'], buskey, k_)
                    inv['bus'] = buskey
        # descend
        if d[k_]['params'].has_key('graph'):
            d[k_]['params']['graph'], idmap = dict_replace_nodekeyrefs(d[k_]['params']['graph'], xid, idmap)

    # ascend
    return d, idmap

def dict_replace_idstr_recursive2(d, xid, idmap = {}):

    d, idmap = dict_replace_nodekeys(d, xid, idmap)

    d, idmap = dict_replace_nodekeyrefs(d, xid, idmap)
        
    return d
            
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
    # print "x.shape", data.keys()
    return (data, rate, offset, d['x'].shape[0])
