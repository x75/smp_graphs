
import traceback
import sys
import pickle

from collections import OrderedDict

from smp_base.common import set_attr_from_dict
from smp_graphs.utils import print_dict

################################################################################
# static config templates
conf_header = """from smp_graphs.experiment import make_expr_id

from collections import OrderedDict
from functools import partial

from smp_graphs.block import Block2, ConstBlock2, CountBlock2, UniformRandomBlock2
from smp_graphs.block import FuncBlock2, LoopBlock2, SeqLoopBlock2
from smp_graphs.block_ols import FileBlock2
from smp_graphs.block_plot import PlotBlock2

from smp_base.plot import timeseries, histogram, rp_timeseries_embedding

import numpy as np

debug = False
showplot = True
saveplot = False
randseed = 0
recurrent = False
ros = True
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
        'ros': ros,
    }
}
"""


def get_config_raw(conf, confvar = 'conf'):
    """base.common.py

    open config file, read it and call get_config_raw_from_string on that string
    """
    # open and read config file containing a python dictionary
    try:
        s_ = open(conf, "r").read()
    except Exception, e:
        print e
        sys.exit(1)

    # compile and evaluate the dictionary code string and return the dict object
    return get_config_raw_from_string(s_, confvar = confvar)

def get_config_raw_from_string(conf, confvar = 'conf'):
    """base.common.get_config_raw_from_string

    Compile the 'conf' string and return the resulting 'confvar' variable
    """
    # prepend / append header and footer
    # FIXME: this is smp_graphs specific
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

def get_input(inputs, inkey):
    """smp_graphs.common.get_input

    An smp_graphs bus operation: return the 'val' field of the inputs' item at 'inkey'
    """
    assert type(inputs) is dict
    assert inkey is not None
    assert inputs.has_key(inkey)
    return inputs[inkey]['val']

def dict_search_recursive(d, k):
    """smp_graphs.common.dict_search_recursive

    Search for the presence of key k recursively over nested smp_graph config dicts
    """
    # FIXME: make it generic recursive search over nested graphs and move to smp_base
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
    """smp_graphs.common.dict_replace_idstr_recursive

    Replace all references in dict 'd' to id 'cid' in the dictionary with 'cid|xid'
    """
    assert d.has_key('params')
    # assert d['params'].has_key('id')

    # print "dict_replace_idstr_recursive", print_dict(d)
    
    if cid is not None:
        d['params']['id'] = "%s|%s" % (cid, xid)

    if d['params'].has_key('graph') and type(d['params']['graph']) is not str:
        tgraph = OrderedDict()
        for k, v in d['params']['graph'].items():
            # v['params']['id'] = k

            # replace ids in dict val
            v = dict_replace_idstr_recursive(d = v, cid = k, xid = xid)
            # replace ids in dict key
            k_ = v['params']['id']
            # reassemble
            tgraph[k_] = v
            d['params']['graph'][k] = v
        # copy new dict to graph
        # print "tgraph", tgraph
        d['params']['graph'] = tgraph
        # print "d['params']['graph']", d['params']['graph']
    return d

def dict_replace_nodekeys(d, xid, idmap = {}):
    """smp_graphs.common.dict_replace_nodekeys

    replace all keys in d by a copy with xid appended (looping) and store key/replacement in idmap

    Returns:
    - d: dictionary
    - idmap: map old -> new ids
    """

    # loop over dictionary items
    for k, v in d.items():
        # print "dict_replace_nodekeys: k = %s, v = %s, idmap = %s" % (k, v.keys(), idmap)
        # new id from old id
        k_ = "%s|%s" % (k, xid)
        # fix key:
        d[k_] = d.pop(k) # FIXME: does this confuse .items(), FIXME: even worse, does this confuse the order in OrderedDict?
        # print "k_", k_, "v_", d[k_].keys()

        # # debug
        # if not d[k_].has_key('params'):
        #     print "\n\n\n\n\nd[k_]", d[k_].keys()
        #     return d, idmap
        
        # fix block id
        d[k_]['params']['id'] = k_
        idmap[k] = k_
        
        # descend into graph hierarchy
        if d[k_]['params'].has_key('graph'):
            d[k_]['params']['graph'], idmap = dict_replace_nodekeys(d[k_]['params']['graph'], xid, idmap)

        # descend into loopblock
        if d[k_]['params'].has_key('loopblock') and d[k_]['params']['loopblock']['params'].has_key('graph'):
            d[k_]['params']['loopblock']['params']['graph'], idmap = dict_replace_nodekeys(d[k_]['params']['loopblock']['params']['graph'], xid, idmap)
            # print "\n\n\n\n\n\n\n", d[k_]['params']['loopblock']['params']['graph']

    # return to surface
    return d, idmap

def dict_replace_nodekeyrefs(d, xid, idmap):
    """smp_graphs.common.dict_replace_nodekeyrefs

    replace all references to ids in idmap.keys with corresponding idmap.values

    Returns:
    - d: dictionary
    - idmap: map old -> new ids
    """
    
    # loop over dictionary items
    for k_, v in d.items():
        # fix bus references in inputs
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
                    
        # fix bus references in outputs
        if d[k_]['params'].has_key('outputs'):
            for ink, inv in [(ink, inv) for ink, inv in d[k_]['params']['outputs'].items() if inv.has_key('buscopy')]:
                # if inv.has_key('bus'):
                invbuss = inv['buscopy'].split("/")
                if invbuss[0] in idmap.keys():
                    # print "ink, inv", ink, inv
                    # print "idmap", idmap
                    buskey = "%s/%s" % (idmap[invbuss[0]], invbuss[1])
                    print "dict_replace_nodekeyrefs: replacing %s with %s in node %s" % (inv['buscopy'], buskey, k_)
                    inv['buscopy'] = buskey
                print "\n\n\n\n\n\n\noutputs, buscopy", d[k_]['params']

        # fix id references in 'copy' models (want to copy the model from the block with that id)
        if d[k_]['params'].has_key('models'):
            for ink, inv in [(ink, inv) for ink, inv in d[k_]['params']['models'].items() if inv.has_key('copyid')]:
                inv['copyid'] = idmap[inv['copyid']]
            # print "\n\n\n\n\n\nfound model", d[k_]['params']['id'], d[k_]['params']['models']

        # descend into graph hierarchy
        if d[k_]['params'].has_key('graph'):
            d[k_]['params']['graph'], idmap = dict_replace_nodekeyrefs(d[k_]['params']['graph'], xid, idmap)

        # descend into loopblock
        if d[k_]['params'].has_key('loopblock') and d[k_]['params']['loopblock']['params'].has_key('graph'):
            for ink, inv in d[k_]['params']['loopblock']['params']['outputs'].items():
                if inv.has_key('buscopy'):
                    invbuss = inv['buscopy'].split("/")
                    if invbuss[0] in idmap.keys():
                        # print "ink, inv", ink, inv
                        # print "idmap", idmap
                        buskey = "%s/%s" % (idmap[invbuss[0]], invbuss[1])
                        print "dict_replace_nodekeyrefs: replacing %s with %s in node %s" % (inv['buscopy'], buskey, k_)
                        inv['buscopy'] = buskey
                        
            d[k_]['params']['loopblock']['params']['graph'], idmap = dict_replace_nodekeyrefs(d[k_]['params']['loopblock']['params']['graph'], xid, idmap)
            # print "\n\n\n\n\n\n\n", d[k_]['params']['loopblock']['params']['graph']

    # return to surface
    return d, idmap

def dict_replace_idstr_recursive2(d, xid, idmap = {}):
    """smp_graphs.common.dict_replace_idstr_recursive2

    replace occurences of an 'id' string with a numbered version for looping

    Two steps
    1 - replace the dict keys
    2 - replace all references in the conf to dict keys

    Args:
    - d: dictionary
    - xid: extension to append to existing id
    - idmap: map of ids that have been modified

    Return:
    - d: the modified dict
    """

    d, idmap = dict_replace_nodekeys(d, xid, idmap)

    d, idmap = dict_replace_nodekeyrefs(d, xid, idmap)
        
    return d
            
def read_puppy_hk_pickles(lfile, key = None):
    """smp_graphs.common.read pickled log dicts from andi's puppy experiments"""
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

