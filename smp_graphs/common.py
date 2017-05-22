
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
randseed = 0
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
        'randseed': randseed,
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
        return None
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
