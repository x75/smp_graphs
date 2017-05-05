################################################################################
# static config templates
conf_header = """
from smp_graphs.experiment import make_expr_id

from collections import OrderedDict

from smp_graphs.block import Block2, ConstBlock2, UniformRandomBlock2
from smp_graphs.block import LoopBlock2
from smp_graphs.block import FileBlock2
from smp_graphs.block_plot import TimeseriesPlotBlock2

from smp_base.plot import timeseries, histogram, rp_timeseries_embedding

import numpy as np

debug = False
"""

conf_footer = """
# top block
conf = {
    'block': Block2,
    'params': {
        'id': make_expr_id(),
        'debug': debug,
        'topblock': True,
        'numsteps': numsteps,
        'graph': graph,
    }
}
"""


def get_config_raw(conf, confvar = 'conf'):
    # open and read config file containing a dictionary spec of the graph
    s_ = open(conf, "r").read()

    # prepend / append header and footer
    s   = "%s\n%s\n%s" % (conf_header, s_, conf_footer)

    # load config by running the code string
    code = compile(s, "<string>", "exec")
    global_vars = {}
    local_vars  = {}
    exec(code, global_vars, local_vars)

    # return resulting variable
    return local_vars[confvar]

