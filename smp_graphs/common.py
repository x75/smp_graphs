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
