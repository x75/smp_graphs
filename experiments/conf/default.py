
from smp_graphs.experiment import make_expr_id

from collections import OrderedDict

from smp_graphs.block import Block, ConstBlock, UniformRandomBlock
from smp_graphs.block import TimeseriesPlotBlock

numsteps = 100

# graph = [
graph = OrderedDict([
    ("b1", {
        'block': ConstBlock,
        'params': {
            'id': 'b1',
            'idim': None,
            'odim': 3,
            'const': 1.3
        },
    }),
    ("b2", {
        'block': UniformRandomBlock,
        'params': {
            'id': 'b2',
            'idim': 6,
            'odim': 3,
            'lo': 0,
            'hi': 1,
            'inputs': ['b1']
        },
    }),
    ("bplot", {
        'block': TimeseriesPlotBlock,
        'params': {
            'id': 'bplot',
            'blocksize': numsteps,
            'idim': 6,
            'odim': 3,
            'inputs': ['b1', 'b2']
        }
    })
])

conf = {
    'block': Block,
    'params': {
        'id': make_expr_id(),
        'topblock': True,
        "numsteps": numsteps,
        "graph": graph
    }
}
