"""smp_graphs: configuration for testing the sequential loop block (SeqLoopBlock2)

explain
 - CountBlock2 drives two function blocks to test FuncBlock2 evaluation
 - SeqLoopBlock2 is driven by the 'loop' object which can be either an
   array of (variable group, {variable name, value}) tuples or a generator
   function returning the same kind of tuple on each call
"""

from smp_graphs.block import CountBlock2, FuncBlock2
from smp_graphs.block import SeqLoopBlock2
from smp_graphs.funcs import *

from functools import partial

# reused variables
loopblocksize = 1
loopsteps = 2000
numsteps = loopsteps * loopblocksize
# showplot = False

# singular block specs for use in loop block
loopblock1 = {
    'block': UniformRandomBlock2,
    'params': {
        'id': 'b3',
        'inputs': {
            'lo': {'val': np.random.uniform(-1, 0, (3, 1)), 'shape': (3, )},
            'hi': {'val': np.random.uniform(0.1, 1, (3, 1)), 'shape': (3, )},
        },
        'outputs': {'y': {'shape': (3,)}},
        'debug': False,
        'blocksize': 1,
    },
}

loopblock = {
    'block': FuncBlock2,
    'params': {
        'id': 'f3',
        'inputs': {'x': {'val': np.random.uniform(-1, 1, (3, 1)), 'shape': (3, )}},
        'outputs': {'x': {'shape': (3, )}, 'y': {'shape': (1, )}},
        'func': f_sinesquare4,
        'blocksize': loopblocksize,
        'debug': False,
        'logging': False,
        },
    }
    
# top graph
graph = OrderedDict([
    # a constant
    ("b1", {
        'block': CountBlock2,
        'params': {
            'id': 'b1',
            'inputs': {},
            'outputs': {'x': {'shape': (1,)}},
            'scale': np.pi/float(numsteps),
            'offset': np.pi/2,
            'blocksize': 1,
            'debug': False,
        },
    }),
    
    # a function
    ("f1", {
        'block': FuncBlock2,
        'params': {
            'id': 'f1',
            'inputs': {'x': {'bus': 'b1/x'}},
            'outputs': {'x': {'shape': (1,)}, 'y': {'shape': (1,)}},
            'func': f_sinesquare2,
            'blocksize': 1,
            'debug': False,
        },
    }),
    
    # a function
    ("f2", {
        'block': FuncBlock2,
        'params': {
            'id': 'f2',
            'inputs': {'x': {'bus': 'b1/x'}},
            'outputs': {'x': {'shape': (1,)}, 'y': {'shape': (1,)}},
            'func': f_sinesquare4,
            'blocksize': 1,
            'debug': False,
        },
    }),
    
    # a loop block calling the enclosed block len(loop) times,
    # returning data of looplength in one outer step
    ("hpo", {
        'block': SeqLoopBlock2,
        'params': {
            'id': 'hpo',
            # loop specification, check hierarchical block to completely
            # pass on the contained in/out space?
            'blocksize': numsteps, # same as loop length
            'numsteps':  numsteps,
            'loopblocksize': loopblocksize,
            # can't do this dynamically yet without changing init passes
            'outputs': {'x': {'shape': (3, )}, 'y': {'shape': (1, )}},
            # 'loop': [('inputs', {
            #     'lo': {'val': np.random.uniform(-i, 0, (3, 1)), 'shape': (3, )}, 'hi': {'val': np.random.uniform(0.1, i, (3, 1)), 'shape': (3, )}}) for i in range(1, 11)],
            # 'loop': lambda ref, i: ('inputs', {'lo': [10 * i], 'hi': [20*i]}),
            # 'loop': [('inputs', {'x': {'val': np.random.uniform(np.pi/2, 3*np.pi/2, (3,1))]}) for i in range(1, numsteps+1)],
            'loop': partial(f_loop_hpo, space = f_loop_hpo_space_f3(pdim = 3)),
            'loopmode': 'sequential',
            'loopblock': loopblock,
        },
    }),
    
    # plot module with blocksize = episode, fetching input from busses
    # and mapping that onto plots
    ("bplot", {
        'block': PlotBlock2,
        'params': {
            'id': 'bplot',
            'blocksize': numsteps,
            'idim': 6,
            'odim': 3,
            'debug': True,
            'inputs': {'d1': {'bus': 'b1/x'}, 'd2': {'bus': 'f1/y'}, 'd3': {'bus': 'hpo/y'},
                           'd4': {'bus': 'f2/y'}, 'd5': {'bus': 'hpo/x'}},
            'outputs': {}, # 'x': {'shape': (1, )}
            'subplots': [
                # [
                #     {'input': 'd1', 'slice': (0, 3),
                #          'plot': partial(timeseries, marker = 'o',
                #                              linestyle = 'None')},
                #     {'input': 'd1', 'slice': (0, 3), 'plot': histogram},
                # ],
                # [
                #     {'input': 'd2', 'slice': (3, 6), 'plot': timeseries},
                #     {'input': 'd2', 'slice': (3, 6), 'plot': histogram},
                # ],
                # [
                #     {'input': {'bus': 'd2'}, 'xaxis': 'd1', 'slice': (3, 6), 'plot': timeseries},
                #     {'input': 'd2', 'slice': (3, 6), 'plot': histogram},
                # ],
                # [
                #     {'input': {'bus': 'd4'}, 'xaxis': 'd1', 'slice': (3, 6), 'plot': timeseries},
                #     {'input': 'd4', 'slice': (3, 6), 'plot': histogram},
                # ],
                [
                    {'input': ['d3'], 'xaxis': 'd5', 'slice': (3, 6),
                         'plot': partial(timeseries, marker = 'o',
                                             linestyle = 'None')},
                    {'input': 'd5', 'slice': (3, 6), 'plot': histogram},
                ],
                [
                    {'input': 'd3', 'slice': (3, 6),
                         'plot': partial(timeseries, marker = 'o',
                                             linestyle = '-')},
                    {'input': 'd3', 'slice': (3, 6), 'plot': histogram},
                ],
            ]
        }
    })
])
