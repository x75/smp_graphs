"""smp_graphs configuration

self-exploration (se)

plot different target functions
"""

numsteps = 1000

# graph
graph = OrderedDict([
    ('cnt', {
        'block': CountBlock2,
        # 'params': {
        #     'blocksize': 1,
        #     'debug': False,
        #     'inputs': {},
        #     'outputs': {'x': {'shape': (dim_s_proprio, 1)}},
        #     },
        }),
    # ('a', {
    #     'block': FuncBlock2,
    #     # 'params': { 'func': lambda x: x, },
    #     }),
    # ('plot', {
    #     'block': PlotBlock2,
    #     'params': {
    #         'inputs': {
    #             'x': {'bus': 'cnt/cnt'},
    #             }
    #         },
    #     }),
])



