# global sweepsys_input_flat, sweepsys_steps, dim_s_proprio
from smp_graphs.funcs import f_meshgrid

graph = OrderedDict([
            ('sweepsys_grid', {
                'block': FuncBlock2,
                'params': {
                    'debug': False,
                    'numsteps': sweepsys_input_flat,
                    'blocksize': sweepsys_input_flat,
                    'inputs': {
                        'ranges': {'val': np.array([[-1, 1]] * dim_s_proprio)},
                        'steps':  {'val': sweepsys_steps},
                        },
                    'outputs': {'meshgrid': {'shape': (dim_s_proprio, sweepsys_input_flat)}},
                    'func': f_meshgrid
                    # 'func': f_random_uniform,
                },
            }),
            
            # sys to sweep
            sweepsys,
])
