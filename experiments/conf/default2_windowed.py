"""smp_graphs conf windowed analysis"""

from smp_graphs.block_meas import WindowedBlock2
from smp_graphs.block_plot import ImgPlotBlock2

numsteps = 256*40

graph = OrderedDict([
    # file source
    ('wav', {
        'block': FileBlock2,
        'params': {
            'blocksize': 1,
            'type': 'wav',
            # 'file': ['data/res_out.wav'],
            'file': {'filename': 'data/res_out.wav', 'filetype': 'wav', 'offset': 100000, 'length': numsteps},
            'file': {'filename': '../../smp/sequence/data/blackbird_XC330200/XC330200-1416_299-01hipan.wav', 'filetype': 'wav', 'offset': 0, 'length': numsteps},
            
            'outputs': {'x': {'shape': (2,)}}
            },
        }),
    # windowed analysis
    ('xgram', {
        'block': WindowedBlock2,
        'params': {
            'blocksize': 256,
            'rate': 128,
            'inputs': {'x': {'bus': 'wav/x'}},
            'outputs': {'x': {'shape': (2,)}},
            },
        }),
        
    # plot file
    ('plot', {
        # 'block': ImgPlotBlock2,
        'block': PlotBlock2,
        'params': {
            'blocksize': numsteps,
            'inputs': {'d1': {'bus': 'wav/x'},
                       'd2': {'bus': 'xgram/x'}},
            'subplots': [
                [
                    {'input': 'd1', 'plot': timeseries},
                ],
                # [
                #     {'input': 'd2', 'plot': timeseries},
                # ]
                ]
            },
        }),
    ('plotimg', {
        'block': ImgPlotBlock2,
        'params': {
            'blocksize': numsteps,
            'inputs': {'d1': {'bus': 'wav/x'},
                       'd2': {'bus': 'xgram/x'}},
            'subplots': [
                # [
                #     {'input': 'd1', 'plot': timeseries},
                # ],
                [
                    {'input': 'd2', 'shape': (256, 2), 'cmap': 'Blues', 'ylog': True,
                     'xslice': (0, 256), 'yslice': (0, 2)}, # 'RdBu'
                ]
                ]
            },
        }),
    ])
