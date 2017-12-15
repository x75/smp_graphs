"""smp_graphs example: windowed analysis of a timeseries (STFT)"""

from smp_graphs.block_meas import WindowedBlock2
from smp_graphs.block_plot import ImgPlotBlock2

bs_win_2 = 256*4
numsteps = bs_win_2*10
bs_win = 128*4
num_win = (numsteps/bs_win)-1

print("steps and wins", bs_win_2, numsteps, bs_win, num_win)
    
graph = OrderedDict([
    # file source
    ('wav', {
        'block': FileBlock2,
        'params': {
            'blocksize': 1,
            'type': 'wav',
            # 'file': ['data/res_out.wav'],
            # 'file': {'filename': 'data/res_out.wav', 'filetype': 'wav', 'offset': 100000, 'length': numsteps},
            'file': {'filename': '../../smp/sequence/data/blackbird_XC330200/XC330200-1416_299-01hipan.wav', 'filetype': 'wav', 'offset': 0, 'length': numsteps},
            
            'outputs': {'x': {'shape': (2, 1)}}
            },
        }),
        
    # windowed analysis
    ('xgram', {
        'block': WindowedBlock2,
        'params': {
            'blocksize': bs_win,
            # 'rate': 128,
            'inputs': {'x': {'bus': 'wav/x', 'shape': (2, bs_win_2)}},
            'outputs': {'x': {'shape': (2, bs_win_2)}},
            },
        }),
        
    # plot as timeseries
    ('plot', {
        # 'block': ImgPlotBlock2,
        'block': PlotBlock2,
        'params': {
            'blocksize': numsteps,
            'debug': False,
            'xlim_share': False,
            'ylim_share': False,
            'inputs': {'d1': {'bus': 'wav/x', 'shape': (2, numsteps)},
                       'd2': {'bus': 'xgram/x', 'shape': (2, num_win * bs_win_2)}},
            'subplots': [
                [
                    {'input': 'd1', 'plot': timeseries},
                    # {'input': 'd1', 'plot': timeseries},
                ],
                [
                    # {'input': 'd2', 'plot': timeseries, 'yslice': (0, 1), 'shape': (((numsteps/128)-1)*2, bs_win_2)},
                    {'input': ['d2' for i in range(num_win * 2)], 'plot': timeseries,
                     'ndslice': [(slice((i%num_win)*bs_win_2, ((i%num_win)+1)* bs_win_2), i/num_win) for i in range(num_win * 2)],
                    'shape': [(1, bs_win_2) for i in range(num_win * 2)]},
                    # {'input': 'd2', 'plot': timeseries, 'xslice': (bs_win_2, 512)},
                ]
                ]
            },
        }),

    # plot as image
    ('plotimg', {
        'block': ImgPlotBlock2,
        'params': {
            'blocksize': numsteps,
            'inputs': {'d1': {'bus': 'wav/x', 'shape': (2, numsteps)},
                       'd2': {'bus': 'xgram/x', 'shape': (2, num_win * bs_win_2)}},
            'wspace': 0.5, 'hspace': 0.5,
            'subplots': [
                # [
                #     {'input': 'd1', 'plot': timeseries},
                # ],
                [
                    {'input': 'd2', 'shape': (num_win, bs_win_2), 'cmap': 'Blues', 'ylog': True,
                     'xslice': (0, num_win * bs_win_2), 'yslice': (0, 1)}, # 'RdBu'
                    {'input': 'd2', 'shape': (num_win, bs_win_2), 'cmap': 'Blues', 'ylog': True,
                     'xslice': (0, num_win * bs_win_2), 'yslice': (1, 2)}, # 'RdBu'
                ]
                ]
            },
        }),
    ])
