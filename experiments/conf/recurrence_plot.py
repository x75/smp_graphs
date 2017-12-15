"""smp_graphs puppy recurrence plot conf
"""

import random

from smp_graphs.block_meas_essentia import eFileBlock2, EssentiaBlock2, AdhocMixBlock2

from smp_base.plot import rp_timeseries_embedding

from smp_graphs.block import StackBlock2
from smp_graphs.block_meas import MomentBlock2

# reuse
numsteps = 44100 * 120 # 65535 # 8192
# xdim = 6
# ydim = 4
xdim = 1
ydim = 1
mfcc_numcoef = 13
winsize = 1024
# offset_range = 

lconf = {
    'file': {
        'filename': 'data/goodPickles/recording_eC0.14_eA0.14_c0.50_n1000_id0.pickle',
        'filetype': 'puppy',
        'offset': 300,
        'length': numsteps,
    },
}


# puppy pickle data homeokinesis episodes
filearray = [
    'data/goodPickles/recording_eC0.14_eA0.14_c0.50_n1000_id0.pickle',
    'data/goodPickles/recording_eC1.12_eA1.66_c0.50_n1000_id0.pickle',
    'data/goodPickles/recording_eC0.24_eA0.77_c0.50_n1000_id0.pickle',
    ]

# puppy pickle data step response sweep
filearray = [
	'data/stepPickles/step_period_34_0.pickle',
	# 'data/stepPickles/step_period_16_0.pickle',
	# 'data/stepPickles/step_period_2_0.pickle',
	# 'data/stepPickles/step_period_64_0.pickle',
	# 'data/stepPickles/step_period_14_0.pickle',
	# 'data/stepPickles/step_period_20_0.pickle',
]

# audio files
filearray = [
    ('/home/lib/audio/work/fm_mix_subcity_sorbierd_20171125//Powell--New-Beta-Vol.1--Boomkat(4).mp3', 'mp3', 5292000, 44100),
    ('/home/lib/audio/work/fm_mix_subcity_sorbierd_20171125//Autechre - spl47.mp3', 'mp3', 14553000, 44100),
    ('/home/lib/audio/work/fm_mix_subcity_sorbierd_20171125//03_db_247927330_soundcloud.mp3', 'mp3', 17992800, 44100),
    ('/home/lib/audio/work/fm_mix_subcity_sorbierd_20171125//10. Joy Division - I Remember Nothing (1979).mp3', 'mp3', 15567300, 44100),
    ('/home/lib/audio/work/fm_mix_subcity_sorbierd_20171125//05. Original Degung Instrumentalia - Sudanese Gamelan Music - [Panineungan].mp3', 'mp3', 17551800, 44100),
    ('/home/lib/audio/work/fm_mix_subcity_sorbierd_20171125//1going_320.mp3', 'mp3', 6174000, 44100),
    ('/home/lib/audio/work/fm_mix_subcity_sorbierd_20171125//Burial - Beachfires.mp3', 'mp3', 26063100, 44100),
    ('/home/lib/audio/work/fm_mix_subcity_sorbierd_20171125//02-magic.ii.mp3', 'mp3', 23152500, 44100),
    ('/home/lib/audio/work/fm_mix_subcity_sorbierd_20171125//02 Man Machine (Kraftwerk).mp3', 'mp3', 4586400, 44100),
    ('/home/lib/audio/work/fm_mix_subcity_sorbierd_20171125//The Beach Boys - Heroes And Villains Sections  (Bonus Track. Stere.mp3', 'mp3', 19227600, 44100),
    ('/home/lib/audio/work/fm_mix_subcity_sorbierd_20171125//11-prefuse_73-we_got_our_own_way_feat._kazu-ftd.mp3', 'mp3', 8775900, 44100),
    ('/home/lib/audio/work/fm_mix_subcity_sorbierd_20171125//Stokes Croft-001-Kamikaze Space Programme-Choke (Original Mix).mp3', 'mp3', 15743700, 44100),
    ('/home/lib/audio/work/fm_mix_subcity_sorbierd_20171125//05 FriscoBum.mp3', 'mp3', 9922500, 44100),
    ('/home/lib/audio/work/fm_mix_subcity_sorbierd_20171125//07-Bricc_Baby_Shitro-IDK_Feat_Casey_Veggies_Prod_By_Metro_Boomin.mp3', 'mp3', 13009500, 44100),
    ('/home/lib/audio/work/fm_mix_subcity_sorbierd_20171125//a deepness upon the sky ~by micrOmega [soundtake.net].mp3', 'mp3', 18963000, 44100),
    ('/home/lib/audio/work/fm_mix_subcity_sorbierd_20171125//militik ~by micrOmega [soundtake.net].mp3', 'mp3', 9702000, 44100),
    ('/home/lib/audio/work/fm_mix_subcity_sorbierd_20171125//The Beach Boys - Heroes And Villains.mp3', 'mp3', 12877200, 44100),
]

looparray = [
    ('file', {
        'filename': ftup[0],
        'filetype': ftup[1],
        # 'offset': random.randint(0, ftup[2] - (ftup[3] * 60)),
        'offset': 0,
        # 'length': min(ftup[2] - (ftup[3] * 60), random.randint(ftup[3] * 60, ftup[3] * 60 * 3)),
        # 'length': None,
        'length': ftup[3] * 60,
        'samplerate': ftup[3],
    }) for ftup in filearray]

looparray += [
    ('file', {
        'filename': ftup[0],
        'filetype': ftup[1],
        # 'offset': random.randint(0, ftup[2] - (ftup[3] * 60)),
        'offset': ftup[2] - ftup[3] * 60,
        # 'length': min(ftup[2] - (ftup[3] * 60), random.randint(ftup[3] * 60, ftup[3] * 60 * 3)),
        # 'length': None,
        'length': ftup[3] * 60,
        'samplerate': ftup[3],
    }) for ftup in filearray]

# # loopblock 1: puppy
# loopblock = {
#     'block': FileBlock2,
#     'params': {
#         'id': 'puppydata',
#         'debug': False,
#         'blocksize': numsteps,
#         'type': 'puppy',
#         'file': {
#             'filename': 'data/goodPickles/recording_eC0.14_eA0.14_c0.50_n1000_id0.pickle',
#             'filetype': 'puppy',
#             'offset': 300,
#             'length': numsteps,
#         },
#         'outputs': {
#             'x': {'shape': (xdim, numsteps)},
#             'y': {'shape': (ydim, numsteps)}
#         },
#         },
#     }

# loopblock 2: audio with essentia
loopblock_graph = OrderedDict([
    # file block
    ('filedata', {
        'block': eFileBlock2,
        'params': {
            'id': 'filedata',
            'debug': False,
            'blocksize': numsteps,
            'logging': True,
            'type': 'puppy',
            # 'lconf': {},
            'file': {},
            'outputs': {
                'x': {'shape': (xdim, numsteps)},
                # 'y': {'shape': (ydim, numsteps)}
            },
        },
    }),
    # analysis block
    ('essentia1', {
        'block': EssentiaBlock2,
        'params': {
            'id': 'essentia1',
            'debug': False,
            'blocksize': numsteps,
            'samplerate': winsize,
            'inputs': {
                'x': {'bus': 'filedata/x'},
            },
            'outputs': {
                'centroid': {'shape': (xdim * 1, numsteps / winsize + 1), 'etype': 'centroid'},
                'mfcc': {'shape': (xdim * mfcc_numcoef, numsteps / winsize + 1), 'etype': 'mfcc', 'numberCoefficients': mfcc_numcoef, 'numberBands': 40},
                # 'sbic': {'shape': (xdim * 1, numsteps / winsize + 1), 'etype': 'sbic'},
                # 'y': {'shape': (xdim, numsteps)},
                # 'y': {'shape': (ydim, numsteps)}
            },
        },
    }),
    # stack
    ('features', {
        'block': StackBlock2,
        'params': {
            'id': 'features',
            'blocksize': numsteps,
            'inputs': {
                'centroid': {'bus': 'essentia1/centroid'},
                'mfcc': {'bus': 'essentia1/mfcc'},
            },
            'outputs': {
                'y': {'shape': (mfcc_numcoef + 1, numsteps / winsize + 1)}
            },
        }
    }),
    # moment block
    ('moments', {
        'block': MomentBlock2,
        'params': {
            'id': 'moments',
            'debug': False,
            'blocksize': numsteps,
            'transpose': True,
            'inputs': {
                'features': {'bus': 'features/y'},
            },
            'outputs': {
                'features_mu':  {'shape': (1, mfcc_numcoef + 1)},
                'features_var': {'shape': (1, mfcc_numcoef + 1)},
                'features_min': {'shape': (1, mfcc_numcoef + 1)},
                'features_max': {'shape': (1, mfcc_numcoef + 1)},
                # 'features_mu': {'shape': (mfcc_numcoef + 1, 1)},
                # 'features_var': {'shape': (mfcc_numcoef + 1,1)},
                # 'features_min': {'shape': (mfcc_numcoef + 1,1)},
                # 'features_max': {'shape': (mfcc_numcoef + 1,1)},
                # 'centroid': {'shape': (xdim * 1, numsteps / winsize + 1), 'etype': 'centroid'},
                # 'mfcc': {'shape': (xdim * mfcc_numcoef, numsteps / winsize + 1), 'etype': 'mfcc', 'numberCoefficients': mfcc_numcoef, 'numberBands': 40},
                # 'sbic': {'shape': (xdim * 1, numsteps / winsize + 1), 'etype': 'sbic'},
                # 'y': {'shape': (xdim, numsteps)},
                # 'y': {'shape': (ydim, numsteps)}
            },
        },
    }),
    
    # action block
    # output / render block
])

loopblock = {
    'block': Block2,
    'params': {
        'id': 'loopblock_graph',
        'debug': False,
        'topblock': False,
        'logging': True,
        'numsteps': numsteps,
        'blocksize': 1,
        'blockphase': [0],
        # 'lconf': {},
        # 'outputs': {'jh': {'shape': (1,1)}},
        # 'outputs': {'jh': {'shape': (1, 1), 'buscopy': 'jh/jh'}},
        'subgraph_rewrite_id': True,
        # contains the subgraph specified in this config file
        'subgraph': loopblock_graph,
        'outputs': { # need outputs here for caching
            # 'features_mu':  {'shape': (1, mfcc_numcoef + 1)},
            # 'features_var': {'shape': (1, mfcc_numcoef + 1)},
            # 'features_min': {'shape': (1, mfcc_numcoef + 1)},
            # 'features_max': {'shape': (1, mfcc_numcoef + 1)},
            # 'x': {'buscopy': 'filedata/x'},
            # 'y': {'buscopy': 'filedata/y'},
        },
    },
}

def make_puppy_rp_plot_inputs():
    global looparray, numsteps, xdim, ydim, winsize, mfcc_numcoef
    from collections import OrderedDict
    # buskey_base = ['puppyloop']
    # buskey_bases = ['filedata', 'essentia1']
    # buskey_signal = ['x', 'y']
    # buskey_bases = ['essentia1']
    # buskey_signal = [ 'y']
    
    # buskey_bases = ['essentia1'] * 2
    # buskey_signal = [ 'centroid', 'mfcc']
    # buskey_shapes = [ (xdim, numsteps / winsize + 1), (mfcc_numcoef, numsteps / winsize + 1)]
    
    # buskey_bases = ['features']
    # buskey_signal = [ 'y']
    # buskey_shapes = [ (mfcc_numcoef + 1, numsteps / winsize + 1)]

    buskey_bases = ['moments'] * 1
    buskey_signal = [ 'features_mu'] # , 'features_var', 'features_min', 'features_max']
    buskey_shapes = [ (1, mfcc_numcoef + 1)] * 1
    
    # buskey_bases = ['puppyloop'] * 1
    # buskey_signal = [ 'features_mu'] # , 'features_var', 'features_min', 'features_max']
    # buskey_shapes = [ (1, mfcc_numcoef + 1)] * 1
    
    # inspec = {'d3': {'bus': 'b2/x', 'shape': (3, numsteps)}}
    # inspec = {}
    inspec = OrderedDict()
    for i,l in enumerate(looparray):
        # print "l", l
        if 'length' not in l[1]:
            l[1]['length'] = 0
        for j, buskey_base in enumerate(buskey_bases):
            inspec['%s_ll%d_%d' % (buskey_signal[j], i, j, )] = {
                'bus': '%s_ll%d_ll0_ll0/%s' % (buskey_base, i, buskey_signal[j]),
                # 'name': '%s %s[%d:%d]' % (buskey_signal[j], l[1]['filename'].split('/')[-1], l[1]['offset'], l[1]['offset'] + l[1]['length']),
                'name': '%s %s' % (buskey_signal[j], l[1]['filename'].split('/')[-1]),
                'shape': buskey_shapes[j], # (xdim, numsteps / winsize + 1),
            }
            # inspec['%s_ll%d_%d' % (buskey_signal[j], i, j, )] = {'bus': '%s_ll%d_ll0_ll0/%s' % (buskey_base, i, buskey_signal[j])}
            # inspec['y_ll%d_%d' % (i, j, )] = {'bus': '%s_ll%d_ll0_ll0/y' % (buskey_base, i, )}
            # inspec['x_e_ll%d' % (i, )] = {'bus': '%s_ll%d_ll0_ll0/x' % (buskey_base, i, )}
    return inspec

# graph
graph = OrderedDict([
    # puppy data
    ('puppyloop', {
        'block': LoopBlock2,
        'params': {
            'id': 'puppydata',
            'blocksize': numsteps,
            'loop': looparray,
            'loopblock': loopblock,
            # 'loopblock': loopblock_graph,
            'numsteps': numsteps,
            'outputs': {
                'features_mu': {'buscopy': 'moments/features_mu', 'shape': (xdim, len(looparray))},
                # 'y': {'shape': (ydim, numsteps)},
            },
        },
    }),
             
    # stack
    ('mus', {
        'block': StackBlock2,
        'params': {
            'id': 'mus',
            'blocksize': numsteps,
            'inputs': dict([('mu%d' % (i, ), {'bus': 'moments_ll%d_ll0_ll0/features_mu' % (i, )}) for i in range(len(looparray))]),
            #     'mu1': {'bus': 'moments_ll1_ll0_ll0/features_mu'},
            # },
            'outputs': {
                'y': {'shape': (len(looparray), mfcc_numcoef + 1)}
            },
        }
    }),
    
    # Ad-hoc mixing block
    ('mix', {
        'block': AdhocMixBlock2,
        'params': {
            'id': 'mix',
            'blocksize': numsteps,
            'filearray': filearray,
            'inputs': {'mus': {'bus': 'mus/y'}},
            #     'mu1': {'bus': 'moments_ll1_ll0_ll0/features_mu'},
            # },
            'outputs': {
                'y': {'shape': (len(looparray), mfcc_numcoef + 1)}
            },
        }
    }),
    
    # # a constant
    # ("b1", {
    #     'block': ConstBlock2,
    #     'params': {
    #         'id': 'b1',
    #         'inputs': {'c': {'val': np.random.uniform(-1, 1, (3, 1))}},
    #         'outputs': {'x': {'shape': (3, 1)}},
    #         'debug': False,
    #     },
    # }),
    
    # # a random number generator, mapping const input to hi
    # ("b2", {
    #     'block': UniformRandomBlock2,
    #     'params': {
    #         'id': 'b2',
    #         # 'lo': 0,
    #         # 'hi': 1,
    #         'inputs': {'lo': {'val': 0}, 'hi': {'bus': 'b1/x'}},
    #         'outputs': {'x': {'shape': (3, 1)}},
    #         'debug': False,
    #     },
    # }),
    
    # plot module with blocksize = episode, fetching input from busses
    # and mapping that onto plots
    ("bplot", {
        'block': PlotBlock2,
        'params': {
            'id': 'bplot',
            'blocksize': numsteps,
            'debug': False,
            'wspace': 0.4, 'hspace': 0.4,
            # 'inputs': make_puppy_rp_plot_inputs(),
            'inputs': {
                'mus': {'bus': 'mus/y'},
            },
            'ylim_share': False,
            'subplots': [
                # [
                # # {'input': '%s_ll%d' % (itup[1], itup[0]), 'plot': timeseries},
                # # {'input': '%s_ll%d' % (itup[1], itup[0]), 'plot': histogram},
                # # {'input': '%s_ll%d' % (itup[1], itup[0]), 'plot': rp_timeseries_embedding}
                
                # {
                #     'input': '%s' % (itup[0], ),
                #     'plot': partial(timeseries, marker = 'o'),
                #     'title': itup[1]['name'],
                #     'shape': itup[1]['shape'],
                # },
                # # , 'y'
                # # ] for itup in zip(map(lambda x: x/2, range(len(looparray)*2)), ['x'] * len(looparray))]
                # ] for itup in make_puppy_rp_plot_inputs().items()]

                [{
                    'input': ['mus'],
                    'plot': partial(timeseries, marker = 'o'),
                    'title': 'stacked moments',
                    'shape': (len(looparray), mfcc_numcoef + 1),}
                    ]
                ]
                
            # [
            #     [
            #     {'input': 'x_%d' % (i, ), 'slice': (0, 6), 'plot': timeseries},
            #     {'input': 'x_%d' % (i, ), 'slice': (0, 6), 'plot': histogram},
            #     {'input': 'x_%d' % (i, ), 'slice': (0, 6), 'plot': rp_timeseries_embedding}]
            #     for ink in zip(range(3)]
                # [[i for i in range(3)]]
                # + [
                #     [
                #     {'input': 'd3', 'slice': (0, 3), 'plot': timeseries},
                #     {'input': 'd3', 'slice': (0, 3), 'plot': histogram},
                #     {'input': 'd3', 'slice': (0, 3), 'plot': rp_timeseries_embedding},
                # ]]
                # [
                #     {'input': 'd1', 'slice': (0, 6), 'plot': timeseries},
                #     {'input': 'd1', 'slice': (0, 6), 'plot': histogram},
                #     {'input': 'd1', 'slice': (0, 6), 'plot': rp_timeseries_embedding},
                # ],
                # [
                #     {'input': 'd2', 'slice': (0, 4), 'plot': timeseries},
                #     {'input': 'd2', 'slice': (0, 4), 'plot': histogram},
                #     {'input': 'd2', 'slice': (0, 4), 'plot': rp_timeseries_embedding},
                # ],
                # ,
            #]
        }
    })
])
