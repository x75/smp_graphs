"""smp_graphs funcs

2017 Oswald Berthold

'small' function blocks for use in loop blocks, configuration files,
and of course everywhere else. this only works for memoryless functions so far
"""

import numpy as np

from hyperopt import hp
from hyperopt import STATUS_OK, STATUS_FAIL
from hyperopt import fmin, tpe, Trials, rand, anneal

# callable test

class f_(object):
    def __call__(self, f):
        # args, kwargs
        # check if first and store stuff
        pass

# some minimal functions
f_sinesquare = lambda args: np.sin(args['x'][0])**2

def f_motivation(args):
    """distance / error motivation for cont activity modulation"""
    for k in ['x_', 'x']:
        assert args.has_key(k), "f_sin needs param '%s'" % (k,)

    x = args['x'][0]
    x_ = args['x_'][0]
    d = x_ - x
    return {'y': d, 'y1': -d}

def f_motivation_bin(args):
    """distance / error motivation for binary activity modulation"""
    for k in ['x_', 'x']:
        assert args.has_key(k), "f_sin needs param '%s'" % (k,)

    x = args['x'][0]
    x_ = args['x_'][0]
    d = x_ - x
    d_ = np.ones_like(d) * 0.03 # thresh
    # if np.linalg.norm(d) < 0.1:
    # mod = 0.0

    mod = (np.abs(d) > d_) * 0.4
    # print mod

    # d = np.ones_like(d)
        
    return {'y': mod, 'y1': -mod}

def f_pulse(args):
    for k in ['x']:
        assert args.has_key(k), "f_pulse needs param '%s'" % (k,)


def f_sin(args):
    """return the sin of input"""
    # FIXME: check that at configuration time?
    for k in ['x', 'f']:
        assert args.has_key(k), "f_sin needs param '%s'" % (k,)
        
    try:
        x = args['x'][0]
        f = args['f'][0]
        return np.sin(x * f)
    except KeyError, e:
        print "KeyError", e, args
        # print "Block %s doesn't have an input %s" % (args['id'], 'x')

def f_sin_noise(args):
    for k in ['x','f','sigma']:
        assert args.has_key(k), "f_sin_noise needs param '%s'" % (k,)
    x = f_sin(args)
    # print "x", x, args['sigma'][0]
    return np.random.normal(x, args['sigma'][0])
        
def f_sinesquare2(args):
    x = args['x'][0]
    scaler = np.arange(1, x.shape[0] + 1).reshape(x.shape) * 0.3 + 1.0
    # offser = np.array([[0.1, -0.05, 0.3]]).T
    # offser = np.random.uniform(-0.3, 0.3, x.shape)
    offser = np.zeros(x.shape)
    # print x.shape, scaler.shape
    # print "scaler", scaler
    return np.sin(x * scaler + offser)**2

def f_sinesquare3(args):
    x = f_sinesquare2(args)
    s1 = np.sum(x, axis = 0)
    return np.ones_like(x) * s1

def f_sinesquare4(args):
    x = f_sinesquare2(args)
    s1 = np.sum(x, axis = 0)
    return np.ones((1,1)) * s1

def f_loop_1(ref, i):
    return ('inputs', {'x': [np.random.uniform(-1, 1, (3, 1))]})

def f_obj_inverse(params, result):
    pass

def f_loop_hpo_space_f3(pdim = 1):
    return [hp.uniform("m%d" % i, np.pi/2, 3*np.pi/2) for i in range(pdim)]

def f_loop_hpo(ref, i, obj, space = None):
    # print "funcs.py: f_loop_hpo: i", i, ref
    # init first time round and store in parent
    if i < 1:
        trials = Trials()
        # paramdim = 1
        # space = [hp.uniform("m%d" % i, np.pi/2, 3*np.pi/2) for i in range(paramdim)]
        # space = ref.space
        bests = []
        lrstate = np.random.RandomState(1237)

        ref.hp_trials = trials
        ref.hp_space = space
        ref.hp_bests = bests
        ref.hp_lrstate = lrstate
    else:
        trials = ref.hp_trials
        space  = ref.hp_space
        bests  = ref.hp_bests
        lrstate = ref.hp_lrstate

    suggest = tpe.suggest

    best = fmin(obj, space, algo = suggest, max_evals=i+1, rstate=lrstate, trials=trials, verbose = 1)
    bests.append(best)
        
    # print "space", ref.hp_space
    # print "results", trials.results
    # print "losses", trials.losses()
    # print "trials", len(trials.trials), trials.trials[-1]
        
    # return ('inputs', {'x': [np.random.uniform(-1, 1, (3, 1))]})
    # return trials.trials[-1]['vals']
    return trials.results[-1]
