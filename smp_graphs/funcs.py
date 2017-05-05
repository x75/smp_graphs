import numpy as np

from hyperopt import hp
from hyperopt import STATUS_OK, STATUS_FAIL
from hyperopt import fmin, tpe, Trials, rand, anneal


# some minimal functions

f_sinesquare = lambda args: np.sin(args['x'][0])**2

def f_sinesquare2(args):
    x = args['x'][0]
    scaler = np.arange(1, x.shape[0] + 1).reshape(x.shape) * 0.3
    offser = np.array([[0.1, -0.05, 0.3]]).T
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

def f_loop_hpo(ref, i, obj):
    print "funcs.py: loop_hpo: i", i
    if i < 1:
        trials = Trials()
        space = [hp.uniform("m%d" % i, np.pi/2, 3*np.pi/2) for i in range(1)]
        bests = []
        lrstate = np.random.RandomState(1234)

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
        
    print "space", ref.hp_space
    print "results", trials.results
    print "losses", trials.losses()
    print "trials", len(trials.trials), trials.trials[-1]
        
    # return ('inputs', {'x': [np.random.uniform(-1, 1, (3, 1))]})
    # return trials.trials[-1]['vals']
    return trials.results[-1]
