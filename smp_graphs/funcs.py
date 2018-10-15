"""smp_graphs funcs

2017 Oswald Berthold

'small' function blocks for use in loop blocks, configuration files,
and of course everywhere else. this only works for memoryless functions so far
"""

import numpy as np

from hyperopt import hp
from hyperopt import STATUS_OK, STATUS_FAIL
from hyperopt import fmin, tpe, Trials, rand, anneal

from scipy.signal import filtfilt, hilbert, butter

from smp_graphs.common import get_input

# # callable test
# class f_(object):
#     def __call__(self, f):
#         # args, kwargs
#         # check if first and store stuff
#         pass

def fu_check_required_args(args, reqs, funcname = "unknown"):
    for k in reqs:
        assert k in args, "function %s wants a param '%s'" % (funcname, k,)
    return True

################################################################################
# generator functions simple, fixed output name
f_sinesquare = lambda args: np.sin(args['x'][0])**2

# def f_pulse(args):
#     for k in ['x']:
#         assert args.has_key(k), "f_pulse needs param '%s'" % (k,)
#     pass

def f_sin(args):
    """return the sin of input"""
    # FIXME: check that at configuration time?
    fu_check_required_args(args, ['x', 'f'], 'f_sin')
        
    x = args['x']['val']
    # print "x", x.shape
    p = np.zeros_like(x)
    # print "p", p.shape
    for i in range(x.shape[0]):
        # p[0,0] = 0.0
        # p[1,0] = np.pi/2.0
        p[i,0] = float(i)/2.0
    f = args['f']['val']

    assert f.shape[0] == 1 or f.shape[0] == x.shape[0]
    
    w = (2 * np.pi * f)
    # print "x*w", (x * w).shape, p.shape
    # 2 * np.pi * f
    # print "w.shape", w.shape, (np.array([[0.0, 1.0/2.0]]).T * w).shape
    # return np.sin((x * w) + np.array([[0.0, 1.0/2.0]]).T * np.pi)
    # return np.sin((x * w) + p * np.pi) * 0.1 + 0.25
    return np.sin((x * w) + p * np.pi)
    # return np.sin(x * w)

def f_sin_noise(args):
    fu_check_required_args(args, ['x','f','sigma'], 'f_sin_noise')
    # print "args", args
    if 'amp' in args and 'offset' in args:
        x = (f_sin(args) * args['amp']['val']) + args['offset']['val']
    else:
        x = f_sin(args)

    # print "x", x.shape
    xn = np.random.normal(x, args['sigma']['val'], size=x.shape)
    # print "x", x, "xn", xn
    return xn

def f_sinesquare2(args):
    fu_check_required_args(args, ['x'], 'f_sinesquare2')
    
    x = args['x']['val']
    scaler = np.arange(1, x.shape[0] + 1).reshape(x.shape) * 0.3 + 1.0
    # offser = np.array([[0.1, -0.05, 0.3]]).T
    # offser = np.random.uniform(-0.3, 0.3, x.shape)
    offser = np.zeros(x.shape)
    # print x.shape, scaler.shape
    # print "scaler", scaler
    return np.sin(x * scaler + offser)**2

def f_sinesquare3(args):
    fu_check_required_args(args, ['x'], 'f_sinesquare2')
    
    x = f_sinesquare2(args)
    s1 = np.sum(x, axis = 0)
    return np.ones_like(x) * s1

def f_sinesquare4(args):
    fu_check_required_args(args, ['x'], 'f_sinesquare2')
    # print "func x", args['x']['val'].shape
    x = f_sinesquare2(args)
    s1 = np.sum(x, axis = 0)
    return {'y': np.ones((1,1)) * s1}

################################################################################
# generator functions

def f_sin_2(args):
    """funcs.f_sin_2

    return the sin of input on named output
    """
    # FIXME: check that at configuration time?
    fu_check_required_args(args, ['x', 'f'], 'f_sin')
        
    x = args['x']['val']
    f = args['f']['val']

    
    return np.sin(x * f)

# def f_sin_noise(args):
#     fu_check_required_args(args, ['x','f','sigma'], 'f_sin_noise')
    
#     x = f_sin(args)
#     xn = np.random.normal(x, args['sigma']['val'], size=x.shape)
#     print "x", x, "xn", xn
#     return xn

################################################################################
# grids

def f_meshgrid(args):
    """f_meshgrid

    create meshgrid
    """
    fu_check_required_args(args, ['ranges', 'steps'], 'f_meshgrid')
    # FIXME: check_outputs: only one item
    
    # create meshgrid over proprio dimensions
    steps = get_input(args, 'steps')
    ranges = get_input(args, 'ranges')
    ndims = len(ranges)
    # dim_axes = [np.linspace(self.environment.conf.m_mins[i], self.environment.conf.m_maxs[i], sweepsteps) for i in range(self.environment.conf.m_ndims)]
    # dim_axes = [np.linspace(ranges[i][0], ranges[i][1], steps) for i in range(ndims)]
    dim_axes = [np.linspace(min_, max_, steps) for min_, max_ in ranges]

    full_axes = np.meshgrid(*tuple(dim_axes), indexing='ij')

    # print "dim_axes", dim_axes
    # print "full_axes", len(full_axes)
    # print "full_axes", full_axes
    
    # for i in range(len(full_axes)):
    #     print i, full_axes[i].shape
    #     print i, full_axes[i].flatten()
            
    # return proxy
    full_axes_flat = np.vstack([full_axes[i].flatten() for i in range(len(full_axes))])
    # print "func", full_axes_flat.shape
    # print "func", full_axes_flat
    return {'meshgrid': full_axes_flat} # .T

def f_meshgrid_mdl(args):
    fu_check_required_args(args, ['ranges', 'steps'], 'f_meshgrid_mdl')

    # create meshgrid over proprio dimensions
    steps = get_input(args, 'steps')
    ranges = get_input(args, 'ranges')
    ndims = len(ranges)
    
    # # extero config
    # dim_axes = [np.linspace(self.environment.conf.s_mins[i], self.environment.conf.s_maxs[i], steps) for i in range(self.environment.conf.s_ndims)]
    # # dim_axes = [np.linspace(self.environment.conf.s_mins[i], self.environment.conf.s_maxs[i], steps) for i in range(self.mdl.idim)]
    # print "rh_model_sweep_generate_input_grid: s_ndims = %d, dim_axes = %s" % (self.environment.conf.s_ndims, dim_axes,)
    # full_axes = np.meshgrid(*tuple(dim_axes), indexing='ij')
    # print "rh_model_sweep_generate_input_grid: full_axes = %s, %s" % (len(full_axes), full_axes,)

    dim_axes = [np.linspace(min_, max_, steps) for min_, max_ in ranges]

    full_axes = np.meshgrid(*tuple(dim_axes), indexing='ij')
    
    for i in range(len(full_axes)):
        print(i, full_axes[i].shape)
        print(i, full_axes[i].flatten())

    # return proxy
    error_grid = np.vstack([full_axes[i].flatten() for i in range(len(full_axes))])
    print("error_grid", error_grid.shape)

    # draw state / goal configurations
    X_accum = []
    states = np.linspace(-1, 1, steps)
    # for state in range(1): # steps):
    for state in states:
        # randomize initial position
        # self.M_prop_pred = self.environment.compute_motor_command(np.random.uniform(-1.0, 1.0, (1, self.odim)))
        # draw random goal and keep it fixed
        # self.goal_prop = self.environment.compute_motor_command(np.random.uniform(-1.0, 1.0, (1, self.odim)))
        goal_prop = np.ones((1, ndims)) * state
        # self.goal_prop = np.random.uniform(self.environment.conf.m_mins, self.environment.conf.m_maxs, (1, self.odim))
        GOALS = np.repeat(goal_prop, error_grid.shape[1], axis = 0) # as many goals as error components
        # FIXME: hacks for M1/M2
        if ndims == 3:
            X = GOALS
        elif ndims == 6:
            X = np.hstack((GOALS, error_grid.T))
        else:
            X = np.hstack((GOALS, error_grid.T))
        X_accum.append(X)

    X_accum = np.array(X_accum)

    # don't need this?
    # X_accum = X_accum.reshape((X_accum.shape[0] * X_accum.shape[1], X_accum.shape[2]))
        
    # print "X_accum.shape = %s, mdl.idim = %d, mdl.odim = %d" % (X_accum.shape, self.mdl.idim, self.mdl.odim)
    # print X_accum
    X = X_accum
    # X's and pred's indices now mean: slowest: goal, e1, e2, fastest: e3
    # self.X_model_sweep = X.copy()
    # print "self.X_model_sweep.shape", self.X_model_sweep.shape
    # return proxy
    full_axes_flat = np.vstack([full_axes[i].flatten() for i in range(len(full_axes))])
    print("func", full_axes_flat)
    return {'meshgrid': full_axes_flat} # .T

def f_random_uniform(args):
    fu_check_required_args(args, ['ranges', 'steps'], 'f_random_uniform')

    ranges = get_input(args, 'ranges')
    steps = get_input(args, 'steps')

    meshgrid = np.random.uniform(ranges[:,[0]], ranges[:,[1]], (ranges.shape[0], int(steps[0,0]**ranges.shape[0])))

    # print "f_random_uniform meshgrid = %s" % (meshgrid.shape, )
    
    return {'meshgrid': meshgrid}

def f_meansquare(args):
    fu_check_required_args(args, ['x'], 'f_meansquare')
    # lambda x: {'y': np.sum(np.square(x['x']['val']))}
    x = get_input(args, 'x')
    y = np.atleast_2d(np.mean(np.square(x)))
    # print "    f_meansquare x = %s, y = %s" % (x, y)
    return {'y': y}

def f_rootmeansquare(args):
    fu_check_required_args(args, ['x'], 'f_meansquare')
    # lambda x: {'y': np.sum(np.square(x['x']['val']))}
    # x = get_input(args, 'x')
    r = f_meansquare(args)
    r['y'] = np.sqrt(r['y'])
    # print "    f_meansquare x = %s, y = %s" % (x, y)
    return r

def f_sum(args):
    fu_check_required_args(args, ['x'], 'f_sum')
    # lambda x: {'y': np.sum(np.square(x['x']['val']))}
    x = get_input(args, 'x')
    y = np.atleast_2d(np.sum(x))
    # print "    f_meansquare x = %s, y = %s" % (x, y)
    return {'y': y}

################################################################################
# model functions

# kinesis motivation
def f_motivation(args):
    """distance / error motivation for cont activity modulation"""
    fu_check_required_args(args, ['x', 'x_'], 'f_motivation')

    x = args['x']['val']
    x_ = args['x_']['val']
    # element-wise if goal dims and proprio dims are the same
    # if args['x']['shape'][0] == args['x__']['shape'][0]:
    d = x_ - x
    if 'x__' in args:
        # else distance norm
        d = np.ones_like(args['x__']['val']) * np.linalg.norm(d, 2)
    return {'y': d, 'y1': -d}

# kinesis binary motivation
def f_motivation_bin(args):
    """distance / error motivation for binary activity modulation"""
    fu_check_required_args(args, ['x', 'x_'], 'f_motivation_bin')

    x = args['x']['val']
    x_ = args['x_']['val']
    d = x_ - x
    d_ = np.ones_like(d) * 0.1 # thresh
    # if np.linalg.norm(d) < 0.1:
    # mod = 0.0

    mod = (np.abs(d) > d_) * 0.4
    # print "f_motivation_bin mod = ", mod

    # d = np.ones_like(d)
        
    return {'y': mod, 'y1': -mod}

def f_envelope(args):
    """amplitude envelope follower

    the internet seems to agree that hilbert is the way to do it
    """
    # option 1: np.abs(hilbert(x))
    fu_check_required_args(args, ['x'], 'f_envelope')
    if 'c' not in args: args['c'] = {'val': 0.5}
    c = args['c']['val']
    # print "funcs.py f_envelope c = %s" % (c, )
    b, a  = butter(4, c)
    x = args['x']['val']
    return {'y': filtfilt(b, a, np.abs(hilbert(x)))}

################################################################################
# exec functions: FIXME separate file, they are not FuncBlock2 funcs but LoopBlock callbacks
def f_loop_1(ref, i):
    return ('inputs', {'x': [np.random.uniform(-1, 1, (3, 1))]})

def f_obj_inverse(params, result):
    pass

def f_loop_hpo_space_f3(pdim = 1):
    """Return pdim-dimensional uniform hyperopt search space
    """
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

    # print "obj", obj
    # print "space", space # ref.hp_space
    # print "algo", suggest
    # print "max_evals", i+1
    # print "lrstate", lrstate
    # print "trials", len(trials.trials) # , trials.trials[-1]
    # print "results", trials.results
    # print "losses", trials.losses()
    
    best = fmin(obj, space, algo = suggest, max_evals=i+1, rstate=lrstate, trials=trials, verbose = 1)
    # best = fmin(obj, space, algo = suggest, max_evals = 10)
    bests.append(best)
        
        
    # return ('inputs', {'x': [np.random.uniform(-1, 1, (3, 1))]})
    # return trials.trials[-1]['vals']
    return trials.results[-1]
