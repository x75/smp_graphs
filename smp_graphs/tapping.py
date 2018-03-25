"""tapping.py

Tapping is a graphical model mapping sensorimotor data indices (time
t, modality s) to a single machine learning input data point at
time t, see [1]

Full tapping
 - tap: get input at times t_tap retaining structure
 - tap_?: do any computations in structured space
 - tap_flat: transpose and flatten structured tap
 - ref.mdl.step: apply to model fit / predict cycle
 - tap_struct: restructure flat prediction

TODO:
 - moving over legacy tapping funcs from :file:`funcs_models.py`
 - move developmental models into smp_base to be able to reuse them
   outside of smp_graphs.

[1] https://arxiv.org/abs/1704.07622
"""

import numpy as np

def tap_tupoff(tup = (), off = 0):
    """block_models.tap_tupoff

    Return the input tuple with constant offset added to both elements
    """
    assert len(tup) == 2, "block_models.py.tap_tupoff wants 2-tuple, got %d-tuple" % (len(tup), )
    return (tup[0] + off, tup[1] + off)

def tap(ref, inkey = None, lag = None):
    """block_models.tap

    Tap into inputs at indices given by lag

    Arguments
    - ref: Reference to model block
    - inkey: input variable
    - lag: tap indices

    Returns:
    - tapped inputs, structured
    """
    assert inkey is not None, "block_models.tap needs input key inkey"
    assert lag is not None, "block_models.tap needs tapping 'lag', None given"
    if type(lag) is tuple:
        tapping = range(lag[0], lag[1])
    # print "%s.%s tap(%s).tap = %s" % (ref.__class__.__name__, ref.id, inkey, tapping)
    return ref.inputs[inkey]['val'][...,tapping]

def tap_flat(tap_struct):
    """block_models.tap_flat

    Return transposed and flattened view of structured input vector (aka matrix) 'tap_struct' 
    """
    return tap_struct.T.reshape((-1, 1))

def tap_unflat(tap_flat, tap_len = 1):
    """block_models..tap_unflat

    Return inverted tap_flat() by reshaping 'tap_flat' into numtaps x -1 and transposing
    """
    if tap_len == 1:
        return tap_flat.T
    else:
        return tap_flat.reshape((tap_len, -1)).T

################################################################################
# legacy tappings from funcs_models.py

def tap_imol_fwd(ref):
    return tap_imol_fwd_time(ref)
    
def tap_imol_fwd_time(ref):
    """tap for imol forward model

    tap into time for imol forward model

    Args:
    - ref(ModelBlock2): reference to ModelBlock2 containing all info

    Returns:
    - tapping(tuple): tuple consisting of predict and fit tappings
    """
    if ref.mdl['fwd']['recurrent']:
        tap_pre_fwd = tapping_imol_pre_fwd(ref)
        # tap_fit_fwd = tapping_imol_recurrent_fit_fwd(ref)
        tap_fit_fwd = tapping_imol_recurrent_fit_fwd_2(ref)
    else:
        tap_pre_fwd = tapping_imol_pre_fwd(ref)
        tap_fit_fwd = tapping_imol_fit_fwd(ref)

    return tap_pre_fwd, tap_fit_fwd

def tap_imol_fwd_modality(tap_pre_fwd, tap_fit_fwd):
    # collated fit and predict input tensors from tapped data
    X_fit_fwd = np.vstack((
        tap_fit_fwd['pre_l0_flat'],  # state.1: motor
        tap_fit_fwd['meas_l0_flat'], # state.2: measurement
        tap_fit_fwd['prerr_l0_flat'],# state.3: last error
    ))
    
    Y_fit_fwd = np.vstack((
        tap_fit_fwd['pre_l1_flat'], # state.2: next measurement
    ))
    
    X_pre_fwd = np.vstack((
        tap_pre_fwd['pre_l0_flat'],   # state.1: motor
        tap_pre_fwd['meas_l0_flat'],  # state.2: measurement
        tap_pre_fwd['prerr_l0_flat'], # state.3: last error
    ))
    return X_fit_fwd, Y_fit_fwd, X_pre_fwd

def tapping_imol_pre_fwd(ref):
    """tapping for imol inverse prediction

    state: pre_l0_{t}   # pre_l0
    state: pre_l0_{t-1} # meas
    """
    mk = 'fwd'
    rate = 1
    # most recent top-down prediction on the input
    pre_l1 = ref.inputs['pre_l1']['val'][
        ...,
        range(
            ref.mdl[mk]['lag_past'][0] + ref.mdl[mk]['lag_off_f2p'],
            ref.mdl[mk]['lag_past'][1] + ref.mdl[mk]['lag_off_f2p'])].copy()

    # most recent state measurements
    meas_l0 = ref.inputs['meas_l0']['val'][
        ...,
        range(
            ref.mdl[mk]['lag_past'][0] + ref.mdl[mk]['lag_off_f2p'],
            ref.mdl[mk]['lag_past'][1] + ref.mdl[mk]['lag_off_f2p'])].copy()
    
    # most 1-recent pre_l1/meas_l0 errors
    prerr_l0 = ref.inputs['prerr_l0']['val'][
        ...,
        range(ref.mdl[mk]['lag_past'][0] + ref.mdl[mk]['lag_off_f2p'],
              ref.mdl[mk]['lag_past'][1] + ref.mdl[mk]['lag_off_f2p'])].copy()
    
    # momentary pre_l1/meas_l0 error
    prerr_l0 = np.roll(prerr_l0, -1, axis = -1)
    # FIXME: get my own prediction from that time
    prerr_l0[...,[-1]] = ref.inputs['pre_fwd_l0']['val'][...,[-ref.mdl[mk]['lag_off_f2p']]] - meas_l0[...,[-1]]

    # corresponding output k steps in the past, 1-delay for recurrent
    pre_l0 = ref.inputs['pre_l0']['val'][
        ...,
        range(
            ref.mdl[mk]['lag_past'][0] + ref.mdl[mk]['lag_off_f2p'] - rate,
            ref.mdl[mk]['lag_past'][1] + ref.mdl[mk]['lag_off_f2p'] - rate)].copy()
    
    pre_l0 = np.roll(pre_l0, -1, axis = -1)
    # prerr_l0[...,[-1]] = ref.inputs['pre_l1']['val'][...,[-ref.mdl[mk]['lag_off_f2p']]] - meas_l0[...,[-1]]
    pre_l0[...,[-1]] = ref.mdl['inv']['pre_l0']
    
    return {
        'pre_l1': pre_l1,
        'meas_l0': meas_l0,
        'pre_l0': pre_l0,
        'prerr_l0': prerr_l0,
        'pre_l1_flat': pre_l1.T.reshape((-1, 1)),
        'meas_l0_flat': meas_l0.T.reshape((-1, 1)),
        'pre_l0_flat': pre_l0.T.reshape((-1, 1)),
        'prerr_l0_flat': prerr_l0.T.reshape((-1, 1)) * 1.0,
    }

def tapping_imol_fit_fwd(ref):
    mk = 'fwd'
    rate = 1
    # Y
    # Y.1: most recent measurement
    pre_l1 = ref.inputs['meas_l0']['val'][
        ...,
        range(
            ref.mdl[mk]['lag_future'][0] + 0,
            ref.mdl[mk]['lag_future'][1] + 0
        )].copy()

    # X
    # X.2 corresponding starting state k steps in the past
    meas_l0 = ref.inputs['meas_l0']['val'][
        ...,
        range(ref.mdl[mk]['lag_past'][0], ref.mdl[mk]['lag_past'][1])].copy()
    
    # X.3: corresponding error k steps in the past, 1-delay for recurrent connection
    prerr_l0 = ref.inputs['prerr_l0']['val'][
        ...,
        range(ref.mdl[mk]['lag_past'][0] + rate, ref.mdl[mk]['lag_past'][1] + rate)].copy()
    
    # X.1: corresponding output k steps in the past, 1-delay for recurrent
    pre_l0 = ref.inputs['pre_l0']['val'][
        ...,
        range(
            ref.mdl[mk]['lag_past'][0] - 0 + rate,
            ref.mdl[mk]['lag_past'][1] - 0 + rate
        )].copy()
    
    # range(ref.mdl[mk]['lag_future'][0], ref.mdl[mk]['lag_future'][1])].copy()
    
    return {
        'pre_l1': pre_l1,
        'meas_l0': meas_l0,
        'prerr_l0': prerr_l0,
        'pre_l0': pre_l0,
        'pre_l1_flat': pre_l1.T.reshape((-1, 1)),
        'meas_l0_flat': meas_l0.T.reshape((-1, 1)),
        'prerr_l0_flat': prerr_l0.T.reshape((-1, 1)) * 1.0,
        'pre_l0_flat': pre_l0.T.reshape((-1, 1)),
    }

def tap_imol_inv(ref):
    return tap_imol_inv_time(ref)
    
def tap_imol_inv_time(ref):
    """tap imol inverse

    tapping for imol inverse model

    Args:
    - ref(ModelBlock2): reference to ModelBlock2 containing all info

    Returns:
    - tapping(tuple): tuple consisting of predict and fit tappings
    """
    if ref.mdl['inv']['recurrent']:
        tap_pre_inv = tapping_imol_pre_inv(ref)
        # tap_fit_inv = tapping_imol_recurrent_fit_inv(ref)
        tap_fit_inv = tapping_imol_recurrent_fit_inv_2(ref)
    else:
        tap_pre_inv = tapping_imol_pre_inv(ref)
        tap_fit_inv = tapping_imol_fit_inv(ref)

    return tap_pre_inv, tap_fit_inv

def tap_imol_inv_modality(tap_pre_inv, tap_fit_inv):
    # collated fit and predict input tensors from tapped data
    X_fit_inv = np.vstack((
        tap_fit_inv['pre_l1_flat'],
        tap_fit_inv['meas_l0_flat'],
        tap_fit_inv['prerr_l0_flat'],
        ))
    Y_fit_inv = np.vstack((
        tap_fit_inv['pre_l0_flat'],
        ))
    
    X_pre_inv = np.vstack((
        tap_pre_inv['pre_l1_flat'],
        tap_pre_inv['meas_l0_flat'],
        tap_pre_inv['prerr_l0_flat'],
        ))
    return X_fit_inv, Y_fit_inv, X_pre_inv

################################################################################
# direct forward / inverse model learning via prediction dataset
def tapping_imol_pre_inv(ref):
    """tapping for imol inverse prediction
    """
    
    # most recent top-down prediction on the input
    pre_l1 = ref.inputs['pre_l1']['val'][
        ...,
        range(
            ref.mdl['inv']['lag_past'][0] + ref.mdl['inv']['lag_off_f2p'],
            ref.mdl['inv']['lag_past'][1] + ref.mdl['inv']['lag_off_f2p'])].copy()

    # most recent state measurements
    meas_l0 = ref.inputs['meas_l0']['val'][
        ...,
        range(ref.mdl['inv']['lag_past'][0] + ref.mdl['inv']['lag_off_f2p'], ref.mdl['inv']['lag_past'][1] + ref.mdl['inv']['lag_off_f2p'])].copy()
    
    # most 1-recent pre_l1/meas_l0 errors
    prerr_l0 = ref.inputs['prerr_l0']['val'][
        ...,
        range(ref.mdl['inv']['lag_past'][0] + ref.mdl['inv']['lag_off_f2p'], ref.mdl['inv']['lag_past'][1] + ref.mdl['inv']['lag_off_f2p'])].copy()
    # momentary pre_l1/meas_l0 error
    prerr_l0 = np.roll(prerr_l0, -1, axis = -1)
    # FIXME: get full tapping
    prerr_l0[...,[-1]] = ref.inputs['pre_l1']['val'][...,[-ref.mdl['inv']['lag_off_f2p']]] - meas_l0[...,[-1]]

    return {
        # 'pre_l1': pre_l1,
        # 'meas_l0': meas_l0,
        'prerr_l0': prerr_l0,
        'pre_l1_flat': pre_l1.T.reshape((-1, 1)),
        'meas_l0_flat': meas_l0.T.reshape((-1, 1)),
        'prerr_l0_flat': prerr_l0.T.reshape((-1, 1)) * 0.0,
    }

def tapping_imol_fit_inv(ref):
    rate = 1
    # X
    # last goal prediction with measurement    
    # most recent goal top-down prediction as input
    pre_l1 = ref.inputs['meas_l0']['val'][
        ...,
        range(ref.mdl['inv']['lag_past'][0] + ref.mdl['inv']['lag_off_f2p'], ref.mdl['inv']['lag_past'][1] + ref.mdl['inv']['lag_off_f2p'])].copy()
    # corresponding starting state k steps in the past
    meas_l0 = ref.inputs['meas_l0']['val'][
        ...,
        range(ref.mdl['inv']['lag_past'][0], ref.mdl['inv']['lag_past'][1])].copy()
    # corresponding error k steps in the past, 1-delay for recurrent
    prerr_l0 = ref.inputs['prerr_l0']['val'][
        ...,
        range(ref.mdl['inv']['lag_past'][0] + rate, ref.mdl['inv']['lag_past'][1] + rate)].copy()
    # Y
    # corresponding output k steps in the past, 1-delay for recurrent
    pre_l0 = ref.inputs['pre_l0']['val'][
        ...,
        range(ref.mdl['inv']['lag_future'][0] - ref.mdl['inv']['lag_off_f2p'] + rate, ref.mdl['inv']['lag_future'][1] - ref.mdl['inv']['lag_off_f2p'] + rate)].copy()
    # range(ref.mdl['inv']['lag_future'][0], ref.mdl['inv']['lag_future'][1])].copy()
    
    return {
        # 'pre_l1': pre_l1,
        # 'meas_l0': meas_l0,
        'prerr_l0': prerr_l0,
        # 'pre_l0': pre_l0,
        'pre_l1_flat': pre_l1.T.reshape((-1, 1)),
        'meas_l0_flat': meas_l0.T.reshape((-1, 1)),
        'prerr_l0_flat': prerr_l0.T.reshape((-1, 1)) * 0.0,
        'pre_l0_flat': pre_l0.T.reshape((-1, 1)),
        }

def tapping_imol_recurrent_fit_inv(ref):
    rate = 1
    # X
    # last goal prediction with measurement    
    # FIXME: rate is laglen

    prerr_l0 = ref.inputs['prerr_l0']['val'][
        ...,
        range(ref.mdl['inv']['lag_past'][0] + ref.mdl['inv']['lag_off_f2p'], ref.mdl['inv']['lag_past'][1] + ref.mdl['inv']['lag_off_f2p'])].copy()
    
    if ref.cnt < ref.thr_predict:
        # take current state measurement
        pre_l1 = ref.inputs['meas_l0']['val'][
            ...,
            range(ref.mdl['inv']['lag_past'][0] + ref.mdl['inv']['lag_off_f2p'], ref.mdl['inv']['lag_past'][1] + ref.mdl['inv']['lag_off_f2p'])].copy()
        # add noise (input exploration)
        pre_l1 += np.random.normal(0.0, 1.0, pre_l1.shape) * 0.01
    else:
        # take top-down prediction
        pre_l1_1 = ref.inputs['pre_l1']['val'][
            ...,
            range(ref.mdl['inv']['lag_past'][0] + ref.mdl['inv']['lag_off_f2p'], ref.mdl['inv']['lag_past'][1] + ref.mdl['inv']['lag_off_f2p'])].copy()
        # and current state
        pre_l1_2 = ref.inputs['meas_l0']['val'][
            ...,
            range(ref.mdl['inv']['lag_past'][0] + ref.mdl['inv']['lag_off_f2p'], ref.mdl['inv']['lag_past'][1] + ref.mdl['inv']['lag_off_f2p'])].copy()
        mdltr = np.square(max(0, 1.0 - np.mean(np.abs(prerr_l0))))
        # print "mdltr", mdltr
        # explore input around current state depending on pe state
        pre_l1 = pre_l1_2 + (pre_l1_1 - pre_l1_2) * mdltr # 0.05

    # most recent measurements
    meas_l0 = ref.inputs['meas_l0']['val'][
        ...,
        range(ref.mdl['inv']['lag_past'][0] + ref.mdl['inv']['lag_off_f2p'], ref.mdl['inv']['lag_past'][1] + ref.mdl['inv']['lag_off_f2p'])].copy()

    # update prediction errors
    prerr_l0 = np.roll(prerr_l0, -1, axis = -1)
    prerr_l0[...,[-1]] = pre_l1[...,[-1]] - meas_l0[...,[-1]]
    
    # Y
    # FIXME check - 1?
    pre_l0 = ref.inputs['pre_l0']['val'][
        ...,
        range(ref.mdl['inv']['lag_future'][0] - ref.mdl['inv']['lag_off_f2p'], ref.mdl['inv']['lag_future'][1] - ref.mdl['inv']['lag_off_f2p'])].copy()
    # range(ref.mdl['inv']['lag_future'][0] - ref.mdl['inv']['lag_off_f2p'] + rate, ref.mdl['inv']['lag_future'][1] - ref.mdl['inv']['lag_off_f2p'] + rate)].copy()

    # print "tapping_imol_recurrent_fit_inv shapes", pre_l1.shape, meas_l0.shape, prerr_l0.shape, pre_l0.shape
    
    return {
        # 'pre_l1': pre_l1,
        # 'meas_l0': meas_l0,
        'prerr_l0': prerr_l0,
        # 'pre_l0': pre_l0,
        'pre_l1_flat': pre_l1.T.reshape((-1, 1)),
        'meas_l0_flat': meas_l0.T.reshape((-1, 1)),
        'prerr_l0_flat': prerr_l0.T.reshape((-1, 1)) * 0.0,
        'pre_l0_flat': pre_l0.T.reshape((-1, 1)),
        }

def tapping_imol_recurrent_fit_inv_2(ref):
    rate = 1
    # X
    # last goal prediction with measurement    
    # FIXME: rate is laglen

    pre_l1 = ref.inputs['meas_l0']['val'][
        ...,
        range(ref.mdl['inv']['lag_past'][0] + ref.mdl['inv']['lag_off_f2p'], ref.mdl['inv']['lag_past'][1] + ref.mdl['inv']['lag_off_f2p'])].copy()
    
    meas_l0 = ref.inputs['meas_l0']['val'][
        ...,
        range(ref.mdl['inv']['lag_past'][0], ref.mdl['inv']['lag_past'][1])].copy()
    
    prerr_l0 = ref.inputs['prerr_l0']['val'][
        ...,
        range(ref.mdl['inv']['lag_past'][0] + ref.mdl['inv']['lag_off_f2p'], ref.mdl['inv']['lag_past'][1] + ref.mdl['inv']['lag_off_f2p'])].copy()
    # range(ref.mdl['inv']['lag_past'][0] + rate, ref.mdl['inv']['lag_past'][1] + rate)]
    prerr_l0 = np.roll(prerr_l0, -1, axis = -1)
    prerr_l0[...,[-1]] = pre_l1[...,[-1]] - meas_l0[...,[-1]]
    
    # pre_l1 -= prerr_l0[...,[-1]] * 0.1
    # Y
    pre_l0 = ref.inputs['pre_l0']['val'][
        ...,
        range(ref.mdl['inv']['lag_future'][0] - ref.mdl['inv']['lag_off_f2p'], ref.mdl['inv']['lag_future'][1] - ref.mdl['inv']['lag_off_f2p'])].copy()
    # range(ref.mdl['inv']['lag_future'][0] - ref.mdl['inv']['lag_off_f2p'] + rate, ref.mdl['inv']['lag_future'][1] - ref.mdl['inv']['lag_off_f2p'] + rate)].copy()
    
    return {
        # 'pre_l1': pre_l1,
        # 'meas_l0': meas_l0,
        'prerr_l0': prerr_l0,
        # 'pre_l0': pre_l0 * 1.0,
        'pre_l1_flat': pre_l1.T.reshape((-1, 1)),
        'meas_l0_flat': meas_l0.T.reshape((-1, 1)),
        'prerr_l0_flat': prerr_l0.T.reshape((-1, 1)) * 0.0,
        'pre_l0_flat': pre_l0.T.reshape((-1, 1)),
    }

