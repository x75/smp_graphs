"""tapping.py

Tapping is a graphical model mapping sensorimotor data at different times t_tap to
a single machine learning training data point at time t

See https://arxiv.org/abs/1704.07622

Full tapping
 - tap: get input at times t_tap retaining structure
 - tap_?: do any computations in structured space
 - tap_flat: transpose and flatten structured tap
 - ref.mdl.step: apply to model fit / predict cycle
 - tap_struct: restructure flat prediction

"""

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
        tapping = list(range(lag[0], lag[1]))
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
