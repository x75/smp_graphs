"""smp_graphs utils

some general utils for data structures and ops
"""

from collections import OrderedDict
from functools import partial

from numpy import ndarray
from numpy import float64
from numpy import transpose, roll, arange, array

def print_dict(pdict = {}, level = 1, indent = None):
    """pretty print a dictionary for debugging by recursive construction of print string

    replace classes, objects, functions with their repr string within ''
    """
    # if level == 0:
    #     print "#" * 80
    pstring = ""
    level_indent = 3
    level_marker = " "
    level_indent_str    = level_marker * level * level_indent
    level_indent_str_m1 = level_marker * (level - 1) * level_indent
    
    if indent is None:
        indent = level_indent_str
    
    # print "pdict[level = %d] = %s, type = %s\n\n" % (level, pdict, type(pdict))
    if type(pdict) in [dict, OrderedDict]:
        # pstring += "\n%s{" % (indent)
        pstring += "{"
        for i, item in enumerate(pdict.items()):
            k, v = item
            # if i > 0:
            pstring += "\n%s" % (indent)
            pstring += print_dict(pdict = k, level = level+1)
            pstring += ": "
            if type(v) in [dict, OrderedDict, list, ndarray, tuple]:
                pstring += print_dict(pdict = v, level = level+1)
            else:
                pstring += print_dict(pdict = v, level = level+1, indent = "")
            pstring += ", "
        pstring += "\n%s}" % (level_indent_str_m1)
    elif type(pdict) in [list, ndarray, tuple]:
        # fix start end markers for list / array / tuple
        s1 = '['
        s2 = ']'
        if type(pdict) == tuple:
            s1 = '('
            s2 = ')'
            
        pstring += s1
        for i, pdict_elem in enumerate(pdict):
            pstring += print_dict(pdict = pdict_elem, level = level+1)
            pstring += ', '
        pstring += s2
    else:
        if type(pdict) not in [int, float, float64, bool, tuple, None]:
            # replace single quotes within string
            pdictstr = str(pdict).replace("'", "\\'")
            # print type(pdictstr), pdict, pdictstr.replace("'", "X")
            pstring += "'%s'"  % (pdictstr,)
        else:
            pstring += "%s"  % (pdict,)
            
    # print "level %d: %s: %s" % (level, k, pstring)
    # if "array" in pstring:
    #     print "array", pstring, level, indent
    return pstring

################################################################################
# utils, TODO move to utils.py
def ordereddict_insert(ordereddict = None, insertionpoint = None, itemstoadd = []):
    """ordereddict_insert

    Self rolled ordered dict insertion from http://stackoverflow.com/questions/29250479/insert-into-ordereddict-behind-key-foo-inplace
    """
    assert ordereddict is not None
    assert insertionpoint in list(ordereddict.keys()), "insp = %s, keys = %s, itemstoadd = %s" % (insertionpoint, list(ordereddict.keys()), itemstoadd)
    new_ordered_dict = ordereddict.__class__()
    for key, value in list(ordereddict.items()):
        new_ordered_dict[key] = value
        if key == insertionpoint:
            # check if itemstoadd is list or dict
            if type(itemstoadd) is list:
                for item in itemstoadd:
                    keytoadd, valuetoadd = item
                    new_ordered_dict[keytoadd] = valuetoadd
            else:
                for keytoadd, valuetoadd in list(itemstoadd.items()):
                    new_ordered_dict[keytoadd] = valuetoadd
        # else:
        #     print "insertionpoint %s doesn't exist in dict" % (insertionpoint)
        #     sys.exit(1)
    ordereddict.clear()
    ordereddict.update(new_ordered_dict)
    return ordereddict

def xproduct(f, tup):
    """compute the cartesian product of a variable number of tensor dimensions"""
    assert len(tup) > 0
    if len(tup) > 1: # still dimensions left
        return xproduct(partial(f, list(range(tup[0]))), tup[1:]) # call yourself with partial of current dim and remaining dims
    # until nothings left
    return [elem for elem in f(list(range(tup[0])))]

def mytupleroll(tup, direction = 1):
    return tuple(roll(array(tup), shift = direction))

def mytuple(tup, direction = 1):
    return tuple(roll(arange(len(tup)), shift = direction))

def myt(a, direction = 1):
    """my transpose (transpose 'time' axis first to last)"""
    return a.transpose(mytuple(a.shape, direction))
