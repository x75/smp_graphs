
from collections import OrderedDict
from functools import partial

from numpy import ndarray
from numpy import float64

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

def xproduct(f, tup):
    """compute the cartesian product of a variable number of tensor dimensions"""
    if len(tup) > 1: # still dimensions left
        return xproduct(partial(f, range(tup[0])), tup[1:]) # call yourself with partial of current dim and remaining dims
    # until nothings left
    return [elem for elem in f(range(tup[0]))]

