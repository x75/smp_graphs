
from collections import OrderedDict

from numpy import ndarray

def print_dict(pdict = {}, level = 1, indent = None):
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
            if type(v) in [dict, OrderedDict, list, ndarray]:
                pstring += print_dict(pdict = v, level = level+1)
            else:
                pstring += print_dict(pdict = v, level = level+1, indent = "")
            pstring += ", "
        pstring += "\n%s}" % (level_indent_str_m1)
    elif type(pdict) in [list, ndarray]:
        pstring += "["
        for i, pdict_elem in enumerate(pdict):
            pstring += print_dict(pdict = pdict_elem, level = level+1)
            pstring += ", "
        pstring += "]"
    else:
        pstring += "%s"  % (pdict,)
        
    # print "level %d: %s: %s" % (level, k, pstring)
    return pstring

