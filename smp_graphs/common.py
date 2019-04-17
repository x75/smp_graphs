
import traceback
import os, sys, pickle, re, ast

from collections import OrderedDict

# import functools
from functools import partial
from types import FunctionType

import logging

import smp_graphs
from smp_base.common import set_attr_from_dict
from smp_base.common import get_module_logger
from smp_graphs.utils import print_dict

logger = get_module_logger(modulename = 'common', loglevel = logging.INFO)

def tuple2inttuple(tpl):
    tpl = tuple((int(_) for _ in tpl))
    return tpl


################################################################################
# static config templates
conf_header = """from smp_graphs.experiment import make_expr_id

from collections import OrderedDict
from functools import partial
import copy 

import numpy as np

from smp_graphs.block import Block2, ConstBlock2, CountBlock2, DelayBlock2, UniformRandomBlock2
from smp_graphs.block import FuncBlock2, LoopBlock2, SeqLoopBlock2
from smp_graphs.block import dBlock2, IBlock2, DelayBlock2
from smp_graphs.block_ols import FileBlock2
from smp_graphs.block_plot import PlotBlock2
from smp_graphs.block_plot import SnsMatrixPlotBlock2, ImgPlotBlock2
from smp_graphs.block_models import ModelBlock2

from smp_graphs.funcs import f_sin, f_motivation, f_motivation_bin, f_random_uniform

from smp_base.plot import timeseries, histogram, histogramnd, linesegments

debug = False
showplot = True
saveplot = False
randseed = 0
recurrent = False
ros = True
"""

conf_footer = """# top block
conf = {
    'block': Block2,
    'params': {
        'id': None,
        'debug': debug,
        'topblock': True,
        'numsteps': numsteps,
        'graph': graph,
        'showplot': showplot,
        'saveplot': saveplot,
        'randseed': randseed,
        'recurrent': recurrent,
        'ros': ros,
    }
}
"""

#         'id': make_expr_id(),


# loop_delim = '|'
loop_delim = '_ll'

################################################################################
# dynamic code and function mangling
def code_compile_and_run(code = '', gv = {}, lv = {}, return_keys = []):
    """Compile and run the code given in the str 'code',
    optionally including the global and local environments given
    by 'gv', and 'lv'.

    FIXME: check if 'rkey' is in str code and convert to '%s = %s' % (rkey, code) if it is not

    Returns:
    - r(dict): lv if |return_keys| = 0, lv[return_keys] otherwise
    """
    code_ = compile(code, "<string>", "exec")
    exec(code, gv, lv)
    # no keys given, return entire local variables dict
    if len(return_keys) < 1:
        return lv
    # single key given, return just the value of this entry
    elif len(return_keys) == 1:
        if return_keys[0] in lv:
            return lv[return_keys[0]]
    # several keys given, filter local variables dict by these keys and return
    else:
        return dict([(k, lv[k]) for k in return_keys if k in lv])

# AST rewriting
class RewriteDict(ast.NodeTransformer):

    def visit_Dict(self, node):
        print("node dict", node, dir(node))
        # return ast.copy_location(ast.Subscript(
        #     value=ast.Name(id='data', ctx=ast.Load()),
        #     slice=ast.Index(value=ast.Str(s=node.id)),
        #     ctx=node.ctx
        # ), node)

class RewriteName(ast.NodeTransformer):

    def visit_Name(self, node):
        if node.id == 'lconf':
            print("node name", node.id, node.ctx) # dir(node)
            if isinstance(node.ctx, ast.Store):
                print("hu store", ast.parse(node.ctx))
        # return ast.copy_location(ast.Subscript(
        #     value=ast.Name(id='data', ctx=ast.Load()),
        #     slice=ast.Index(value=ast.Str(s=node.id)),
        #     ctx=node.ctx
        # ), node)
        
class RewriteAssign(ast.NodeTransformer):
    def __init__(self, lconf):
        self.lconf = lconf

    def visit_Assign(self, node):
        # print "node.targets", node.targets
        if isinstance(node.targets[0], ast.Name):
            # print "Found Name", type(node.targets[0].id), node.targets[0].id
            if node.targets[0].id == 'lconf' and self.lconf is not None:
                # print "founde lconf", type(node.targets[0].id), node.targets[0].id
                # print "lconf", type(self.lconf)
                lconf_parsed = ast.parse(str(self.lconf), '<string>', 'exec')
                # print "lconf_parsed", lconf_parsed.body[0].value
                # print "lconf_old", node.value
                # lconf_parsed = ast.fix_missing_locations(lconf_parsed)
                # print "node.value", type(node.value)
                # print "node.value", node.value._attributes
                # print "node.value", node.value._fields
                # print "node.value", node.value.keys
                # print "node.value", node.value.values

                # # print debugging ast objects
                # for k, v in zip(node.value.keys, node.value.values):
                #     # print "dir(k)", dir(k)
                #     # print "dir(v)", dir(v)
                #     print "node.value old",
                #     if hasattr(k, 's'):
                #         print "k = %s" % (k.s, ),
                #     if hasattr(v, 'n'):
                #         print "v = %s" % (v.n, ),
                #     print ""
                    
                # for k, v in zip(lconf_parsed.body[0].value.keys, lconf_parsed.body[0].value.values):
                #     # print "dir(k)", dir(k)
                #     # print "dir(v)", dir(v)
                #     print "node.value new",
                #     if hasattr(k, 's'):
                #         print "k = %s" % (k.s, ),
                #     if hasattr(v, 'n'):
                #         print "v = %s" % (v.n, ),
                #     print ""
                    
                node.value = lconf_parsed.body[0].value
                # node = ast.fix_missing_locations(node)
                # print "node", node.lineno, lconf_parsed.body[0].value.lineno
                # return node
        #         # swap old for parsed
        # return ast.copy_location(ast.Subscript(
        #     value=ast.Assign(id='data', ctx=ast.Load()),
        #     slice=ast.Index(value=ast.Str(s=node.id)),
        #     ctx=node.ctx
        # ), node)
        return node
        
def get_config_raw(conf, confvar = 'conf', lconf = None, fexec = True):
    """base.common.py

    Open config file, read it and call get_config_raw_from_string on that string
    """
    # open and read config file containing a python dictionary
    try:
        s_ = open(conf, "r").read()
    except Exception as e:
        print(e)
        sys.exit(1)

    # compile and evaluate the dictionary code string and return the dict object
    if fexec:
        return get_config_raw_from_string(s_, confvar = confvar, lconf = lconf)
    # or just return the string
    else:
        return s_

def get_config_raw_from_string(conf, confvar = 'conf', lconf = None):
    """base.common.get_config_raw_from_string

    Compile the 'conf' string and return the resulting 'confvar' variable
    """
    # prepend / append header and footer
    # FIXME: this is smp_graphs specific
    lineoffset = 0
    lineoffset += conf_header.count('\n')
    lineoffset += conf_footer.count('\n')
    s   = "%s\n%s\n%s" % (conf_header, conf, conf_footer)

    # new 20171002 with ast replacements
    # tree = compile(s, "<string>", "exec", ast.PyCF_ONLY_AST)
    tree = ast.parse(s, "<string>", "exec")
    # print "tree", tree.body
    rwa = RewriteAssign(lconf = lconf)
    tree = rwa.visit(tree)
    tree = ast.fix_missing_locations(tree)
    
    # sys.exit()
    # tree = RewriteName().visit(tree)
    # tree = RewriteDict().visit(tree)

    # # print "tree", ast.dump(tree)
    # sawlconf = False
    # for node in ast.walk(tree):
    #     # print "node", node, dir(node)
    #     # print "node", node.id
    #     if isinstance(node, ast.Name):
    #         # print(node.name)
    #         print "    ", node.id
    #         if node.id == 'lconf':
    #             print("lconf", node.id)
    #             sawlconf = True
    #     elif sawlconf:
    #         print "node", node
    #         sawlconf = False
    # # for node in ast.walk(tree):
    # #     print "tree", node

    # load config by compiling and running the configuration code

    # make sure to catch syntax errors in the config for debugging
    try:
        # code = compile(s, "<string>", "exec")
        # code = compile(tree, "<ast>", "exec")
        code = compile(tree, "<ast>", "exec")
        
        # ast = ast.parse(s, '<string>', 'exec')
        # print "ast", dir(ast)
        # print "code", dir(code)
        # print "code.argcount", code.co_argcount
        # print "code.cellvars", code.co_cellvars
        # print "code.consts", code.co_consts
        # print "code.freevars", code.co_freevars
        # print "code.nlocals", code.co_nlocals
        
    except Exception as e:
        print("\n%s" % (e.text))
        print("Compilation of %s failed with %s in %s at line %d" % (conf, e.msg, e.filename, e.lineno))
        sys.exit(1)

    # prepare variables
    global_vars = {}
    local_vars  = {}

    # copy lconf
    if lconf is not None:
        # global_vars['lconf'] = lconf
        # local_vars.update(lconf)
        global_vars.update(lconf)
    
    # run the code
    try:
        exec(code, global_vars, local_vars)
    except Exception as e:
        # FIXME: how to get more stack context?
        traceback.print_exc(limit = 10)
        # print traceback
        print("Error running config code: %s" % (repr(e),))
        print("Probably causes:\n    missing parentheses or comma\n    dict key followed by , instead of :\nin config")
        sys.exit(1)

    # print "get_config_raw_from_string local_vars", local_vars.keys()
    # print "get_config_raw_from_string global_vars", global_vars.keys()
        
    # return resulting variable
    if confvar is None:
        return local_vars
    else:
        return local_vars[confvar]

def get_input(inputs, inkey):
    """smp_graphs.common.get_input

    An smp_graphs bus operation: return the 'val' field of the inputs' item at 'inkey'
    """
    assert type(inputs) in [dict, OrderedDict]
    assert inkey is not None
    assert inkey in inputs
    return inputs[inkey]['val']

def compress_loop_id(k):
    if '/' in k: # FIXME: control explicitly with argument
        k_ = re.sub(r'([A-Za-z0-9]+)(_(ll)([0-9]+))+\/(.*)', r'\1/\5', k)
    else:
        k_ = re.sub(r'([A-Za-z0-9]+)(_(ll)([0-9]+))+', r'\1', k)
    # logger.debug('compress_loop_id: squashing %s down to %s', k, k_)
    return k_

def dict_replace_nodekeys(d, xid, idmap = {}):
    """smp_graphs.common.dict_replace_nodekeys

    replace all keys in d by a copy with xid appended (looping) and store key/replacement in idmap

    Returns:
    - d: dictionary
    - idmap: map old -> new ids
    """

    # loop over dictionary items
    for k, v in list(d.items()):
        # print "dict_replace_nodekeys: k = %s, v = %s, idmap = %s" % (k, v.keys(), idmap)
        # new id from old id
        if type(xid) is tuple:
            k_ = "%s%s%s%s" % (k, xid[0], loop_delim, xid[1])
        else:
            k_ = "%s%s%s" % (k, loop_delim, xid)
        # fix key:
        d[k_] = d.pop(k) # FIXME: does this confuse .items(), FIXME: even worse, does this confuse the order in OrderedDict?
        # print "k_", k_, "v_", d[k_].keys()

        # # debug
        # if not d[k_].has_key('params'):
        #     print "\n\n\n\n\nd[k_]", d[k_].keys()
        #     return d, idmap
        
        # fix block id
        d[k_]['params']['id'] = k_
        idmap[k] = k_
        
        # descend into graph hierarchy
        if 'graph' in d[k_]['params']:
            d[k_]['params']['graph'], idmap = dict_replace_nodekeys(d[k_]['params']['graph'], xid, idmap)

        # descend into loopblock
        if 'loopblock' in d[k_]['params'] and 'graph' in d[k_]['params']['loopblock']['params']:
            d[k_]['params']['loopblock']['params']['graph'], idmap = dict_replace_nodekeys(d[k_]['params']['loopblock']['params']['graph'], xid, idmap)
            # print "\n\n\n\n\n\n\n", d[k_]['params']['loopblock']['params']['graph']

    # return to surface
    return d, idmap

def dict_replace_nodekeyrefs(d, xid, idmap):
    """smp_graphs.common.dict_replace_nodekeyrefs

    replace all references to ids in idmap.keys with corresponding idmap.values

    Returns:
    - d: dictionary
    - idmap: map old -> new ids
    """
    
    # loop over dictionary items
    for k_, v in list(d.items()):
        # fix bus references in inputs
        if 'inputs' in d[k_]['params']:
            for ink, inv in [(ink, inv) for ink, inv in list(d[k_]['params']['inputs'].items()) if 'bus' in inv]:
                # if inv.has_key('bus'):
                invbuss = inv['bus'].split("/")
                if invbuss[0] in list(idmap.keys()):
                    # print "ink, inv", ink, inv
                    # print "idmap", idmap
                    buskey = "%s/%s" % (idmap[invbuss[0]], invbuss[1])
                    # print "dict_replace_nodekeyrefs: replacing %s with %s in node %s" % (inv['bus'], buskey, k_)
                    inv['bus'] = buskey
                    
        # fix bus references in outputs
        if 'outputs' in d[k_]['params']:
            for ink, inv in [(ink, inv) for ink, inv in list(d[k_]['params']['outputs'].items()) if dict_has_keys_any(inv, ['buscopy', 'trigger'])]:
                # if inv.has_key('bus'):
                for replkey in ['buscopy', 'trigger']:
                    if replkey not in inv: continue
                    invbuss = inv[replkey].split("/")
                    if invbuss[0] in list(idmap.keys()):
                        # print "ink, inv", ink, inv
                        # print "idmap", idmap
                        buskey = "%s/%s" % (idmap[invbuss[0]], invbuss[1])
                        # print "dict_replace_nodekeyrefs: replacing %s with %s in node %s" % (inv[replkey], buskey, k_)
                        inv[replkey] = buskey
                    # print "\n\n\n\n\n\n\noutputs, buscopy", d[k_]['params']

        # fix id references in 'copy' models (want to copy the model from the block with that id)
        if 'models' in d[k_]['params']:
            for ink, inv in [(ink, inv) for ink, inv in list(d[k_]['params']['models'].items()) if 'copyid' in inv]:
                inv['copyid'] = idmap[inv['copyid']]
            # print "\n\n\n\n\n\nfound model", d[k_]['params']['id'], d[k_]['params']['models']

        # descend into graph hierarchy
        if 'graph' in d[k_]['params']:
            d[k_]['params']['graph'], idmap = dict_replace_nodekeyrefs(d[k_]['params']['graph'], xid, idmap)

        # descend into loopblock
        if 'loopblock' in d[k_]['params'] and 'graph' in d[k_]['params']['loopblock']['params']:
            for ink, inv in list(d[k_]['params']['loopblock']['params']['outputs'].items()):
                # if dict_has_keys_any(inv, ['buscopy', 'trigger']):
                for replkey in ['buscopy', 'trigger']:
                    if replkey not in inv: continue
                    invbuss = inv[replkey].split("/")
                    if invbuss[0] in list(idmap.keys()):
                        # print "ink, inv", ink, inv
                        # print "idmap", idmap
                        buskey = "%s/%s" % (idmap[invbuss[0]], invbuss[1])
                        # print "dict_replace_nodekeyrefs: replacing %s with %s in node %s" % (inv[replkey], buskey, k_)
                        inv[replkey] = buskey

            d[k_]['params']['loopblock']['params']['graph'], idmap = dict_replace_nodekeyrefs(d[k_]['params']['loopblock']['params']['graph'], xid, idmap)
            # print "\n\n\n\n\n\n\n", d[k_]['params']['loopblock']['params']['graph']

    # return to surface
    return d, idmap

def dict_has_keys_any(d, keys):
    return len([k_ for k_ in list(d.keys()) if k_ in keys]) > 0

def dict_replace_idstr_recursive2(d, xid, idmap = {}):
    """smp_graphs.common.dict_replace_idstr_recursive2

    replace occurences of an 'id' string with a numbered version for looping

    Two steps
    1 - replace the dict keys
    2 - replace all references in the conf to dict keys

    Args:
    - d: dictionary
    - xid: extension to append to existing id
    - idmap: map of ids that have been modified

    Return:
    - d: the modified dict
    """

    d, idmap = dict_replace_nodekeys(d, xid, idmap)

    d, idmap = dict_replace_nodekeyrefs(d, xid, idmap)
        
    return d

def dict_replace_idstr_recursive(d, cid, xid):
    """smp_graphs.common.dict_replace_idstr_recursive

    Replace all references in dict 'd' to id 'cid' with 'cid{loop_delim}xid'

    Returns:
    - d(dict): the modified dict 'd'
    """
    assert 'params' in d
    # assert d['params'].has_key('id')

    # print "dict_replace_idstr_recursive", print_dict(d)

    if cid is None:
        return d
    
    # if cid is not None:
    # change param 'id' with loop marker and number
    d['params']['id'] = "%s%s%s" % (cid, loop_delim, xid)
    logger.debug("dict_replace_idstr_recursive newid = %s", d['params']['id'])

    # change param 'inputs'
    if 'inputs' in d['params']:
        for ink, inv in list(d['params']['inputs'].items()):
            # print "        cid = %s, id = %s, ink = %s, inv = %s" % (
            #    cid, d['params']['id'], ink, inv.keys())
            if 'bus' in inv:
                # print "        bus old", inv['bus']
                inv['bus'] = re.sub(
                    r'%s' % (cid, ),
                    r'%s' % (d['params']['id'], ),
                    inv['bus'])
                # print "        bus new", inv['bus']
                    
        # change param '?'

    # change params 'outputs'
    if 'outputs' in d['params']:
        for outk, outv in list(d['params']['outputs'].items()):
            # print "        cid = %s, id = %s, outk = %s, outv = %s" % (
            #    cid, d['params']['id'], outk, outv.keys())
            # if outv.has_key('bus'):
            for outvk in [k_ for k_ in list(outv.keys()) if k_ in ['trigger', 'buscopy']]:
                # if outv.has_key('bus'):
                #     print "        bus old", outv['bus']
                outv[outvk] = re.sub(
                    r'%s' % (cid, ),
                    r'%s' % (d['params']['id'], ),
                    outv[outvk])
                # print "        bus new", outv['bus']
                
    if 'graph' in d['params'] and type(d['params']['graph']) is not str:
        tgraph = OrderedDict()
        for k, v in list(d['params']['graph'].items()):
            # v['params']['id'] = k

            # replace ids in dict val
            v = dict_replace_idstr_recursive(d = v, cid = k, xid = xid)
            # replace ids in dict key
            k_ = v['params']['id']
            # reassemble
            tgraph[k_] = v
            d['params']['graph'][k] = v
        # copy new dict to graph
        # print "tgraph", tgraph
        d['params']['graph'] = tgraph
        # print "d['params']['graph']", d['params']['graph']
    return d

def dict_get_nodekeys_recursive(d):
    """dict_get_nodekeys_recursive

    Recursively get all nodekeys from a nested graph
    """
    nodekeys = set(d.keys())
    for nk in nodekeys:
        # print "nodekey", nk
        # print "graphkeys", d[nk]['params'].keys()
        if 'graph' in d[nk]['params']:
            # print "graphkeys", d[nk]['params']['graph'].keys()
            nodekeys = nodekeys.union(dict_get_nodekeys_recursive(d[nk]['params']['graph']))
    return nodekeys

def dict_replace_nodekeys_loop(d = {}, nodekeys = set(), loopiter = 0):
    """replace occurences of node keys in a dict with loop index
    """
    # logger.debug("dict_replace_nodekeys_loop dict = %s, nodekeys = %s, loopiter = %s", d.keys(), nodekeys, loopiter)
    loopiter_ = None
    if type(loopiter) is tuple:
        loopiter_ = loopiter
        loopiter = loopiter_[1]

    # iterate input dict
    for k, v in list(d.items()):
        # new id from old id
        # k_ = "%s%s%s" % (k, loop_delim, xid)
        # if key in nodekeys
        if k in nodekeys:
            # consider hierarchy info loopiter
            if loopiter_ is not None:
                k_ = "%s%s%s%s" % (k, loopiter_[0], loop_delim, loopiter)
            # else cold start
            else:
                k_ = re.sub(r'%s' % (k, ), r'%s%s%s' % (k, loop_delim, loopiter), k)
            # logger.debug("old k = %s, new k = %s" % (k, k_, ))
            # overwrite old key with replacement
            d[k_] = d.pop(k)
        # key unmodified
        else:
            k_ = k
            
        # print "k", k, "k_", k_, type(loopiter), nodekeys, type(d[k_])
        # d[k_] is number, str, list, dict

        # check value
        if type(d[k_]) is str:
            # logger.debug("dict value is str = %s", d[k_])
            for nk in nodekeys:
                # logger.debug("checking nodekey = %s", nk)
                # logger.debug("replacing occurence of k = %s in d[k_] = %s with string k_ =%s / %s", nk,  d[k_], nk, loopiter)
                if loopiter_ is not None:
                    d[k_] = re.sub(r'%s/' % nk, r'%s%s%s%s/' % (nk, loopiter_[0], loop_delim, loopiter), d[k_])
                else:
                    d[k_] = re.sub(r'%s/' % nk, r'%s%s%s/' % (nk, loop_delim, loopiter), d[k_])
                # print "replacing string k with string k_", d[k_]
                # logger.debug("replaced occurence of nodekey = %s in key = %s?", nk, d[k_])
                
        elif type(d[k_]) is dict:
            d[k_] = dict_replace_nodekeys_loop(d[k_], nodekeys, loopiter_)
        # elif type(d[k_]) is list:
        #     d[k_] = dict_replace_nodekeys_loop(d[k_], nodekeys, loopiter)
    return d

def vtransform(v):
    """transform pointer type objects to string repr for useful hashing
    """
    vtype = type(v)
    # print "vtransform v = %s, vtype = %s" % (v, vtype)
    if vtype is dict or vtype is OrderedDict or vtype is list:
        v_ = conf_strip_variables(v)
    else:
        # FIXME: this destroys information about the partial config,
        # maybe OK for changing the plots?
        # print "vtype", vtype, dir(smp_graphs)
        if vtype is partial: # functools.partial
            # print "    type partial"
            v_ = 'partial.func.%s' % v.func.__name__
        elif vtype is FunctionType:
            v_ = 'func.%s' % v.__name__
            # print "    type FunctionType", v, v_
        # FIXME: this produces import error when block_models isnt
        # imported in config?
        elif vtype is smp_graphs.block_models.model:
            v_ = 'model.%s' % v.modelstr
            # print "    type model", v, v_
        elif vtype is type:
            v_ = str(v)
            # print "    type type", v, v_
        else:
            v_ = v
    # if "func" in str(vtype):
    #     print "    type %s, v = %s, v_ = %s" % (vtype, v, v_)
    return v_
                
def conf_strip_variables(conf, omits = ['PlotBlock2']):
    """strip variables with uncontrollable values from a config dict for useful hashing
    """
    conftype = type(conf)
    # print "conf = %s" % (conftype, )
    # instantiate type
    conf_ = conftype()
    # omit keys
    omit_keys = ['timestamp', 'datafile_expr', 'cache_clear', 'saveplot', 'desc', 'plotgraph', 'docache']
    
    if conftype is dict or conftype is OrderedDict:
        # strip analysis blocks / PlotBlock2 from hashing
        if 'block' in conf:
            for omit in omits:
                # if 'PlotBlock2' in str(conf['block']): return conf_
                if omit in str(conf['block']): return conf_

        # for ok in omit_keys:
        #     if conf.has_key(ok):
        #         conf.pop(ok)
                    
        for k, v in list(conf.items()):
            if k in omit_keys: continue
            # print "v", v
            # if k == 'block' and 'PlotBlock2' in v: continue
            # vtype = type(v)
            # print "conf_strip_variables k = %s, v = %s/%s" % (k, vtype, v)
            v_ = vtransform(v)
            conf_[k] = v_
    elif conftype is list:
        for v in conf:
            conf_.append(vtransform(v))
            
    return conf_

def listify(obj, k):
    """listify

    Common configuration task: 1) get conf item, 2) check if it is compound
    (list, array, collection, ...) or atomic (str, number, scalar,
    ...), 3) if list get element at index k, 4) else return obj

    v1 based on
    ```
    block_plot.py:729 FIXME: robust listify-list-or-string-on-the-fly
                             and wrapped index for len-1 lists
    ```
    """
    # assert obj is not None, 'listify: obj is %s, failing' % (type(obj))
    if type(obj) in [list]:
        # logger.info('listify: obj = %s, k = %s', obj, k)
        # assert k < len(obj), "listify: index %s out of bounds %s, failing" % (k, len(obj))
        if len(obj) == 1:
            return obj[0]
        return obj[k]
    else:
        return obj
        
def read_puppy_hk_pickles(lfile, key = None):
    """smp_graphs.common.read pickled log dicts from andi's puppy experiments"""
    d = pickle.load(open(lfile, 'rb'))
    # print "d.keys", d.keys()
    # data = d["y"][:,0,0] # , d["y"]
    rate = 20
    offset = 0
    # data = np.atleast_2d(data).T
    # print "wavdata", data.shape
    # special treatment for x,y with addtional dimension
    data = d
    x = d['x']
    data['x'] = x[:,:,0]
    y = d['y']
    data['y'] = y[:,:,0]
    # print "x.shape", data.keys()
    return (data, rate, offset, d['x'].shape[0])

def md5(obj):
    """md5

    Compute md5 hash for obj using hashlib
    """
    import hashlib
    # print "self.conf", str(self.conf)
    # if type(obj) is not str:
    #     obj = str(obj)
    # print('type(obj)', type(obj))
    m = hashlib.md5(obj.encode())
    return m

def create_datadir(datadir = None):
    assert datadir is not None, "create_datadir needs a datadir argument"
    try:
        os.mkdir(datadir)
    except OSError as e:
        print("Couldn't create datadir = %s with error %s" % (datadir, e))
        return False
    
    return True

def check_datadir(conf = {}):
    """check if path :data:`conf['datadir']` exists and create it if it doesn't

    Arguments:
     - conf(dict): smp_graphs configuration dict containing the variables

      - datadir(str):  path top level datadir
      - datadir_expr(str):  path to experiment's datasubdir
      - datafile_expr(str):  filename base for experiment's files

    Returns:
     - status(bool): True if ok, False on error
    """
    for k in ['datadir', 'datadir_expr', 'datafile_expr']:
        assert k in conf, "check_datadir expects key %s in its 'conf' argument, conf.keys = %s" % (k, list(conf.keys()), )

    r = True
    
    if r and not os.path.exists(conf['datadir']):
        r = create_datadir(conf['datadir'])

    # print "r datadir", r
    if r and not os.path.exists(conf['datadir_expr']):
        r = create_datadir(conf['datadir_expr'])

    # print "r datadir_expr", r

    return r

def escape_backslash(s):
    return re.sub(r'_', r'\\_', s)

if __name__ == '__main__':

    print("testing function composition")

    def double(x):
        return x * 2

    def inc(x):
        return x + 1

    def dec(x):
        return x - 1

    inc_and_double = compose2(double, inc)
    print("double(inc(10)) = %s" % (inc_and_double(10), ))
    
    inc_double_and_dec = compose2(compose2(dec, double), inc)
    inc_double_and_dec(10)
    print("dec(double(inc(10))) = %s" % (inc_double_and_dec(10), ))
    
    inc_double_and_dec = compose(dec, double, inc )
    print("dec(double(inc(10))) = %s with compose(*functions)" % (inc_double_and_dec(10), ))
