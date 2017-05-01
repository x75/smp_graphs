
import argparse
import time

import matplotlib.pyplot as plt

from smp_graphs.block import Block

def get_args():
    # define defaults
    default_conf     = "conf/default.py"
    default_numsteps = None # 10
    # create parser
    parser = argparse.ArgumentParser()
    # add required arguments
    parser.add_argument("-c", "--conf",     type=str, default=default_conf,     help="Configuration file [%s]" % default_conf)
    parser.add_argument("-n", "--numsteps", type=int, default=default_numsteps, help="Number of outer loop steps [%s]" % default_numsteps)
    # parse arguments
    args = parser.parse_args()
    # return arguments
    return args

def get_config_raw(conf):
    # open and read config file containing a dictionary spec
    s = open(conf, "r").read()

    # parse config into variable, easy
    # conf = eval(s)

    # proper version with more powerS!
    code = compile(s, "<string>", "exec")
    global_vars = {}
    local_vars  = {}
    exec(code, global_vars, local_vars)

    conf = local_vars["conf"]

    # print "conf", conf

    return conf

def set_config_defaults(conf):
    if not conf['params'].has_key("numsteps"):
        conf['params']['numsteps'] = 100
    return conf

def make_expr_id(name = "experiment"):
    """return experiment signature as name and timestamp"""
    return "%s_%s" % (name, make_expr_sig())

def make_expr_sig(args =  None):
    """return experiment timestamp"""
    return time.strftime("%Y%m%d_%H%M%S")

class Experiment(object):
    """smp_graphs Experiment

Arguments:
   args: argparse configuration namespace (key, value)


Load a config from the file in args.conf

    """
    def __init__(self, args):
        self.conf = get_config_raw(args.conf)
        self.conf = set_config_defaults(self.conf)

        # print "%s.init: conf = %s" % (self.__class__.__name__, self.conf)

        self.numsteps = self.conf['params']['numsteps']

        self.topblock = Block(
            block = self.conf['block'],
            conf = self.conf['params'],
        )

        print self.conf['params']

    def run(self):
        print "%s.run: conf['numsteps'] = %d" % (self.__class__.__name__, self.numsteps)
        
        for i in xrange(self.numsteps):
            topblock_x = self.topblock.step(x = None)

        print "final return value topblock.x = %s" % (topblock_x)

        plt.show()
