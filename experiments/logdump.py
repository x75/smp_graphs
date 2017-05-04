"""smp_graphs logging util

2017 Oswald Berthold

load a logfile and dump the initial and final config dictionaries to stdout
"""


import argparse
import smp_graphs.logging as log

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--logfile", default="data/logfile.h5", help="Logfile to load", type=str)

    args = parser.parse_args()

    print "conf ini", log.log_pd_dump_config(args.logfile, storekey = 'conf')
    print "conf fin", log.log_pd_dump_config(args.logfile, storekey = 'conf_final')
