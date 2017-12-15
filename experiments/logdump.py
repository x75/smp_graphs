"""smp_graphs logging util

2017 Oswald Berthold

load a logfile and dump the initial and final config dictionaries to stdout
"""


import argparse
import smp_graphs.utils_logging as log

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--logfile", default=None, help="Logfile to load", type=str)

    args = parser.parse_args()

    assert args.logfile is not None, "Need a logfile to load, use --logfile logfile.h5"
    
    print("conf ini", log.log_pd_dump_config(args.logfile, storekey = 'conf'))
    print("conf fin", log.log_pd_dump_config(args.logfile, storekey = 'conf_final'))
