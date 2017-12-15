#!/usr/bin/env python
# -*- encoding: utf-8

"""Pandas dataframe gui display

grabbed 20171116 from 
https://stackoverflow.com/questions/10636024/python-pandas-gui-for-viewing-a-dataframe-or-matrix
https://github.com/bluenote10/PandasDataFrameGUI
"""



"""
If you are getting wx related import errors when running in a virtualenv:
Either make sure that the virtualenv has been created using
`virtualenv --system-site-packages venv` or manually add the wx library
path (e.g. /usr/lib/python2.7/dist-packages/wx-2.8-gtk2-unicode) to the
python path.
"""
import argparse, os
import datetime
import pandas as pd
import numpy as np
import dfgui


# def create_dummy_data(size):

#     user_ids = np.random.randint(1, 1000000, 10)
#     product_ids = np.random.randint(1, 1000000, 100)

#     def choice(*values):
#         return np.random.choice(values, size)

#     random_dates = [
#         datetime.date(2016, 1, 1) + datetime.timedelta(days=int(delta))
#         for delta in np.random.randint(1, 50, size)
#     ]

#     return pd.DataFrame.from_items([
#         ("Date", random_dates),
#         ("UserID", choice(*user_ids)),
#         ("ProductID", choice(*product_ids)),
#         ("IntColumn", choice(1, 2, 3)),
#         ("FloatColumn", choice(np.nan, 1.0, 2.0, 3.0)),
#         ("StringColumn", choice("A", "B", "C")),
#         ("Gaussian 1", np.random.normal(0, 1, size)),
#         ("Gaussian 2", np.random.normal(0, 1, size)),
#         ("Uniform", np.random.uniform(0, 1, size)),
#         ("Binomial", np.random.binomial(20, 0.1, size)),
#         ("Poisson", np.random.poisson(1.0, size)),
#     ])

# df = create_dummy_data(1000)

def load_hdf(args):
    h5f = None
    if os.path.exists(args.file):
        h5f = pd.HDFStore(args.file)
    else:
        print("Couldn't load file %s" % (args.file, ))
    return h5f

def main_list_groups(args):
    h5f = load_hdf(args)

    if h5f is None: return 1

    print("Datafile = %s" % (h5f.filename))
    print("    Keys = %s" % (list(h5f.keys())))    
    print("  Groups = %s" % (h5f.groups()))    
    
def main_show_group(args):
    h5f = load_hdf(args)
    
    # dfs['experiments'] = '/experiments'
    # dfs['blocks'] = '/blocks'
        
    if h5f is None: return 1

    print("HDFStore.keys = %s" % (list(h5f.keys()), ))

    df = None
    if args.group in list(h5f.keys()):
        print("Found group %s in h5f.keys" % (args.group, ))
        df = h5f[args.group]

    if df is None: return 1
        
    dfgui.show(df)

if __name__ == '__main__':

    # interactive mode, ongoing reload etc
    # modes = ['list_groups', 'show_group']
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file',  type=str, default = 'data.h5',     help = 'Data file to load [data.h5]')
    parser.add_argument('-g', '--group', type=str, default = 'experiments', help = 'For mode \'show_group\' which group to show [experiments]')
    parser.add_argument('-m', '--mode',  type=str, default = 'list_groups', help = 'Program exec mode [list_groups]')
    
    args = parser.parse_args()

    if 'list_group' in args.mode:
        main_list_groups(args)
    elif args.mode == 'show_group':
        main_show_group(args)
    
