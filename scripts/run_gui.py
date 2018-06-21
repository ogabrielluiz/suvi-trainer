#!/usr/bin/env python
import sys

sys.path.append('..')
sys.path.append("../smachy")
from deepdish.io import load as h5load
import argparse

from astropy.io import fits

#import suvitrainer
from suvitrainer.config import *
from suvitrainer import gui


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("input",
                    help="input hdf5 database")
    ap.add_argument("output",
                    help="name of fits output")
    ap.add_argument("--group",
                    help="name of group to load")
    ap.add_argument("--relabel",
                    help="filepath of previously labeled map to retrain")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="prints verbose status information")
    ap.add_argument("--blank",
                    help="does not make suggestions for limb, outer space, or solar disk")
    args = vars(ap.parse_args())
    return args


def open_group(group, directory, zero=True):
    images = dict()
    headers = dict()
    for channel, path in group.items():
        with fits.open(directory + group[channel]) as f:
            header = f[0].header
            data = f[0].data
            if zero:
                data[data < 0] = 0
            images[channel] = data
            headers[channel] = header
    return images, headers


if __name__ == '__main__':
    args = get_args()
    database = h5load(args['input'])

    if args['group']:
        groupname = args['group']
        group = database['groups'][groupname]['files']
    else:
        groupnames = list(database['groups'].keys())
        groupname = groupnames[0]
        group = database['groups'][groupname]['files']

    if args['verbose']:
        print("Running on {}".format(groupname))
        print("The group contents are:")
        print(group)

    data, headers = open_group(group, database['directory'])
    suvitrainer.gui.App(data, args['output'], group,
                   database['directory'], headers[DEFAULT_HEADER],
                   blank=args['blank']).mainloop()

