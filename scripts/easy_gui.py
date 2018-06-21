#!/usr/bin/env python

import os, argparse, glob, sys, random
import deepdish as dd
import numpy as np
from astropy.io import fits

import suvitrainer
from suvitrainer.config import *
from suvitrainer import gui


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--directory", help="working directory with structure")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="prints detailed status information")
    ap.add_argument("-r", "--repeat", default=2, help="number of times to label each image", type=int)
    ap.add_argument("--composite",
                    action="store_true",
                    help="assumes SUVI images are composite and looks in at second HDU")
    args = vars(ap.parse_args())
    return args


def open_group(group, directory, zero=True, hdunum=0):
    images = dict()
    headers = dict()
    for channel, path in group.items():
        with fits.open(directory + group[channel]) as f:
            header = f[hdunum].header
            data = f[hdunum].data
            if zero:
                data = np.nan_to_num(data)
                data[data < 0] = 0
            images[channel] = data
            headers[channel] = header
    return images, headers


if __name__ == "__main__":
    args = get_args()
    if not args['directory']:
        args['directory'] = os.getcwd()

    if args['composite']:
        hdunum = 1
    else:
        hdunum = 0

    db = dd.io.load(args['directory'] + "/groups.h5")
    groups = list(db['groups'].keys())
    flatten = lambda l: [item for sublist in l for item in sublist]
    groups = flatten([[group + "-" + str(i) for i in range(args['repeat'])] for group in groups])
    labeled = glob.glob(args['directory'] + "/labeled/*.fits")
    labeled = [l.split("/")[-1].split('.fits')[0] for l in labeled]
    to_label = list(set(groups).difference(labeled))

    print("There are {} left to label. You're doing great! Thanks!".format(len(to_label)))
    if len(to_label) > 0:
        to_label = random.choice(to_label)
        groupname = to_label.split("-")[0]
        group = db['groups'][groupname]['files']
    else:
        print("Great job! You're finished!")
        sys.exit()

    if args['verbose']:
        print("Running on {}".format(groupname))
        print("The group contents are:")
        print(group)

    data, headers = open_group(group, args['directory'] + "/images/", hdunum=hdunum)  # db['directory'])
    suvitrainer.gui.App(data, "{}/labeled/{}.fits".format(args['directory'], to_label),
                   group, db['directory'], headers[DEFAULT_HEADER]).mainloop()


