import os
import argparse
import glob
from datetime import datetime
from astropy.io import fits
import numpy as np


def parse_args():
    """ parse arguments"""
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", help="where the labeled data is saved")
    ap.add_argument("output", help="the drectory to save files")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # check that the output directory exists
    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    # load all the files and determine their dates
    file_names = glob.glob(args.directory + "/*.fits")
    dates = [datetime.strptime(os.path.basename(fn).split("_")[1], "%Y%m%d%H%M%S")
             for fn in file_names]

    # group labeled images together
    grouped = dict()
    for fn, date in zip(file_names, dates):
        key = date.strftime("%Y%m%d")
        if key in grouped:
            grouped[key].append(fn)
        else:
            grouped[key] = [fn]

    for group in grouped:
        for el in grouped[group]:
            print(el)
        labels = np.array([fits.open(fn)[0].data for fn in grouped[group]])
        if len(grouped[group]) > 1:
            same = np.prod([(labels[i - 1, :, :] == labels[i, :, :])
                            for i in range(1, len(labels))], axis=0)
            labels[0][np.logical_not(same)] = 0
        ref_fn = grouped[group][0]
        ref = fits.open(ref_fn)
        ref[0].data = labels[0]
        ref.writeto(os.path.join(args.output,
                                 "_".join(os.path.basename(ref_fn).split("_")[:2]) + "_consensus.fits"))
        print("-"*80)