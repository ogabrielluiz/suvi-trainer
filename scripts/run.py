#!/usr/bin/env python

import os, argparse, glob, sys, random
import deepdish as dd
from datetime import datetime
import numpy as np
from astropy.io import fits

import glob
from suvitrainer.gui import App
from suvitrainer.fileio import Fetcher
from suvitrainer.config import *


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="prints detailed status information")
    args = vars(ap.parse_args())
    return args


if __name__ == "__main__":
    args = get_args()
    date = datetime(2018, 5, 25, 18, 36)

    f = Fetcher(date)
    results = f.fetch()

    data, headers = dict(), dict()
    for product, head, d in results:
        data[PRODUCTS_MAP[product]] = d
        headers[PRODUCTS_MAP[product]] = head

    App(data, "test.fits",
        None, None, headers).mainloop()


