#!/usr/bin/env python

import argparse
import os
import random
import sys
import urllib.request
from datetime import datetime, timedelta

from dateutil import parser as dateparser

from suvitrainer.fileio import Fetcher, get_dates_link, get_dates_file
from suvitrainer.gui import App
from suvitrainer.config import Config

import numpy as np


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true", dest='verbose',
                    help="prints detailed status information")
    ap.add_argument("--date",
                    help="path to text file or URL with dates on each line alternatively it can be a single date",
                    default="https://raw.githubusercontent.com/jmbhughes/suvi-trainer/master/dates.txt")
    ap.add_argument("--config",
                    help="path to config file",
                    default="config.json")
    return ap.parse_args()


if __name__ == "__main__":
    args = get_args()
    config = Config(args.config)
    if args.verbose:
        print("Launching trainer")

    # Try to load the dates file and pick a date
    if os.path.isfile(args.date):
        dates = get_dates_file(args.date)
    else:  # not a local file so it's either a website or a date
        try:  # it's a link
            dates = get_dates_link(args.date)
        except:# (ValueError, urllib.error.URLError) as e:
            try:
                dates = [dateparser.parse(args.date)]
            except:
                sys.exit("The date you specified is in valid. " +
                         "It must be a link to a dates file, a path to a local dates file, or a date. " +
                         "Date formatting is not picky but an acceptable template is 2018-08-01T23:15")

    # pick a date to annotate
    date = np.random.choice([date for date, prob in dates],
                            p=[prob for date, prob in dates])

    if args.verbose:
        print("Running for {}".format(date))

    # Load data and organize input
    f = Fetcher(date, products=config.products)
    results = f.fetch()

    data, headers = dict(), dict()
    for product in results:
        head, d = results[product]
        data[config.products_map[product]] = d
        headers[config.products_map[product]] = head

    # the output filenames are structured as the "thmap_[date of observation]_[date of labeling].fits"
    out_file_name = "thmap_{}_{}.fits".format(date.strftime("%Y%m%d%H%M%S"),
                                              datetime.utcnow().strftime("%Y%m%d%H%M%S"))
    App(data, out_file_name, None, None, headers, args.config).mainloop()