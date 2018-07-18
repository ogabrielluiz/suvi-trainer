#!/usr/bin/env python

import argparse
import os
import random
import sys
import urllib.request
from datetime import datetime, timedelta

from dateutil import parser as dateparser

from suvitrainer.config import *
from suvitrainer.fileio import Fetcher
from suvitrainer.gui import App


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true", dest='verbose',
                    help="prints detailed status information")
    ap.add_argument("--date",
                    help="path to text file or URL with dates on each line alternatively it can be a single date",
                    default="https://raw.githubusercontent.com/jmbhughes/suvi-trainer/master/dates.txt")
    return ap.parse_args()


def convert_time_string(date_str):
    """ Change a date string from the format 2018-08-15T23:55:17 into a datetime object """
    dt, _, us= date_str.partition(".")
    dt = datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S")
    us = int(us.rstrip("Z"), 10)
    return dt + timedelta(microseconds=us)


def get_dates_file(path):
    with open(path) as f:
        dates = f.readlines()
    return [convert_time_string(date_string) for date_string in dates]


def get_dates_link(url):
    # TODO: remove this temp.txt and make it less hardcoded, e.g. check for permissions
    urllib.request.urlretrieve(url, "temp.txt")
    dates = get_dates_file("temp.txt")
    os.remove("temp.txt")
    return dates


if __name__ == "__main__":
    args = get_args()
    if args.verbose:
        print("Launching trainer")

    # Try to load the dates file and pick a date
    if os.path.isfile(args.date):
        dates = get_dates_file(args.date)
    else:  # not a local file so it's either a website or a date
        try:  # it's a link
            dates = get_dates_link(args.date)
        except (ValueError, urllib.error.URLError) as e:
            try:
                dates = [dateparser.parse(args.date)]
            except:
                sys.exit("The date you specified is in valid. " +
                         "It must be a link to a dates file, a path to a local dates file, or a date. " +
                         "Date formatting is not picky but an acceptable template is 2018-08-01T23:15")

    # pick a date to annotate
    date = random.sample(dates, 1)[0]

    if args.verbose:
        print("Running for {}".format(date))

    # Load data and organize input
    f = Fetcher(date)
    results = f.fetch()

    data, headers = dict(), dict()
    for product in results:
        head, d = results[product]
        data[PRODUCTS_MAP[product]] = d
        headers[PRODUCTS_MAP[product]] = head

    # the output filenames are structured as the "thmap_[date of observation]_[date of labeling].fits"
    out_file_name = "thmap_{}_{}.fits".format(date.strftime("%Y%m%d%H%M%S"),
                                              datetime.now().strftime("%Y%m%d%H%M%S"))
    App(data, out_file_name, None, None, headers).mainloop()