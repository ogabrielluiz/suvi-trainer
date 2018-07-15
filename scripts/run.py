#!/usr/bin/env python

import argparse
import random
from datetime import datetime, timedelta

from suvitrainer.config import *
from suvitrainer.fileio import Fetcher
from suvitrainer.gui import App


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true", dest='verbose',
                    help="prints detailed status information")
    ap.add_argument("dates_path", help="path to text file with dates on each line")
    args = ap.parse_args()
    return args


def convert_time_string(date_str):
    dt, _, us= date_str.partition(".")
    dt = datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S")
    us = int(us.rstrip("Z"), 10)
    return dt + timedelta(microseconds=us)


if __name__ == "__main__":
    args = get_args()
    if args.verbose:
        print("Launching trainer")

    with open(args.dates_path) as f:
        dates = f.readlines()
    dates = [convert_time_string(date_string) for date_string in dates]
    date = random.sample(dates, 1)[0]

    if args.verbose:
        print("Running for {}".format(date))

    f = Fetcher(date)
    results = f.fetch()

    data, headers = dict(), dict()
    for product in results:
        head, d = results[product]
        data[PRODUCTS_MAP[product]] = d
        headers[PRODUCTS_MAP[product]] = head

    out_file_name = "thmap_{}_{}.fits".format(date.strftime("%Y%m%d%H%M%S"), datetime.now().strftime(("%Y%m%d%H%M%S")))
    App(data, out_file_name,
        None, None, headers).mainloop()


