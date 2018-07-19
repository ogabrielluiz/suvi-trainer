import os
import numpy as np
import argparse
import pandas as pd
from astropy.io import fits
from dateutil import parser as dateparser
import matplotlib.pyplot as plt
import glob

from suvitrainer.config import Config
from suvitrainer.fileio import get_dates_link, get_dates_file

import ipdb
from collections import Counter
"""
Using training data, creates a csv database that lists the date of observation, the date trained, 
how many pixels were classified key themes ('flare', 'coronal_hole', 'bright_region', 'filament', 'prominence'), 
and who the trainer is. 
"""

themes = ['flare', 'coronal_hole', 'bright_region', 'filament', 'prominence']


def parse_args():
    """ parse arguments"""
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", help="where the labeled data is saved")
    ap.add_argument("--config",
                    help="path to config file",
                    default="config.json")
    return ap.parse_args()


def process_file(path):
    """ Open a single labeled image at path and get needed information, return as a dictionary"""
    info = dict()
    with fits.open(path) as hdu:
        head = hdu[0].header
        data = hdu[0].data
        labels = {theme: value for value, theme in list(hdu[1].data)}
    info['filename'] = os.path.basename(path)
    info['trainer'] = head['expert']
    info['date-label'] = dateparser.parse(head['date-lab'])
    info['date-observation'] = dateparser.parse(head['date-end'])
    for theme in themes:
        info[theme + "_count"] = np.sum(data == labels[theme])
    return info


def plot_counts(df, theme):
    """ plot the counts of a given theme from a created database over time"""
    dates, counts = df['date-observation'], df[theme + "_count"]
    fig, ax = plt.subplots()
    ax.set_ylabel("{} pixel counts".format(" ".join(theme.split("_"))))
    ax.set_xlabel("observation date")
    ax.plot(dates, counts, '.')
    fig.autofmt_xdate()
    plt.show()


def update_dates_priority(df,
                          dates_url="https://raw.githubusercontent.com/jmbhughes/suvi-trainer/master/dates.txt"):
    priority = dict()
    dates = get_dates_link(dates_url)
    counted_dates = Counter([row['filename'].split("_")[1] for index, row in df.iterrows()])
    for date, weight in dates:
        key = date.strftime("%Y%m%d%H%M%S")
        if key in counted_dates:
            priority[date] = 1/(counted_dates[key] + 1)
        else:
            priority[date] = 1
    norm_factor = sum([pri for date, pri in priority.items()])
    priority = sorted([(date, pri) for date, pri in priority.items()], key = lambda entry: entry[1])
    new_dates = "\n".join([date.strftime("%Y-%m-%dT%H:%M:%S") + " " + str(pri/norm_factor)
                           for date, pri in priority])
    with open("dates.txt", 'w') as f:
        f.write(new_dates)


if __name__ == "__main__":
    args = parse_args()
    config = Config(args.config)

    fns = glob.glob(args.directory + "/*.fits")

    data = [process_file(fn) for fn in fns]
    df = pd.DataFrame(data)
    update_dates_priority(df)

    df.to_csv(os.path.join(args.directory, "database.csv"))

    # make cool plots
    for theme in themes:
        plot_counts(df, theme)