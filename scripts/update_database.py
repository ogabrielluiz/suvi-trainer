import os
import numpy as np
import argparse
import pandas as pd
from astropy.io import fits
from dateutil import parser as dateparser
import matplotlib.pyplot as plt

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
    return ap.parse_args()


def process_file(path):
    """ Open a single labeled image at path and get needed information, return as a dictionary"""
    info = dict()
    with fits.open(path) as hdu:
        head = hdu[0].header
        data = hdu[0].data
        labels = {theme: value for value, theme in list(hdu[1].data)}
    info['trainer'] = head['expert']
    info['date-label'] = dateparser.parse(head['date-lab'])
    info['date-observation'] = dateparser.parse(head['date-beg'])
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


if __name__ == "__main__":
    args = parse_args()
    fns = [fn for fn in os.listdir(args.directory) if ".fits" in fn]
    data = [process_file(fn) for fn in fns]
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(args.directory, "database.csv"))

    # make cool plots
    for theme in themes:
        plot_counts(df, theme)
