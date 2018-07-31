#!/usr/bin/env python3
from suvitrainer.fileio import Fetcher
from datetime import datetime, timedelta
from dateutil import parser as date_parser
import argparse
from suvitrainer.config import Config
import numpy as np
import matplotlib.pyplot as plt


def get_args():
    """
    request the arguments for running
    """
    ap = argparse.ArgumentParser(description="Create frames for a movie that can be compiled using ffmpeg")
    ap.add_argument("start", help="date string as start time")
    ap.add_argument("end", help="date string as end time")
    ap.add_argument("step", type=float, help="fraction of a day to step by")
    ap.add_argument("--config", help="path to a config file", default="config.json")
    return ap.parse_args()


def make_three_color(data, time, step, config, shape=(1280, 1280), lower_val=(0, 0, 0), upper_val=(2.5, 2.5, 2.5)):
    """
    create a three color image according to the config file
    :param data: a dictionary of fetched data where keys correspond to products
    :param config: a config object
    :param shape: the size of a composite image
    :param lower_val: a tuple of lower values for RGB, any value below this is set to the low value
    :param upper_val: a tuple of upper values for RGB, any value above this is set to the high value
    :return: a (m,n,3) numpy array for a three color image where all values are between 0 and 1
    """
    order = {'red': 0, 'green': 1, 'blue': 2}

    three_color = np.zeros((shape[0], shape[1], 3))
    channel_colors = {color: config.default[color] for color in ['red', 'green', 'blue']}

    for color, channel in channel_colors.items():
        if data[channel][1] is None or \
                abs((time - date_parser.parse(data[channel][0]['date-end'])).total_seconds()) > step.total_seconds():
            return np.zeros((shape[0], shape[1], 3))

        three_color[:, :, order[color]] = data[channel][1]

        # scale the image by the power
        three_color[:, :, order[color]] = np.power(three_color[:, :, order[color]],
                                                   config.default["{}_power".format(color)])

        # adjust the percentile thresholds
        lower = lower_val[order[color]]
        upper = upper_val[order[color]]
        three_color[np.where(three_color[:, :, order[color]] < lower)] = lower
        three_color[np.where(three_color[:, :, order[color]] > upper)] = upper

    # image values must be between (0,1) so scale image
    for color, index in order.items():
        three_color[:, :, index] /= upper_val[order[color]]
    return three_color


def main():
    """
    process the main task
    """
    args = get_args()
    args.start = date_parser.parse(args.start)
    args.end = date_parser.parse(args.end)
    args.step = timedelta(args.step)
    config = Config(args.config)

    times = [args.start + i * args.step for i in range(int((args.end - args.start) / args.step))]

    for i, time in enumerate(times):
        make_plot(time, config, args.step)


def make_plot(time, config, step):
    """
    create a three color and all composite images for a given time
    NOTE: channel mins and maxes are currently hardcoded since this is a very specific script
    :param i: the index to save the file as
    :param time:
    :param config:
    :return:
    """
    fig, ax = plt.subplots()
    try:
        result = Fetcher(time, products=config.products,
                         suvi_composite_path=config.suvi_composite_path).fetch(multithread=False)
        if result:
            arr = make_three_color(result, time, step, config, upper_val=(2.4, 2.4, 2.4))
        else:
            arr = np.zeros((1280, 1280, 3))
    except ValueError:

        arr = np.zeros((1280, 1280, 3))

    ax.imshow(arr, origin='lower')
    timestr = time.strftime("%Y-%m-%d %H:%M:%S")
    fnextend = time.strftime("%Y%m%d%H%M%S")
    ax.set_title(timestr)
    ax.set_axis_off()
    fig.savefig("three_{}.png".format(fnextend), bbox_inches='tight', dpi=300)
    plt.close(fig)

    channel_min = {'suvi-l2-ci094': 0,
                   'suvi-l2-ci131': 0,
                   'suvi-l2-ci171': 0,
                   'suvi-l2-ci195': 0,
                   'suvi-l2-ci284': 0,
                   'suvi-l2-ci304': 0}
    channel_max = {'suvi-l2-ci094': 1,
                   'suvi-l2-ci131': 1,
                   'suvi-l2-ci171': 1.8,
                   'suvi-l2-ci195': 1.8,
                   'suvi-l2-ci284': 1.8,
                   'suvi-l2-ci304': 2.5}

    for channel in channel_min:
        fig, ax = plt.subplots()
        if result[channel][1] is not None \
                and abs((time - date_parser.parse(result[channel][0]['date-end'])).total_seconds()) < step.total_seconds():
            dat = np.power(result[channel][1], 0.25)
            dat[np.isnan(dat)] = 0
        else:
            dat = np.zeros((1280, 1280))
        ax.imshow(dat, vmin=channel_min[channel], vmax=channel_max[channel], cmap='gray', origin='lower')
        ax.set_title(timestr)
        ax.set_axis_off()
        fig.savefig("{}_{}.png".format(channel, fnextend), bbox_inches='tight', dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    main()
