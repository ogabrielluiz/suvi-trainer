from datetime import timedelta, datetime
import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord

import sunpy.map
from sunpy.net import hek
from sunpy.time import parse_time
from sunpy.coordinates import frames

from skimage.draw import polygon

import os
from dateutil import parser as dateparser
import argparse
from suvitrainer.config import Config
from suvitrainer.fileio import Fetcher, Outgest


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dates", help="a text file listing which dates to run or a single date")
    parser.add_argument("output", help="where to put output maps")
    parser.add_argument("-v", "--verbose", action="store_true", help="prints helpful logging information during run")
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    return args


def query_hek(time, time_window=1):
    """
    requests hek responses for a given time
    :param time: datetime object
    :param time_window: how far in hours on either side of the input time to look for results
    :return: hek response list
    """
    hek_client = hek.HEKClient()
    start_time = time - timedelta(hours=time_window)
    end_time = time + timedelta(hours=time_window)
    responses = hek_client.query(hek.attrs.Time(start_time, end_time))
    return responses


def make_thmap(suvi_data, responses, config, include_human=False,
               origins=('SPoCA', 'Flare Detective - Trigger Module'),
               themes=('AR', 'CH', 'FI', 'FL')):
    """
    constructs thematic map from input information
    :param suvi_data: (header, data) for a suvi image, prefer 195
    :param responses: a list of hek responses
    :param config: a configuration object from suvitrainer
    :param include_human: if true, will automatically include all human labels
    :param origins: which systems to use in thematic map from hek
    :param themes: which hek labels to use
    :return: a (m,n) array of hek labels as thematic map image gridded to suvi_data
    """
    theme_name_map = {'AR': "bright_region", "CH": "coronal_hole", "FI": "filament", "FL": "flare"}
    suvimap = sunpy.map.Map(suvi_data[1], suvi_data[0])  # scaled version of the above for nice display
    thmap = np.zeros_like(suvi_data[1])
    responses = sorted(responses, key=lambda e: e['event_type'])
    for response in responses:
        if response['event_type'] == 'FI' or response['event_type'] == 'FL':
            print(response['event_type'], response['frm_name'])
        if response['event_type'] in themes and \
                ((response['frm_name'] in origins) or (include_human and response['frm_humanflag']=='true')):
            if response['frm_name'] == "SPoCA":
                p1 = response['hpc_boundcc'][9:-2]
            else:
                p1 = response["hpc_bbox"][9:-2]
            p2 = p1.split(',')
            p3 = [v.split(" ") for v in p2]
            date = parse_time(response['event_starttime'])

            boundary = SkyCoord(
                [(float(v[0]), float(v[1])) * u.arcsec for v in p3],
                obstime=date,
                frame=frames.Helioprojective)
            xs, ys = boundary.to_pixel(suvimap.wcs)
            xx, yy = polygon(xs, ys)
            thmap[yy, xx] = config.solar_class_index[theme_name_map[response['event_type']]]
    return thmap


def main():
    """
    fetches hek data and makes thematic maps as requested
    """
    args = get_args()
    config = Config(args.config)

    # Load dates
    if os.path.isfile(args.dates):
        with open(args.dates) as f:
            dates = [dateparser.parse(line.split(" ")[0]) for line in f.readlines()]
    else:  # assume it's a date
        dates = [dateparser.parse(args.dates)]

    if args.verbose:
        print("Dates are:")
        for date in dates:
            print(date)

    for date in dates:
        if args.verbose:
            print('Processing {}'.format(date))
        suvi_data = Fetcher(date, ['suvi-l2-ci195'],
                            suvi_composite_path=config.suvi_composite_path).fetch(multithread=False)['suvi-l2-ci195']
        if suvi_data[0] is not None:
            config.expert = 'HEK'
            responses = query_hek(date)
            thmap = make_thmap(suvi_data, responses, config)
            Outgest(os.path.join(args.output, "thmap_hek_{}.fits".format(date.strftime("%Y%m%d%H%M%S"))),
                    thmap, {"c195": suvi_data[0], "suvi-l2-ci195": suvi_data[0]}, args.config).save()


if __name__ == "__main__":
    main()
