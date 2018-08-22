import os
import ftplib
import re
import urllib.request
from datetime import datetime, timedelta
from multiprocessing.dummy import Pool as ThreadPool

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from bs4 import BeautifulSoup
from dateutil import parser as date_parser
from suvitrainer.config import Config
from skimage.transform import resize

from sunpy.net import vso
from astropy.units import Quantity
from skimage.transform import AffineTransform, warp

from scipy.signal import medfilt


def convert_time_string(date_str):
    """ Change a date string from the format 2018-08-15T23:55:17 into a datetime object """
    dt, _, _ = date_str.partition(".")
    dt = datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S")
    return dt


def get_dates_file(path):
    """ parse dates file of dates and probability of choosing"""
    with open(path) as f:
        dates = f.readlines()
    return [(convert_time_string(date_string.split(" ")[0]), float(date_string.split(" ")[1]))
            for date_string in dates]


def get_dates_link(url):
    """ download the dates file from the internet and parse it as a dates file"""
    urllib.request.urlretrieve(url, "temp.txt")
    dates = get_dates_file("temp.txt")
    os.remove("temp.txt")
    return dates


class Fetcher:
    """ retrieves channel images for a specific time """

    def __init__(self, date,
                 products,
                 suvi_base_url="https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l1b/",
                 suvi_composite_path="/home/marcus/grive/suvi_training/images/",
                 verbose=True):
        """
        :param date: a date object the indicates when the observation is from
        :param suvi_base_url: the url to the top level goes-16 data page
        :param suvi_composite_path: the path to local suvi composites, not required for other file types
        :param products: a list of products to fetch, e.g. suvi-l2-ci094, suvi-l1b-fe094, halpha, aia-94
        """
        self.date = date
        self.suvi_base_url = suvi_base_url
        self.suvi_composite_path = suvi_composite_path
        self.products = products
        self.verbose = verbose

    def fetch(self, multithread=True, median_kernel=5):
        """
        For all products in products, will call the correct fetch routine and download an image
        """

        def func_map(product):
            if "halpha" in product:
                result = self.fetch_halpha(median_kernel=median_kernel)
            elif "aia" in product:
                result = self.fetch_aia(product, median_kernel=median_kernel)
            elif "l1b" in product:
                result = self.fetch_suvi_l1b(product, median_kernel=median_kernel)
            elif "l2-ci" in product:
                result = self.fetch_suvi_composite(product, median_kernel=median_kernel)
            else:
                raise ValueError("{} is not a valid product.".format(product))
            return result

        if multithread:
            pool = ThreadPool()
            results = pool.map(func_map, self.products)
        else:
            results = [func_map(product) for product in self.products]

        results = {product: (head, data) for product, head, data in results}
        # for product, (_, data) in results.items():
        #     if data is None:
        #         raise ValueError("The {} channel was empty. Please try again.".format(product) +
        #                          "Sometimes a second reload or a different date is needed")
        return results

    def fetch_halpha(self, correct=True, median_kernel=5):
        """
        pull halpha from that time from Virtual Solar Observatory GONG archive
        :param correct: remove nans and negatives
        :return: tuple of "halpha", a fits header, and data object for the GONG image at that time
            header and data will be None if request failed
        """

        if self.verbose:
            print("Requesting halpha")

        def time_interval(time):
            """ get a window of three minutes around the requested time to ensure an image at GONG cadence"""
            return time - timedelta(minutes=3), time + timedelta(minutes=3)

        # setup the query for an halpha image and fetch, saving the image in the current directory
        client = vso.VSOClient()
        halpha, source = Quantity(656.28, "nm"), "gong"
        query = client.search(vso.attrs.Time(*time_interval(self.date)),
                              vso.attrs.Source(source),
                              vso.attrs.Wavelength(halpha))

        if not query: # no images found in query
            return "halpha", None, None
        else:
            query = query[0]

        result = client.fetch([query], path="./").wait(progress=False)

        # open the image and remove the file
        with fits.open(result[0]) as hdu:
            head = hdu[1].header
            data = hdu[1].data
        os.remove(result[0])

        # scale halpha to suvi
        scale = 2.35
        tform = AffineTransform(scale=(scale, scale))  # , translation=(-1280/2,-1280/2))
        data = warp(data, tform, output_shape=(1280, 1280))
        tform = AffineTransform(
            translation=(-(640 - 1024 / scale), -(640 - 1024 / scale)))  # , translation=(-1280/2,-1280/2))
        data = warp(data, tform)

        if correct:
            data[np.isnan(data)] = 0
            data[data < 0] = 0

        if median_kernel:
            data = medfilt(data, median_kernel)

        return "halpha", head, data

    def fetch_aia(self, product, correct=True, median_kernel=5):
        """
        pull halpha from that time from Virtual Solar Observatory GONG archive
        :param product: aia-[REQUESTED CHANNEL IN ANGSTROMS], e.g. aia-131 gets the 131 angstrom image
        :param correct: remove nans and negatives
        :return: tuple of product name, fits header, and data object
            the header and data object will be None if the request failed
        """

        if self.verbose:
            print("Requesting {}".format(product))

        wavelength = product.split("-")[1]

        def time_interval(time):
            """ get a window of three minutes around the requested time to ensure an image at GONG cadence"""
            return time - timedelta(minutes=15), time + timedelta(minutes=15)

        # setup the query for an halpha image and fetch, saving the image in the current directory
        client = vso.VSOClient()
        wave, source = Quantity(wavelength, "angstrom"), "aia"
        query = client.search(vso.attrs.Time(*time_interval(self.date)),
                              vso.attrs.Instrument(source),
                              vso.attrs.Wavelength(wave))
        if self.verbose:
            print("Query length for {} is {}".format(product, len(query)))

        if not query: # failed to get a result
            return product, None, None
        else:
            query = query[0]

        result = client.fetch([query], path="./").wait(progress=False)

        # open the image and remove the file
        with fits.open(result[0]) as hdu:
            hdu.verify('fix')
            head = hdu[1].header
            data = hdu[1].data
        os.remove(result[0])

        data, head = self.align_solar_fov(head, data, 2.5, 1280, rotate=False)
        data = resize(data, (1280, 1280))
        head['NAXIS1'] = 1280
        head['NAXIS2'] = 1280
        head['CRPIX1'] = 640
        head['CRPIX2'] = 640
        head['CDELT1'] = 2.5
        head['CDELT2'] = 2.5

        if correct:
            data[np.isnan(data)] = 0
            data[data < 0] = 0

        if median_kernel:
            data = medfilt(data, median_kernel)

        return product, head, data

    def fetch_suvi_l1b(self, product, correct=True, median_kernel=5):
        """
        Given a product keyword, downloads the SUVI l1b image into the current directory.
        NOTE: the suvi_l1b_url must be properly set for the Fetcher object
        :param product: the keyword for the product, e.g. suvi-l1b-fe094
        :param correct: remove nans and negatives
        :return: tuple of product name, fits header, and data object
            the header and data object will be None if the request failed
        """
        if self.date < datetime(2018, 5, 23) and not (self.date >= datetime(2017, 9, 6) \
                and self.date <= datetime(2017, 9, 10, 23, 59)):
            print("SUVI data is only available after 2018-5-23")
            return product, None, None

        url = self.suvi_base_url + product + "/{}/{:02d}/{:02d}".format(self.date.year, self.date.month, self.date.day)
        if self.verbose:
            print("Requesting from {}".format(url))

        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            try:
                page = response.read()
            except ConnectionError:
                print("Unable to access website. Wait a few moments and try again.")

        soup = BeautifulSoup(page, 'html.parser')
        links = [link['href'] for link in soup.find_all('a', href=True)]
        links = [link for link in links if "SUVI" in link]
        meta = [self.parse_filename_meta(fn) for fn in links if ".fits" in fn]
        links = sorted(meta, key=lambda m: np.abs((m[2] - self.date).total_seconds()))[:10]
        links = [fn for fn, _, _, _, _ in links]

        i = 0

        def download_and_check(i):
            urllib.request.urlretrieve(url + "/" + links[i], "{}.fits".format(product))
            with fits.open("{}.fits".format(product)) as hdu:
                head = hdu[0].header
            return head['exptime'] > 0.5

        while not download_and_check(i):
            i += 1

        with fits.open("{}.fits".format(product)) as hdu:
            head = hdu[0].header
            data = hdu[0].data
        os.remove("{}.fits".format(product))

        if correct:
            data[np.isnan(data)] = 0
            data[data < 0] = 0

        if median_kernel:
            data = medfilt(data, median_kernel)

        data, head = self.align_solar_fov(head, data, 2.5, 2.0, rotate=True, scale=False)
        if self.verbose:
            print(product, " is using ", head['date-obs'])
        return product, head, data

    def fetch_suvi_composite(self, product, correct=True, median_kernel=5):
        """
        Fetches a suvi composite from a local directory
        NOTE: the suvi_composite_path must be properly set for this methd
        :param product: the requested product, e.g. suvi-l2-c094
        :param correct: remove nans and negatives
        :return: tuple of product name, fits header, and data object
            the header and data object will be None if the request failed
        """
        path = os.path.join(self.suvi_composite_path, product,
                            "{:4d}/{:02d}/{:02d}/".format(self.date.year, self.date.month, self.date.day))

        if not os.path.isdir(path):
            return product, None, None
        else:  # exists!
            # find the composite with the closest time code
            candidate_fns = [fn for fn in os.listdir(path) if ".fits" in fn
                             and os.path.getsize(os.path.join(path, fn)) > 0]
            candidate_fns = sorted([self.parse_filename_meta(fn) for fn in candidate_fns],
                                   key=lambda entry: abs((self.date - entry[2]).total_seconds()))
            candidate_fns = [entry[0] for entry in candidate_fns]
            with fits.open(os.path.join(path, candidate_fns[0])) as hdus:
                # if the file has the expected number of headers, use it!
                if len(hdus) == 2 and 'empty' in hdus[1].header and not hdus[1].header['empty']:
                    head, data = hdus[1].header, hdus[1].data
                    if correct:
                        data[np.isnan(data)] = 0
                        data[data < 0] = 0

                    if median_kernel:
                        data = medfilt(data, median_kernel)
                    return product, head, data
                else:
                    return product, None, None

    @staticmethod
    def parse_filename_meta(filename):
        """
        taken from suvi code by vhsu
        Parse the metadata from a product filename, either L1b or l2.
           - file start
           - file end
           - platform
           - product

        :param filename: string filename of product
        :return: (start datetime, end datetime, platform)
        """
        common_pattern = "_%s_%s" % (
            "(?P<product>[a-zA-Z]{3}[a-zA-Z]?-[a-zA-Z0-9]{2}[a-zA-Z0-9]?-[a-zA-Z0-9]{4}[a-zA-Z0-9]?)",
            # product l1b, or l2
            "(?P<platform>[gG][1-9]{2})"  # patform, like g16
        )
        patterns = {  # all patterns must have the common componennt
            "l2_pattern": re.compile("%s_s(?P<start>[0-9]{8}T[0-9]{6})Z_e(?P<end>[0-9]{8}T[0-9]{6})Z" % common_pattern),
            "l1b_pattern": re.compile('%s_s(?P<start>[0-9]{14})_e(?P<end>[0-9]{14})' % common_pattern),
            "dayfile_pattern": re.compile("%s_d(?P<start>[0-9]{8})" % common_pattern),
            "monthfile_pattern": re.compile("%s_m(?P<start>[0-9]{6})" % common_pattern),
            "yearfile_pattern": re.compile("%s_y(?P<start>[0-9]{4})" % common_pattern),
        }
        match, dt_start, dt_end = None, None, None
        for pat_type, pat in patterns.items():
            match = pat.search(filename)
            if match is not None:
                if pat_type == "l2_pattern":
                    # parse l2
                    dt_start = datetime.strptime(match.group("start"), '%Y%m%dT%H%M%S')
                    dt_end = datetime.strptime(match.group("end"), '%Y%m%dT%H%M%S')
                elif pat_type == "l1b_pattern":
                    # parse l1b
                    dt_start = datetime.strptime(match.group("start"), '%Y%j%H%M%S%f')
                    dt_end = datetime.strptime(match.group("end"), '%Y%j%H%M%S%f')
                elif pat_type == "dayfile_pattern":
                    dt_start = datetime.strptime(match.group("start"), "%Y%m%d")
                    dt_end = dt_start + timedelta(hours=24)
                elif pat_type == "monthfile_pattern":
                    dt_start = datetime.strptime(match.group("start"), "%Y%m")
                    dt_end = datetime(dt_start.year, dt_start.month + 1,
                                      1)  # will raise exception in December, fix when needed
                elif pat_type == "yearfile_pattern":
                    dt_start = datetime.strptime(match.group("start"), "%Y")
                    dt_end = datetime(dt_start.year + 1, 1, 1)
                break

        if match is None:
            if "NCEI" in filename and ".fits" in filename:
                dt_start = datetime.strptime("T".join(filename.split("_")[4:6]), "%Y%m%dT%H%M%S")
                dt_end = dt_start
                angstroms = int(filename.split("_")[2])
                atom = "Fe" if angstroms != 304 else "He"
                product = "SUVI-L1b-{}{}".format(atom, angstroms)
                return filename, dt_start, dt_end, "g16", product
            else:
                # we didn't find any matching patterns...
                raise ValueError("Timestamps not detected in filename: %s" % filename)
        else:
            return filename, dt_start, dt_end, match.group("platform"), match.group("product")

    @staticmethod
    def align_solar_fov(header, data, cdelt_min, naxis_min,
                        translate_origin=True, rotate=True, scale=True):
        """
        taken from suvi code by vhsu
        Apply field of view image corrections

        :param header: FITS header
        :param data: Image data
        :param cdelt_min: Minimum plate scale for images (static run config param)
        :param naxis_min: Minimum axis dimension for images (static run config param)
        :param translate_origin: Translate image to specified origin (dtype=bool)
        :param rotate: Rotate image about origin (dtype=bool)
        :param scale: Scale image (dtype=bool)

        :rtype: numpy.ndarray
        :return: data_corr (corrected/aligned image)
        :rtype: astropy.io.fits.header Header instance
        :return: upd_meta (updated metadata after image corrections)

        NOTES: (1) The associative property of matrix multiplication makes it possible to multiply
                   transformation matrices together to produce a single transformation. However, the order
                   of each transformation matters. In this algorithm, the order is:
                   1. Translate image center to origin (required)
                   2. Translate image solar disk center to origin
                   3. Rotate image about the solar disk center to align with solar spin axis
                   4. Scale the image so that each pixel is square
                   5. Translate the image to the image center (required)
               (2) In python, the image transformations are about the axis origin (0, 0). Therefore, the image point
                   to rotate about should be shifted to (0, 0) before the rotation.
               (3) Axis 1 refers to the physical x-axis and axis 2 refers to the physical y-axis, e.g. CRPIX1 is
                   the center pixel value wrt the x-axis and CRPIX2 is wrt the y-axis.
        """
        from skimage.transform import ProjectiveTransform

        # Start with 3x3 identity matrix and original header metadata (no corrections)
        t_matrix = np.identity(3)
        upd_meta = header

        # (1) Translate the image center to the origin (required transformation)

        # Read in required keywords from header
        try:
            img_dim = (header["NAXIS1"], header["NAXIS2"])
        except KeyError:
            return None, None
        else:
            # Transformation matrix
            t_matrix = np.matmul(np.array([[1., 0., -(img_dim[0] + 1) / 2.],
                                           [0., 1., -(img_dim[1] + 1) / 2.],
                                           [0., 0., 1.]]), t_matrix)

        # (2) Translate image solar disk center to origin
        if translate_origin:
            # Read in required keywords from header
            try:
                sun_origin = (header["CRPIX1"], header["CRPIX2"])
            except KeyError:
                return None, None
            else:
                # Transformation matrix
                t_matrix = np.matmul(np.array([[1., 0., -sun_origin[0] + (img_dim[0] + 1) / 2.],
                                               [0., 1., -sun_origin[1] + (img_dim[1] + 1) / 2.],
                                               [0., 0., 1.]]), t_matrix)

                # Update metadata: CRPIX1 and CRPIX2 are at the center of the image
                upd_meta["CRPIX1"] = (img_dim[0] + 1) / 2.
                upd_meta["CRPIX2"] = (img_dim[1] + 1) / 2.

        # (3) Rotate image to align with solar spin axis
        if rotate:
            # Read in required keywords from header
            try:
                PC1_1 = header['PC1_1']
                PC1_2 = header['PC1_2']
                PC2_1 = header['PC2_1']
                PC2_2 = header['PC2_2']
            except KeyError:
                try:
                    CROTA = header['CROTA'] * (np.pi / 180.)  # [rad]
                    plt_scale = (header["CDELT1"], header["CDELT2"])
                except KeyError:
                    return None, None
                else:
                    t_matrix = np.matmul(np.array([[np.cos(CROTA), -np.sin(CROTA) * (plt_scale[1] / plt_scale[0]), 0.],
                                                   [np.sin(CROTA) * (plt_scale[0] / plt_scale[1]), np.cos(CROTA), 0.],
                                                   [0., 0., 1.]]), t_matrix)

                    # Update metadata: CROTA is zero and PCi_j matrix is the identity matrix
                    upd_meta["CROTA"] = 0.
                    upd_meta["PC1_1"] = 1.
                    upd_meta["PC1_2"] = 0.
                    upd_meta["PC2_1"] = 0.
                    upd_meta["PC2_2"] = 1.
            else:
                t_matrix = np.matmul(np.array([[PC1_1, PC1_2, 0.],
                                               [PC2_1, PC2_2, 0.],
                                               [0., 0., 1.]]), t_matrix)

                # Update metadata: CROTA is zero and PCi_j matrix is the identity matrix
                upd_meta["CROTA"] = 0.
                upd_meta["PC1_1"] = 1.
                upd_meta["PC1_2"] = 0.
                upd_meta["PC2_1"] = 0.
                upd_meta["PC2_2"] = 1.

        # (4) Scale the image so that each pixel is square
        if scale:
            # Read in required keywords from header
            try:
                plt_scale = (header["CDELT1"], header["CDELT2"])
            except KeyError:
                return None, None
            else:
                # Product of minimum plate scale and axis dimension
                min_scl = cdelt_min * naxis_min

                # Determine smallest axis
                naxis_ref = min(img_dim)

                # Transformation matrix
                t_matrix = np.matmul(np.array([[(plt_scale[0] * naxis_ref) / min_scl, 0., 0.],
                                               [0., (plt_scale[1] * naxis_ref) / min_scl, 0.],
                                               [0., 0., 1.]]), t_matrix)

                # Update the metadata: CDELT1 and CDELT2 are scaled by factor to make each pixel square
                upd_meta["CDELT1"] = plt_scale[0] / ((plt_scale[0] * naxis_ref) / min_scl)
                upd_meta["CDELT2"] = plt_scale[1] / ((plt_scale[1] * naxis_ref) / min_scl)

        # (5) Translate the image to the image center (required transformation)
        t_matrix = np.matmul(np.array([[1., 0., (img_dim[0] + 1) / 2.],
                                       [0., 1., (img_dim[1] + 1) / 2.],
                                       [0., 0., 1.]]), t_matrix)

        # Transform the image with all specified operations
        # NOTE: The inverse transformation needs to be applied because the transformation matrix
        # describes operations on the pixel coordinate frame instead of the image itself. The inverse
        # transformation will perform the intended operations on the image. Also, any values outside of
        # the image boundaries are set to zero.
        data_corr = warp(data, ProjectiveTransform(matrix=t_matrix).inverse, cval=0., preserve_range=True)

        # Check if NaNs are generated from transformation
        try:
            assert not np.any(np.isnan(data_corr))
        except AssertionError:
            pass

        return data_corr, upd_meta


class PNGMaker:
    """ Creates helpful pngs to view data """

    def __init__(self, thematic_map_hdu, config_path):
        """
        :param thematic_map_hdu: a list of hdus for a thematic map, 0 is the map data, 1 is the theme labels
        :param config_path: the path to a configuration object
        """
        self.config = Config(config_path)
        self.thmap = thematic_map_hdu[0].data
        self.date = date_parser.parse(thematic_map_hdu[0].header['date-end'])

    def make_thematic_png(self, outpath=None):
        """
        Convert a thematic map into png format with a legend
        :param outpath: if specified, will save the image instead of showing it
        """
        from matplotlib.patches import Patch

        fig, previewax = plt.subplots()
        shape = self.thmap.shape
        previewax.imshow(self.thmap,
                         origin='lower',
                         interpolation='nearest',
                         cmap=self.config.solar_cmap,
                         vmin=-1, vmax=len(self.config.solar_classes)-1)

        legend_elements = [Patch(facecolor=c, label=sc, edgecolor='k')
                           for sc, c in self.config.solar_colors.items()]
        previewax.legend(handles=legend_elements, fontsize='x-small',
                         bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                         ncol=2, mode="expand", borderaxespad=0.)
        previewax.set_xlim([0, shape[0]])
        previewax.set_ylim([0, shape[0]])
        previewax.set_aspect("equal")
        previewax.set_axis_off()

        if outpath:
            fig.savefig(outpath, dpi=300,
                        transparent=True,
                        bbox_inches='tight',
                        pad_inches=0.)
            plt.close()
        else:
            plt.show()

    def make_three_color(self, upper_percentile=100, lower_percentile=0):
        """
        Load the configured input channel images and create a three color image
        :param upper_percentile: pixels above this percentile are suppressed
        :param lower_percentile: pixels below this percentile are suppressed
        :return: a numpy array (m,n,3) representing a three-color image
        """
        order = {'red': 0, 'green': 1, 'blue': 2}
        shape = self.thmap.shape
        three_color = np.zeros((shape[0], shape[1], 3))
        channel_colors = {color: self.config.default[color] for color in ['red', 'green', 'blue']}

        data = Fetcher(self.date, products=list(channel_colors.values()), verbose=False).fetch()
        for color, channel in channel_colors.items():
            three_color[:, :, order[color]] = data[channel][1]

            # scale the image by the power
            three_color[:, :, order[color]] = np.power(three_color[:, :, order[color]],
                                                       self.config.default["{}_power".format(color)])

            # adjust the percentile thresholds
            lower = np.nanpercentile(three_color[:, :, order[color]], lower_percentile)
            upper = np.nanpercentile(three_color[:, :, order[color]], upper_percentile)
            three_color[np.where(three_color[:, :, order[color]] < lower)] = lower
            three_color[np.where(three_color[:, :, order[color]] > upper)] = upper

        # image values must be between (0,1) so scale image
        for color, index in order.items():
            three_color[:, :, index] /= np.nanmax(three_color[:, :, index])
        return three_color

    def make_comparison_png(self, outpath=None, include_legend=False):
        """
        Creates a thematic map image with a three color beside it
        :param outpath:  if specified, will save the image instead of showing it
        :param include_legend: if true will include the thamatic map label legend
        """
        from matplotlib.patches import Patch

        fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True)

        three_color = self.make_three_color()
        axs[0].imshow(three_color)
        axs[0].set_axis_off()

        shape = self.thmap.shape
        axs[1].imshow(self.thmap,
                     origin='lower',
                     interpolation='nearest',
                     cmap=self.config.solar_cmap,
                     vmin=-1, vmax=len(self.config.solar_classes)-1)

        if include_legend:
            legend_elements = [Patch(facecolor=c, label=sc, edgecolor='k')
                               for sc, c in self.config.solar_colors.items()]
            axs[1].legend(handles=legend_elements, fontsize='x-small',
                         bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                         ncol=2, mode="expand", borderaxespad=0.)
        axs[1].set_xlim([0, shape[0]])
        axs[1].set_ylim([0, shape[0]])
        axs[1].set_aspect("equal")
        axs[1].set_axis_off()

        if outpath:
            fig.savefig(outpath, dpi=300,
                        transparent=True,
                        bbox_inches='tight',
                        pad_inches=0.)
            plt.close()
        else:
            plt.show()


class Outgest:
    """ saves a thematic map in the correct format for classification """
    def __init__(self, filename, thematic_map, headers, config_path):
        self.config = Config(config_path)
        self.filename = filename
        self.thmap = thematic_map
        self.ref_hdr = headers[self.config.products_map[self.config.default['header']]]
        self.start_time = date_parser.parse(self.ref_hdr['date-obs'])
        self.end_time = date_parser.parse(self.ref_hdr['date-obs'])
        for _, header in headers.items():
            date = date_parser.parse(header['date-obs'])
            if date < self.start_time:
                self.start_time = date
            if date > self.end_time:
                self.end_time = date

    @staticmethod
    def set_fits_header(hdr_key, hdr_src, hdu):
        try:
            card_ind = hdr_src.index(hdr_key)
        except ValueError:
           print('Invalid FITS header keyword: %s --> omitting from file' % hdr_key)
        else:
            try:
                card = hdr_src.cards[card_ind]
                hdu.header.append((card.keyword, card.value, card.comment))
            except:
                print("{} not saved".format(hdr_key))

    def upload(self):
        if self.config.upload:
            try:
                session = ftplib.FTP('ftp.jmbhughes.com', 'trainer@jmbhughes.com', self.config.upload_password)
                f = open(self.filename, 'rb')
                session.storbinary('STOR ' + self.filename, f)  # send the file
                f.close()  # close file and FTP
                session.quit()
                success = True
            except:
                success = False
            if success:
                os.remove(self.filename)

    def save(self):
        """ modified from suvi code by vhsu """
        pri_hdu = fits.PrimaryHDU(data=self.thmap)

        # Temporal Information
        date_fmt = '%Y-%m-%dT%H:%M:%S.%f'
        date_beg = self.start_time.strftime(date_fmt)
        date_end = self.end_time.strftime(date_fmt)
        date_now = datetime.utcnow().strftime(date_fmt)
        self.set_fits_header("TIMESYS", self.ref_hdr, pri_hdu)
        pri_hdu.header.append(("DATE-BEG", date_beg, "sun observation start time on sat"))
        pri_hdu.header.append(("DATE-END", date_end, "sun observation end time on sat"))
        pri_hdu.header.append(("DATE", date_now, "file generation time"))
        pri_hdu.header.append(("EXPERT", self.config.expert, "person who labeled image"))
        pri_hdu.header.append(("DATE-LAB", date_now, "date of labeling for the image"))

        # Instrument & Spacecraft State during Observation
        pri_hdu.header.append(("EXPTIME", 1., "[s] effective imaging exposure time"))
        self.set_fits_header("YAW_FLIP", self.ref_hdr, pri_hdu)
        self.set_fits_header("ECLIPSE", self.ref_hdr, pri_hdu)

        # Pointing & Projection
        self.set_fits_header("WCSNAME", self.ref_hdr, pri_hdu)
        self.set_fits_header("CTYPE1", self.ref_hdr, pri_hdu)
        self.set_fits_header("CTYPE2", self.ref_hdr, pri_hdu)
        self.set_fits_header("CUNIT1", self.ref_hdr, pri_hdu)
        self.set_fits_header("CUNIT2", self.ref_hdr, pri_hdu)
        self.set_fits_header("PC1_1", self.ref_hdr, pri_hdu)
        self.set_fits_header("PC1_2", self.ref_hdr, pri_hdu)
        self.set_fits_header("PC2_1", self.ref_hdr, pri_hdu)
        self.set_fits_header("PC2_2", self.ref_hdr, pri_hdu)
        self.set_fits_header("CDELT1", self.ref_hdr, pri_hdu)
        self.set_fits_header("CDELT2", self.ref_hdr, pri_hdu)
        self.set_fits_header("CRVAL1", self.ref_hdr, pri_hdu)
        self.set_fits_header("CRVAL2", self.ref_hdr, pri_hdu)
        self.set_fits_header("CRPIX1", self.ref_hdr, pri_hdu)
        self.set_fits_header("CRPIX2", self.ref_hdr, pri_hdu)
        self.set_fits_header("DIAM_SUN", self.ref_hdr, pri_hdu)
        self.set_fits_header("LONPOLE", self.ref_hdr, pri_hdu)
        self.set_fits_header("CROTA", self.ref_hdr, pri_hdu)
        self.set_fits_header("SOLAR_B0", self.ref_hdr, pri_hdu)

        # File Provenance
        pri_hdu.header.append(("TITLE", "Expert Labeled Thematic Map Image", "image title"))
        pri_hdu.header.append(("MAP_MTHD", "human", "thematic map classifier method"))
        try:
            # Add COMMENT cards
            pri_hdu.header.insert(pri_hdu.header.index("TITLE") + 1,
                                  ("COMMENT", '------------------------------------------------------------------------'))
            pri_hdu.header.insert(pri_hdu.header.index("TITLE") + 2, ("COMMENT", 'USING SUVI THEMATIC MAP FILES'))
            pri_hdu.header.insert(pri_hdu.header.index("TITLE") + 3,
                                  ("COMMENT", '------------------------------------------------------------------------'))
            pri_hdu.header.insert(pri_hdu.header.index("TITLE") + 4,
                                  ("COMMENT", 'Map labels are described in the FITS extension.'))
            pri_hdu.header.insert(pri_hdu.header.index("TITLE") + 5, ("COMMENT", 'Example:'))
            pri_hdu.header.insert(pri_hdu.header.index("TITLE") + 6, ("COMMENT", 'from astropy.io import fits as pyfits'))
            pri_hdu.header.insert(pri_hdu.header.index("TITLE") + 7, ("COMMENT", 'img = pyfits.open(<filename>)'))
            pri_hdu.header.insert(pri_hdu.header.index("TITLE") + 8, ("COMMENT", 'map_labels = img[1].data'))
            pri_hdu.header.insert(pri_hdu.header.index("TITLE") + 9,
                                  ("COMMENT", '------------------------------------------------------------------------'))
            pri_hdu.header.insert(pri_hdu.header.index("TITLE") + 10, ("COMMENT", 'TEMPORAL INFORMATION'))
            pri_hdu.header.insert(pri_hdu.header.index("TITLE") + 11,
                                  ("COMMENT", '------------------------------------------------------------------------'))
            pri_hdu.header.insert(pri_hdu.header.index("DATE") + 1,
                                  ("COMMENT", '------------------------------------------------------------------------'))
            pri_hdu.header.insert(pri_hdu.header.index("DATE") + 2,
                                  ("COMMENT", 'INSTRUMENT & SPACECRAFT STATE DURING OBSERVATION'))
            pri_hdu.header.insert(pri_hdu.header.index("DATE") + 3,
                                  ("COMMENT", '------------------------------------------------------------------------'))
            pri_hdu.header.insert(pri_hdu.header.index("ECLIPSE") + 1,
                                  ("COMMENT", '------------------------------------------------------------------------'))
            pri_hdu.header.insert(pri_hdu.header.index("ECLIPSE") + 2, ("COMMENT", 'POINTING & PROJECTION'))
            pri_hdu.header.insert(pri_hdu.header.index("ECLIPSE") + 3,
                                  ("COMMENT", '------------------------------------------------------------------------'))
            pri_hdu.header.insert(pri_hdu.header.index("SOLAR_B0") + 1,
                                  ("COMMENT", '------------------------------------------------------------------------'))
            pri_hdu.header.insert(pri_hdu.header.index("SOLAR_B0") + 2, ("COMMENT", 'FILE PROVENANCE'))
            pri_hdu.header.insert(pri_hdu.header.index("SOLAR_B0") + 3,
                                  ("COMMENT", '------------------------------------------------------------------------'))
        except:
            print("This thematic map may be degraded and missing many keywords.")

        # Thematic map feature list (Secondary HDU extension)
        map_val = []
        map_label = []
        for key, value in self.config.solar_class_index.items(): #sorted(SOLAR_CLASS_INDEX.items(), key=lambda p: (lambda k, v: (v, k))):
            map_label.append(key)
            map_val.append(value)
        c1 = fits.Column(name="Thematic Map Value", format="B", array=np.array(map_val))
        c2 = fits.Column(name="Feature Name", format="22A", array=np.array(map_label))
        bintbl_hdr = fits.Header([("XTENSION", "BINTABLE")])
        sec_hdu = fits.BinTableHDU.from_columns([c1, c2], header=bintbl_hdr)

        # Output thematic map as the primary HDU and the list of map features as an extension BinTable HDU
        hdu = fits.HDUList([pri_hdu, sec_hdu])
        hdu.writeto(self.filename, overwrite=True, checksum=True)
