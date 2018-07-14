from datetime import datetime, timedelta
from astropy.io import fits
from suvitrainer.config import PRODUCTS, BASE_URL
import urllib.request
from bs4 import BeautifulSoup
import re
import numpy as np


class Fetcher:
    """ retrieves channel images for a specific time """

    def __init__(self, date,
                 products=PRODUCTS,
                 suvi_base_url=BASE_URL):
        """
        :param date: a date object the indicates when the observation is from
        :param suvi_base_url: the url to the top level goes-16 data page
        :param products: a list of products to fetch
        """
        self.date = date
        self.suvi_base_url = suvi_base_url
        self.products = products

    def fetch(self):
        """
        For all products in products, will call the correct fetch routine and download an image
        """
        for product in self.products:
            self.fetch_suvi(product)

    def fetch_suvi(self, product):
        """
        Given a product keyword, downloads the image into the current directory.
        :param product: the keyword for the product, e.g. suvi-l1b-fe094
        """
        url = self.suvi_base_url + product + "/{}/{:02d}/{:02d}".format(self.date.year, self.date.month, self.date.day)
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            page = response.read()

        soup = BeautifulSoup(page, 'html.parser')
        links = [link['href'] for link in soup.find_all('a', href=True)]
        links = [link for link in links if "SUVI" in link]
        meta = [self.parse_filename_meta(fn) for fn in links]
        links = sorted(meta, key=lambda m: np.abs((m[2] - self.date).total_seconds()))[:10]
        links = [fn for fn, start, end, platform, product in meta]

        i = 0

        def download_and_check(i):
            urllib.request.urlretrieve(url + "/" + links[i], "{}.fits".format(product))
            with fits.open("{}.fits".format(product)) as hdu:
                head = hdu[0].header
            return head['exptime'] > 0.5

        while not download_and_check(i):
            i += 1

    @staticmethod
    def parse_filename_meta(filename):
        """
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
            # we didn't find any matching patterns...
            raise ValueError("Timestamps not detected in filename: %s" % filename)
        return filename, dt_start, dt_end, match.group("platform"), match.group("product")


if __name__ == "__main__":
    f = Fetcher(datetime(2018, 6, 30, 5, 13))
    f.fetch()

