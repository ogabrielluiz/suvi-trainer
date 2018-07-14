from datetime import datetime, timedelta
from astropy.io import fits
from suvitrainer.config import PRODUCTS, BASE_URL, SOLAR_CLASS_INDEX, DEFAULT_HEADER
import urllib.request
from bs4 import BeautifulSoup
import re, os
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
from dateutil import parser as date_parser


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
        pool = ThreadPool()
        results = pool.map(self.fetch_suvi, self.products)
        return results

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

        with fits.open("{}.fits".format(product)) as hdu:
            head = hdu[0].header
            data = hdu[0].data
        os.remove("{}.fits".format(product))

        return product, head, data

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


class Outgest:
    """ saves a thematic map in the correct format for classification """
    def __init__(self, filename, thematic_map, headers):
        self.filename = filename
        self.thmap = thematic_map
        self.ref_hdr = headers[DEFAULT_HEADER]
        self.start_time = date_parser.parse(headers[DEFAULT_HEADER]['date-obs'])
        self.end_time = date_parser.parse(headers[DEFAULT_HEADER]['date-obs'])
        for channel, header in headers.items():
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
            card = hdr_src.cards[card_ind]
            hdu.header.append((card.keyword, card.value, card.comment))

    def save(self):
        pri_hdu = fits.PrimaryHDU(data=self.thmap)

        # Temporal Information
        date_fmt = '%Y-%m-%dT%H:%M:%S.%f'
        date_beg = self.start_time.strftime(date_fmt)
        date_end = self.end_time.strftime(date_fmt)
        date_now = datetime.now().strftime(date_fmt)
        self.set_fits_header("TIMESYS", self.ref_hdr, pri_hdu)
        pri_hdu.header.append(("DATE-BEG", date_beg, "sun observation start time on sat"))
        pri_hdu.header.append(("DATE-END", date_end, "sun observation end time on sat"))
        pri_hdu.header.append(("DATE", date_now, "file generation time"))

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

        # Thematic map feature list (Secondary HDU extension)
        map_val = []
        map_label = []
        for key, value in SOLAR_CLASS_INDEX.items(): #sorted(SOLAR_CLASS_INDEX.items(), key=lambda p: (lambda k, v: (v, k))):
            map_label.append(key)
            map_val.append(value)
        c1 = fits.Column(name="Thematic Map Value", format="B", array=np.array(map_val))
        c2 = fits.Column(name="Feature Name", format="22A", array=np.array(map_label))
        bintbl_hdr = fits.Header([("XTENSION", "BINTABLE")])
        sec_hdu = fits.BinTableHDU.from_columns([c1, c2], header=bintbl_hdr)

        # Output thematic map as the primary HDU and the list of map features as an extension BinTable HDU
        hdu = fits.HDUList([pri_hdu, sec_hdu])
        hdu.writeto(self.filename, overwrite=True, checksum=True)


if __name__ == "__main__":
    f = Fetcher(datetime(2018, 6, 30, 5, 13))
    f.fetch()

