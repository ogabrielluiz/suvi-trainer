# from labeling gui
import json
import matplotlib


class Config:
    def __init__(self, config_file_path):
        with open(config_file_path) as f:
            config = json.load(f)

        self.products = config['train']['products']
        self.products_map = {p: "c" + str(int(p.split("-")[-1][2:])) if p!= "halpha" else "halpha" for p in self.products}

        if 'upload' in config['train']:
            self.upload = config['train']['upload']
            self.upload_password = config['train']['upload_password']
        else:
            self.upload = False
            self.upload_password = None

        self.expert = config['train']['name']
        self.suvi_base_url = config['train']['suvi_url']

        self.solar_classes = [(c, int(n)) for c,n in config['classes'].items()]
        print(self.solar_classes)
        self.solar_class_index = {c: n for c, n in self.solar_classes}
        self.solar_class_name = {n: c for c, n in config['classes'].items()}
        self.solar_colors = config['display']['colors']
        self.color_table = [self.solar_colors[self.solar_class_name[i]] if i in self.solar_class_name else 'black'
                            for i in range(max(list(self.solar_class_name.keys())))]
        self.solar_cmap = matplotlib.colors.ListedColormap(self.color_table)

        self.default = dict()
        for k,v in config['display']['default'].items():
            self.default[k] = v

        self.ranges = dict()
        for k, v in config['display']['ranges'].items():
            self.ranges[k] = v

        self.boldfont = config['display']['font']['bold']


# DELIMITER = '|'
#
# TRAINER = "Marcus Hughes"
#
# BASE_URL = "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l1b/"
#
# PRODUCTS = ['suvi-l1b-fe094', 'suvi-l1b-fe131', 'suvi-l1b-fe171',
#             'suvi-l1b-fe195', 'suvi-l1b-fe284', 'suvi-l1b-he304', 'halpha']
#
# PRODUCTS_MAP = {p: "c" + str(int(p.split("-")[-1][2:])) for p in PRODUCTS if p is not 'halpha'}
# PRODUCTS_MAP['halpha'] = 'chalpha'
#
# SOLAR_CLASSES = [('unlabeled', 0),
#                  ('empty_outer_space', 1),
#                  ('structured_outer_space', 2),
#                  ('bright_region', 3),
#                  ('filament', 4),
#                  ('prominence', 5),
#                  ('coronal_hole', 6),
#                  ('quiet_sun', 7),
#                  ('limb', 8),
#                  ('flare', 9)]
#
# SOLAR_CLASS_INDEX = {theme: number for theme, number in SOLAR_CLASSES}
# SOLAR_CLASS_NAME = {number: theme for theme, number in SOLAR_CLASSES}
# ORDERED_SOLAR_CLASSES = [SOLAR_CLASS_NAME[index] for index in range(len(SOLAR_CLASSES))]
# SOLAR_CLASSES_HDRSTRING = DELIMITER.join(ORDERED_SOLAR_CLASSES)
#
# SOLAR_COLORS = {'unlabeled': 'white',
#                 'empty_outer_space': '#191970',  # 'midnight blue',
#                 'structured_outer_space': '#8b7b8b',  # 'thistle4',
#                 'bright_region': 'yellow',
#                 'filament': '#FF3E96',  # 'VioletRed1',
#                 'prominence': '#FF69B4',  # 'HotPink1',
#                 'coronal_hole': '#0085b2',  # 'DeepSkyBlue3',
#                 'quiet_sun': '#006400',  # 'dark green',
#                 'limb': '#8FBC8F',  # 'dark sea green'}
#                 'flare': "#ffdb99"}  # orange
#
# CONFIDENCE_LEVELS = [("100% confident", 7),
#                      ("extremely confident", 6),
#                      ("very confident", 5),
#                      ("fairly confident", 4),
#                      ("somewhat confident", 3),
#                      ("more than fifty-fifty", 2),
#                      ("fifty-fifty", 1)]
#
# DEFAULT_HEADER = 'c195'
# DEFAULT_CHANNELS = {'red': 'c171', 'blue': 'c284', 'green': 'c195', 'single': 'chalpha'}
#
# DEFAULT_POWER = {'red': 0.26, 'blue': 0.26, 'green': 0.29, 'single': 2}
# DEFAULT_VMIN = {'red': 0.0, 'blue': 0.0, 'green': 0.0, 'single': 0.0}
# DEFAULT_VMAX = {'red': 4.0, 'blue': 4.0, 'green': 4.0, 'single': 4.0}
#
# MULTICOLOR_RESOLUTION = 0.01  # the resolution of the multicolor sliders
# MULTICOLOR_RANGE = [MULTICOLOR_RESOLUTION, 1]  # the range of the multicolor sliders
# MULTICOLOR_VRANGE = [0, 1]
# MULTICOLOR_VRESOLUTION = 0.01
# BOLDFONT = 'Helvetica 18 bold'
# SINGLECOLOR_RESOLUTION = MULTICOLOR_RESOLUTION
# SINGLECOLOR_VRESOLUTION = MULTICOLOR_VRESOLUTION
# SINGLECOLOR_RANGE = [0.25, 2.5]  # MULTICOLOR_RANGE
# SINGLECOLOR_VRANGE = [0, 1]
#
# COLORTABLE = [SOLAR_COLORS[SOLAR_CLASS_NAME[i]] for i in range(len(SOLAR_CLASSES))]
# SOLAR_CMAP = matplotlib.colors.ListedColormap(COLORTABLE)

# import configparser, argparse

# class Configuration:
#     def __init__(self, path):
#         self.path = path
#         reader = configparser.ConfigParser()
#         self.config = reader.read(self.path)
#         print(self.config.get('basic'))
#         self.get_classes()

#     def get_classes(self):
#         self.classes = self.config['basic']['classes'].split("|")
#         self.colors = self.config['basic']['colors'].split("|")
#         self.SOLAR_CLASSES = [(label, index) for index, label in enumerate(self.classes)]
#         self.SOLAR_COLORS = {label:color for label, color in zip(self.classes, self.colors)}


# if __name__ == "__main__":
#     ap = argparse.ArgumentParser()
#     ap.add_argument("path", help="path to configuration file")
#     args = vars(ap.parse_args())
#     config = Configuration(args['path'])
#     print(self.SOLAR_CLASSES)
