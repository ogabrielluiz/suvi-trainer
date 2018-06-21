# from labeling gui
import matplotlib

DELIMITER = '|'

SOLAR_CLASSES = [('unlabeled', 0),
                 ('empty_outer_space', 1),
                 ('structured_outer_space', 2),
                 ('bright_region', 3),
                 ('filament', 4),
                 ('prominence', 5),
                 ('coronal_hole', 6),
                 ('quiet_sun', 7),
                 ('limb', 8),
                 ('flare', 9)]

SOLAR_CLASS_INDEX = {theme: number for theme, number in SOLAR_CLASSES}
SOLAR_CLASS_NAME = {number: theme for theme, number in SOLAR_CLASSES}
ORDERED_SOLAR_CLASSES = [SOLAR_CLASS_NAME[index] for index in range(len(SOLAR_CLASSES))]
SOLAR_CLASSES_HDRSTRING = DELIMITER.join(ORDERED_SOLAR_CLASSES)

SOLAR_COLORS = {'unlabeled': 'white',
                'empty_outer_space': '#191970',  # 'midnight blue',
                'structured_outer_space': '#8b7b8b',  # 'thistle4',
                'bright_region': 'yellow',
                'filament': '#FF3E96',  # 'VioletRed1',
                'prominence': '#FF69B4',  # 'HotPink1',
                'coronal_hole': '#0085b2',  # 'DeepSkyBlue3',
                'quiet_sun': '#006400',  # 'dark green',
                'limb': '#8FBC8F',  # 'dark sea green'}
                'flare': "#ffdb99"}  # orange

CONFIDENCE_LEVELS = [("100% confident", 7),
                     ("extremely confident", 6),
                     ("very confident", 5),
                     ("fairly confident", 4),
                     ("somewhat confident", 3),
                     ("more than fifty-fifty", 2),
                     ("fifty-fifty", 1)]

DEFAULT_HEADER = 'c171'
DEFAULT_CHANNELS = {'red': 'c171', 'blue': 'c284', 'green': 'c195', 'single': 'chalpha'}

DEFAULT_POWER = {'red': 0.26, 'blue': 0.26, 'green': 0.29, 'single': 2}
DEFAULT_VMIN = {'red': 0.0, 'blue': 0.0, 'green': 0.0, 'single': 0.0}
DEFAULT_VMAX = {'red': 4.0, 'blue': 4.0, 'green': 4.0, 'single': 4.0}

MULTICOLOR_RESOLUTION = 0.01  # the resolution of the multicolor sliders
MULTICOLOR_RANGE = [MULTICOLOR_RESOLUTION, 1]  # the range of the multicolor sliders
MULTICOLOR_VRANGE = [0, 1]
MULTICOLOR_VRESOLUTION = 0.01
BOLDFONT = 'Helvetica 18 bold'
SINGLECOLOR_RESOLUTION = MULTICOLOR_RESOLUTION
SINGLECOLOR_VRESOLUTION = MULTICOLOR_VRESOLUTION
SINGLECOLOR_RANGE = [0.25, 2.5]  # MULTICOLOR_RANGE
SINGLECOLOR_VRANGE = [0, 1]

COLORTABLE = [SOLAR_COLORS[SOLAR_CLASS_NAME[i]] for i in range(len(SOLAR_CLASSES))]
SOLAR_CMAP = matplotlib.colors.ListedColormap(COLORTABLE)

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
