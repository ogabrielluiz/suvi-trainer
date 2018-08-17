# from labeling gui
import json
import matplotlib


class Config:
    def __init__(self, config_file_path):
        """
        Handles configuring the annotation tool
        :param config_file_path: string path to configuration file
        """
        with open(config_file_path) as f:
            config = json.load(f)

        self.products = config['train']['products']
        self.products_map = dict()

        for product in self.products:
            if "aia" in product:
                self.products_map[product] = product
            elif "halpha" in product:
                self.products_map["halpha"] = "halpha"
            else:
                self.products_map[product] = "c" + str(int(product.split("-")[-1][2:]))

        if 'upload' in config['train']:
            self.upload = config['train']['upload']
            self.upload_password = config['train']['upload_password']
        else:
            self.upload = False
            self.upload_password = None

        if "suvi_path" in config['train']:
            self.suvi_composite_path = config['train']['suvi_path']
        else:
            self.suvi_composite_path = ""

        self.expert = config['train']['name']
        self.suvi_base_url = config['train']['suvi_url']

        self.solar_classes = [(c, int(n)) for c,n in config['classes'].items()]
        self.solar_class_index = {c: n for c, n in self.solar_classes}
        self.solar_class_name = {n: c for c, n in config['classes'].items()}
        self.solar_colors = config['display']['colors']
        self.color_table = [self.solar_colors[self.solar_class_name[i]] if i in self.solar_class_name else 'black'
                            for i in range(max(list(self.solar_class_name.keys()))+1)]
        self.solar_cmap = matplotlib.colors.ListedColormap(self.color_table)

        self.default = dict()
        for k,v in config['display']['default'].items():
            self.default[k] = v

        self.ranges = dict()
        for k, v in config['display']['ranges'].items():
            self.ranges[k] = v

        self.boldfont = config['display']['font']['bold']

        if 'data' in config and 'median_kernel' in config['data']:
            self.median_kernel = config['data']['median_kernel']
            if self.median_kernel == 0:
                self.median_kernel = None
        else:
            self.median_kernel = None
