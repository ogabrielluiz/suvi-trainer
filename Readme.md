# suvi-trainer

The [Solar Ultraviolet Imager](https://www.goes-r.gov/spacesegment/suvi.html) (SUVI) aboard the National Oceanic 
and Atmospheric Administration's Geostationary Operational Environmental R-Series satellites is used 
in machine learning applications to create thematic maps, images showing where different features are on the Sun. This
tool allows domain experts to load images, manipulate them, and create labeled maps, which are used in [training solar 
image classifiers](https://github.com/jmbhughes/smachy). 
 
## Getting Started

These instructions will get you a copy of the project up and running on your local machine 
for development and testing purposes. At the moment, it is advised to run this code in a fresh virtual environment with
Python 3.5. Installation should only require running the `setup.py` script. 
 
### Installing
After [creating a separate virtual environment](https://realpython.com/python-virtual-environments-a-primer/), installation should be simply:
```
python3 setup.py install
```
(There is also a stable version that can be installed from PyPi using `pip3 install suvitrainer`)
You will need to edit the [configuration file](config_example.json) to include your name and the upload password. Please 
[contact me](mailto:hughes.jmb@gmail.com) for the password and further information. 

*Please note that the [extra scripts](scripts/) may require packages that are not automatically installed. They are auxiliary
and not fundamental for the annotation tool to run. Everything needed for [run.py](run.py) and the main annotation tool should be automatically
installed.*

### Running
The [run.py](run.py) script should provide all needed functionality for the average user. 
It takes a couple optional arguments: verbosity and dates.
The verbosity, `-v` or `--verbose` argument will print helpful status information while running. 
The `date` option allows three methods of specifying which date to run on: 
simply a date string (2018-08-05T17:52), a path to a local file that contains a list of date 
strings where each one is on a different line, or a url to an online list of dates. 
The default is to pull using the url for [dates.txt](dates.txt) stored in this repository. This is preferred to create
a large curated data-set with some repeats for validation. 

## Data
The output of training is saved as a FITS file which is later converted into a labeled png. 
I will share the results of labeling with this tool using 
[Google Drive](https://drive.google.com/open?id=1QYdTTFDYs9Yg1g2zs7rxpj8znXCOwPeY) or as a
 [zipped file](https://drive.google.com/open?id=1J0FGmoa_n37E0Ffzz5MDNDRvUntWWNPA).
A [couple examples](examples/) are available here. 
This labeled data is then used in [machine learning classification of solar images](https://github.com/jmbhughes/smachy). 

<p align="center">
<img src="https://raw.githubusercontent.com/jmbhughes/suvi-trainer/master/examples/thmap_20180604002622_20180716083629.png" width="500">
</p>

## Contributing

Please contact me ([email](mailto:hughes.jmb@gmail.com) or just through pull/issue requests) for updates. 
If you're interested in labeling images or would like access to the labeled database, please contact me 
through [email](mailto:hughes.jmb@gmail.com).
 
## Authors

* **J. Marcus Hughes** - *Initial work*

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Guidance from Dan Seaton, Jon Darnel, and Vicki Hsu
