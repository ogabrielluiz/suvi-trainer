# suvi-trainer

The [Solar Ultraviolet Imager](https://www.goes-r.gov/spacesegment/suvi.html) (SUVI) aboard the National Oceanic 
and Atmospheric Administration's Geostationary Operational Environmental R-Series satellites is used 
in machine learning applications to create thematic maps, images showing where different features are on the Sun. This
tool allows domain experts to load images, manipulate them, and create labeled maps, which are used in training solar 
image classifiers. 
 
## Getting Started

These instructions will get you a copy of the project up and running on your local machine 
for development and testing purposes. At the moment, it is advised to run this code in a fresh virtual environment with
Python 3.5. Installation should only require running the `setup.py` script. 
 
### Installing
After [creating a separate virtual environment](https://realpython.com/python-virtual-environments-a-primer/), installation should be simply:
```
python3 setup.py install
```

### Running
The [run.py](scripts/run.py) script should provide all needed functionality. It takes a couple optional arguments: verbosity and dates.
The verbosity, `-v` or `--verbose` argument will print helpful information while running. The `date` option allows three
methods of specifying which date to run on: simply a date string (2018-08-05T17:52), a path to a local file that contains a list of date 
strings where one is on each line, or a url to an online list of dates. The default is to pull using the url for [dates.txt](dates.txt) stored 
in this repository. 

## Contributing

Please contact me ([email](mailto:hughes.jmb@gmail.com) or just through pull/issue requests) for updates. 
If you're interested in labeling images or would like access to the labeled database, please contact me 
through [email](mailto:hughes.jmb@gmail.com).
 
## Authors

* **J. Marcus Hughes** - *Initial work*

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details



Now you're set to use `easy_gui.py`. Note that there is a more customizable version of the GUI, runs the same thing but just with different parameters called `run_gui.py`. The inputs are described if you call it with no input or with a -h flag.

## Acknowledgments

* Guidance from Dan Seaton, Jon Darnel, and Vicki Hsu
