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
After creating a separate virtual environment, installation should be simply:
```
python setup.py install
```
The [scripts](./scripts) will then allow you to execute the training tool. The scripts require a HDF5 database of images
to run, called the groups database. It collects concurrent images at different wavelengths together for labeling; see 
the [example](./examples/groups.h5). If you're labeling images for me, this should all be provided and you will only 
need to run [easy_gui.py](./scripts/easy_gui.py). 

## Todo
* Supplement information on how to create groups database and get running from raw imagse
* Provide Docker deployment version
* Add feature to overlay drawn regions onto the Sun
* Add more I/O for revising images
* Test on more systems for bugs

## Contributing

Please contact me ([email](mailto:hughes.jmb@gmail.com) or just through pull/issue requests) for updates. 
If you're interested in labeling images or would like access to the labeled database, please contact me 
through [email](mailto:hughes.jmb@gmail.com).
 
## Authors

* **J. Marcus Hughes** - *Initial work*

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Guidance from Dan Seaton and Jon Darnell
