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

### Example

Assuming you have been given a pre-built image directory, grouping, and labels directory 
you should just be able to move into it and call easy_gui.py. 
You can also call easy_gui.py from outside the directory by using the --directory flag 
and then including where your labeling directory is:

```
easy_gui.py --directory marcus_labeling
```

If you're not so fortunate to have been given a pre-built image directory, here is how you can do it:

1. Make a labeling directory. I'll call it `user_labeling` here. 
2. Make a subdirectory for all your images. `cd user_labeling; mkdir images`
3. Divide your images into subdirectories, one subdirectory for each channel, e.g. 304, 195, 171. 
4. Use the build_file_database tool to create a database index for your images. 
It will group them together into "concurrent" observations: a mega-image combining one image 
from each channel at the "same" time, or as close to it as it can. 
```build_file_databases.py images user_labeling/groups.h5 --repeat```
5. In your labeling directory make a "labeled" subdirectory. `cd user_labeling; mkdir labeled`

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



Now you're set to use `easy_gui.py`. Note that there is a more customizable version of the GUI, runs the same thing but just with different parameters called `run_gui.py`. The inputs are described if you call it with no input or with a -h flag.

## Acknowledgments

* Guidance from Dan Seaton and Jon Darnell