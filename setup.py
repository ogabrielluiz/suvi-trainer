from setuptools import setup, find_packages
import glob

setup(name='suvitrainer',
      version='0.0.1',
      description='Training tool for SUVI thematic map machine learning',
      url='https://github.com/jmbhughes/suvitrainer',
      author='J. Marcus Hughes',
      author_email='hughes.jmb@gmail.com',
      license='MIT',
      packages=find_packages(),
      include_package_data=True,
      # test_suite='nose.collector',
      # tests_require=['nose'],
      scripts=["scripts/easy_gui.py",
               "scripts/run_gui.py"],
      install_requires=['astropy',
                        'numpy',
                        'sunpy',
                        'scikit-image',
                        'matplotlib',
                        'deepdish',
                        'bs4'])