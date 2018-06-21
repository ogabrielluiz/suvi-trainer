from setuptools import setup, find_packages
import glob

scripts = glob.glob("bin/*.py") + glob.glob("bin/*.sh")

setup(name='suvi-trainer',
      version='0.0.1',
      description='Training tool for SUVI thematic map machine learning',
      url='https://github.com/jmbhughes/suvi-trainer',
      author='J. Marcus Hughes',
      author_email='hughes.jmb@gmail.com',
      license='MIT',
      packages=find_packages(),
      include_package_data=True,
      # test_suite='nose.collector',
      # tests_require=['nose'],
      scripts=scripts,
      install_requires=['astropy',
                        'numpy',
                        'sunpy',
                        'scikit-image',
                        'matplotlib'])

      # install_requires=['astropy==2.0.3',
      #                   'cycler==0.10.0',
      #                   'decorator==4.2.1',
      #                   'deepdish==0.3.6',
      #                   'matplotlib==2.1.1',
      #                   'networkx==2.0',
      #                   'numexpr==2.6.4',
      #                   'h5py==2.7.0',
      #                   'numpy==1.13.3',
      #                   'pandas==0.22.0',
      #                   'Pillow==4.3.0',
      #                   'pyparsing==2.2.0',
      #                   'python-dateutil==2.6.1',
      #                   'pytz==2017.3',
      #                   'PyWavelets==0.5.2',
      #                   'scikit-image==0.13.1',
      #                   'scipy==1.0.0',
      #                   'six==1.11.0',
      #                   'smachy===0.0.0d1',
      #                   'sunpy==0.7.9', # does not have errors with net importing
      #                   'tables==3.4.2',
      #                   'scikit-learn==0.19.1',
      #                   'tensorflow==1.1.0',
      #                   'keras==2.1.4',
      #                   'suds-jurko'])# for sunpy because suds will not install on ubuntu
