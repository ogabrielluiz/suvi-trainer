from setuptools import setup, find_packages

with open("Readme.md", "r") as fh:
    long_description = fh.read()

setup(name='suvitrainer',
      version='1.0.6',
      description='Training tool for SUVI thematic map machine learning',
      url='https://github.com/jmbhughes/suvi-trainer',
      author='J. Marcus Hughes',
      author_email='hughes.jmb@gmail.com',
      license='MIT',
      packages=find_packages(),
      include_package_data=True,
      # test_suite='nose.collector',
      # tests_require=['nose'],
      scripts=["run.py"],
      long_description=long_description,
      long_description_content_type="text/markdown",
      install_requires=['astropy',
                        'numpy',
                        'sunpy',
                        'scikit-image',
                        'matplotlib',
                        'bs4',
                        'requests',
                        'suds-jurko',
                        'drms',
                        "scipy",
                        "python-dateutil"])
