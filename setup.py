import os
import codecs

from setuptools import setup, find_packages

def readme():
    with codecs.open('README.rst', encoding='utf-8-sig') as f:
        return f.read()

version_file= os.path.join('binning', '_version.py')
__version__= "0.0.1"
with open(version_file) as f:
    exec(f.read())

DISTNAME= 'daa-adaptive-binning'
DESCRIPTION= 'Some general purpose binning functionalities'
LONG_DESCRIPTION= readme()
LONG_DESCRIPTION_CONTENT_TYPE='text/x-rst'
MAINTAINER= 'The DAALAB'
MAINTAINER_EMAIL= 'gyorgy.kovacs@daalab.com'
URL= 'https://github.com/TheDAALab/binning'
LICENSE= 'MIT'
DOWNLOAD_URL= 'https://github.com/TheDAALab/binning'
VERSION= __version__
CLASSIFIERS= [  'Intended Audience :: Science/Research',
                'Intended Audience :: Developers',
                'Development Status :: 3 - Alpha',
                'License :: OSI Approved :: MIT License',
                'Programming Language :: Python',
                'Topic :: Scientific/Engineering :: Artificial Intelligence',
                'Topic :: Software Development',
                'Operating System :: Microsoft :: Windows',
                'Operating System :: POSIX',
                'Operating System :: Unix',
                'Operating System :: MacOS']

on_rtd = os.environ.get('READTHEDOCS') == 'True'
if on_rtd:
    INSTALL_REQUIRES = ['numpy', 'scipy', 'kmeans1d']
else:
    INSTALL_REQUIRES= ['numpy', 'scipy', 'kmeans1d']

EXTRAS_REQUIRE= {'tests': ['pytest'],
                    'docs': ['sphinx', 'sphinx-gallery', 'sphinx_rtd_theme']}

PYTHON_REQUIRES= '>=3.8'
PACKAGE_DIR= {'binning': 'binning'}
SETUP_REQUIRES=['setuptools>=41.0.1', 'wheel>=0.33.4', 'pytest-runner']
TESTS_REQUIRE=['pytest']

setup(name=DISTNAME,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        author=MAINTAINER,
        author_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        url=URL,
        version=VERSION,
        download_url=DOWNLOAD_URL,
        long_description=LONG_DESCRIPTION,
        long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
        zip_safe=False,
        classifiers=CLASSIFIERS,
        install_requires=INSTALL_REQUIRES,
        extras_require=EXTRAS_REQUIRE,
        python_requires=PYTHON_REQUIRES,
        setup_requires=SETUP_REQUIRES,
        package_dir=PACKAGE_DIR,
        packages=find_packages(exclude=[]),
        package_data={},
        include_package_data=True)
