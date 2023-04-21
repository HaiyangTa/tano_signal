from setuptools import setup, find_packages
import codecs
import os

VERSION = '0.0.1'
DESCRIPTION = 'A package for one dimensional data process'
LONG_DESCRIPTION = 'Contains CNN-transformer, CNN model and data representations implementation.'

# Setting up
setup(
    name="DeepSpectra",
    version=VERSION,
    author="Haiyang Tang",
    author_email="<Haiyang.Tang@outlook.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=['opencv-python', 'torch'],
    keywords=['python', 'spectra','spectrum','CNN-transformer', 'CNN', 'Representation learning' ,'data representation'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: scientist",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
