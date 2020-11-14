from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='adtrack_ctr',
    version='0.1',
    description='CTR Prediction using AdTrack data.',
    author='Weng Hoe Chen',
    packages=find_packages()
)
