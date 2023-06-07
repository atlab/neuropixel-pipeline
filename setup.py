#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

# read in version number into __version__
with open(path.join(here, "neuropixel_pipeline", "version.py")) as f:
    exec(f.read())

with open(path.join(here, "requirements.txt")) as f:
    requirements = [line.split("#", 1)[0].rstrip() for line in f.readlines()]

setup(
    name="neuropixel-pipeline",
    version=__version__,  # noqa: F821
    description="Schemata and libraries for ingesting and clustering Neuropixel data.",
    author="Christos Papadopoulos",
    author_email="cpapadop@bcm.edu",
    packages=find_packages(exclude=[]),
    install_requires=requirements,
)
