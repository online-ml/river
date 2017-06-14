import os
from setuptools import setup, find_packages

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "scikit-multiflow",
    version = "0.1.0",
    author = "Guilherme Matsumoto",
    author_email = "guilherme.kurike-matsumoto@ensta-paristech.fr",
    description = ("Setup for the scikit-multiflow package"),
    packages=find_packages(),
    long_description=read('README.md'),
)
