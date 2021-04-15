import os
from setuptools import setup, find_packages

__version__ = None
pth = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "montara",
    "_version.py")
with open(pth, 'r') as fp:
    exec(fp.read())


setup(
    name='montara',
    version=__version__,
    packages=find_packages(),
    include_package_data=True,
)
