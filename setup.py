from setuptools import setup, find_packages
import os
import glob

scripts = glob.glob('./bin/*')
scripts = [os.path.basename(f) for f in scripts if f[-1] != '~']
scripts = [os.path.join('bin', s) for s in scripts]

setup(
    name='montara',
    version='0.2',
    packages=find_packages(),
    scripts=scripts,
    include_package_data=True,
)
