#!/usr/bin/env python
import sys
import warnings
import versioneer

try:
    from setuptools import setup
except ImportError:
    try:
        from setuptools.core import setup
    except ImportError:
        from distutils.core import setup

from distutils.core import setup

setup(
    name='pyxrf',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author='Brookhaven National Laboratory',
    packages=['pyxrf', 'pyxrf.model', 'pyxrf.view'],
    entry_points={'console_scripts': ['pyxrf = pyxrf.gui:run']},
    package_data={'pyxrf.view': ['*.enaml'], 'configs': ['*.json']},
    include_package_data=True,
    license='BSD',
    classifiers=['Development Status :: 3 - Alpha',
                 "License :: OSI Approved :: BSD License",
                 "Programming Language :: Python :: 2.7",
                 "Topic :: Software Development :: Libraries",
                 "Intended Audience :: Science/Research"]
)
