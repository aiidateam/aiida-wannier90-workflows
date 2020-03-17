# -*- coding: utf-8 -*-

import re
import json

from setuptools import setup, find_packages

if __name__ == '__main__':
    with open('setup.json', 'r') as info:
        kwargs = json.load(info)
    setup(
        include_package_data=True,
        setup_requires=['reentry'],
        reentry_register=True,
        packages=find_packages(exclude=['aiida']),
        **kwargs
    )
