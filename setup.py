# -*- coding: utf-8 -*-
"""Setup for the `aiida-wannier90-workflows` plugin."""
import json
from setuptools import setup, find_packages

if __name__ == '__main__':
    # pylint: disable=invalid-name

    with open('setup.json', 'r') as handle:
        # Remove comments
        lines = ''.join(line for line in handle if not line.strip().startswith('//'))
        setup_json = json.loads(lines)

    with open('README.md', 'r') as handle:
        long_description = handle.read()

    setup(
        include_package_data=True,
        packages=find_packages(),
        setup_requires=['reentry'],
        reentry_register=True,
        long_description=long_description,
        long_description_content_type='text/markdown',
        **setup_json
    )
