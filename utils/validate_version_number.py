#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pre-commit script to ensure that version numbers are matched.

Check `setup.json` and `aiida_wannier90_workflows/__init__.py`.
"""
import os
import json
import sys

import click

FILEPATH_SCRIPT = os.path.split(os.path.realpath(__file__))[0]
FILEPATH_ROOT = os.path.join(FILEPATH_SCRIPT, os.pardir)
FILENAME_SETUP_JSON = 'setup.json'
FILEPATH_SETUP_JSON = os.path.join(FILEPATH_ROOT, FILENAME_SETUP_JSON)


def get_setup_json():
    """Return the `setup.json` as a python dictionary."""
    with open(FILEPATH_SETUP_JSON, 'r') as handle:
        # Remove comments
        lines = ''.join(line for line in handle if not line.strip().startswith('//'))
        setup_json = json.loads(lines)

    return setup_json


@click.group()
def cli():
    """Define the main CLI group."""


@cli.command('version')
def validate_version():
    """Check that version numbers in `setup.json` and `aiida_wannier90_workflows/__init__.py` match."""
    sys.path.insert(0, FILEPATH_ROOT)
    import aiida_wannier90_workflows  # pylint: disable=wrong-import-position
    version = aiida_wannier90_workflows.__version__

    setup_content = get_setup_json()

    if version != setup_content['version']:
        click.echo('Version number mismatch detected:')
        click.echo(f"Version number in '{FILENAME_SETUP_JSON}': {setup_content['version']}")
        click.echo(f"Version number in 'aiida_wannier90_workflows/__init__.py': {version}")
        click.echo(f"Updating version in '{FILENAME_SETUP_JSON}' to: {version}")

        setup_content['version'] = version
        with open(FILEPATH_SETUP_JSON, 'w') as handle:
            # Write with indentation of two spaces and explicitly define separators to not have spaces at end of lines
            json.dump(setup_content, handle, indent=4, separators=(',', ': '))

        sys.exit(1)


if __name__ == '__main__':
    cli()  # pylint: disable=no-value-for-parameter
