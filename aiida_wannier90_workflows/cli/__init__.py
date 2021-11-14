# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position,wildcard-import
"""Module for the command line interface."""
import click_completion

# Activate the completion of parameter types provided by the click_completion package
click_completion.init()

from .root import cmd_root
from .list import cmd_list
from .plot import cmd_plot
from .node import cmd_node
from .estimate import cmd_estimate
from .group import cmd_group
