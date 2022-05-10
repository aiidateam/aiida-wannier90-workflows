# pylint: disable=wrong-import-position,wildcard-import
"""Module for the command line interface."""
import click_completion

# Activate the completion of parameter types provided by the click_completion package
click_completion.init()

from .estimate import cmd_estimate
from .group import cmd_group
from .list import cmd_list
from .node import cmd_node
from .plot import cmd_plot
from .root import cmd_root
from .statistics import cmd_statistics
