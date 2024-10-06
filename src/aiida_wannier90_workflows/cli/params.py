"""Module for the workflow parameter type."""

import typing as ty

import click

from aiida.cmdline.params import types
from aiida.cmdline.params.options import DRY_RUN, OverridableOption
from aiida.cmdline.utils.decorators import with_dbenv
from aiida.plugins.entry_point import get_entry_point_from_string

__all__ = ("RUN", "DRY_RUN", "FilteredWorkflowParamType")

RUN = OverridableOption(
    "-r", "--run", is_flag=True, help="Perform an actual submission."
)


class FilteredWorkflowParamType(types.WorkflowParamType):
    """The ParamType for identifying WorkflowNode entities or its subclasses.

    Filter the ``process_class`` by the ``self._prcess_classes``.
    """

    def __init__(self, sub_classes=None, process_classes: ty.Sequence[str] = None):
        """Initialize.

        :param process_classes: valid entry points for ``aiida.workflows``, defaults to None
        :type process_classes: ty.Sequence[str], optional
        """
        from aiida.common import exceptions

        super().__init__(sub_classes=sub_classes)

        self._process_classes = []

        if process_classes is not None:
            if not isinstance(process_classes, tuple):
                raise TypeError(
                    "process_classes should be a tuple of entry point strings"
                )

            for entry_point_string in process_classes:
                try:
                    entry_point = get_entry_point_from_string(entry_point_string)
                except (ValueError, exceptions.EntryPointError) as exception:
                    raise ValueError(
                        f"{entry_point_string} is not a valid entry point string: {exception}"
                    ) from exception
                self._process_classes.append(entry_point)

    @with_dbenv()
    def convert(self, value, param, ctx):
        """Attempt to convert the given value to an instance of the orm class using the orm class loader.

        Also check the ``process_class``.
        """
        entity = super().convert(value, param, ctx)

        if self._process_classes:
            for entry_point in self._process_classes:
                try:
                    process_class = entry_point.load()
                except ImportError as exception:
                    raise RuntimeError(
                        f"failed to load the entry point {entry_point}: {exception}"
                    ) from exception

                if issubclass(entity.process_class, process_class):
                    break
            else:
                raise click.BadParameter(
                    f"the process type {entity.process_type} of entity {entity} is not a sub class "
                    f"of {[_.name for _ in self._process_classes]}"
                )

        return entity
