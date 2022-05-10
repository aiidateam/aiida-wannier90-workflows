"""Tests for the `Wannier90WorkChain` class."""
from aiida.engine.utils import instantiate_process
from aiida.manage.manager import get_manager


def instantiate_process_builder(builder):
    """Instantiate a process, from a `ProcessBuilder`."""
    manager = get_manager()
    runner = manager.get_runner()
    return instantiate_process(runner, builder)


def instantiate_process_cls(process_cls, inputs):
    """Instantiate a process, from its class and inputs."""
    manager = get_manager()
    runner = manager.get_runner()
    return instantiate_process(runner, process_cls, **inputs)
