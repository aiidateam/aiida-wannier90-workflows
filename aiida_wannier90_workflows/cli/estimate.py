"""Commands to estimate file size of wannier90."""
import click

from aiida.cmdline.params import arguments as arguments_core
from aiida.cmdline.params import types
from aiida.cmdline.utils import decorators, echo

from .root import cmd_root


@cmd_root.group("estimate")
def cmd_estimate():  # pylint: disable=unused-argument
    """Estimate size of AMN,MMN,EIG,UNK,CHK files."""


def format_option(function):
    """Click option for selecting file format."""
    function = click.option(
        "-f",
        "--formatted/--unformatted",
        is_flag=True,
        default=False,
        show_default=True,
        help="Estimate Fortran formatted or unformatted file size.",
    )(function)
    return function


@cmd_estimate.command("group")
@arguments_core.GROUP(nargs=1)
@format_option
@click.option(
    "-h",
    "--hdf-file",
    type=click.Path(exists=False, readable=True),
    default="wannier_storage_estimation.h5",
    show_default=True,
    help="HDF5 file name to store the estimation.",
)
@decorators.with_dbenv()
def cmd_estimate_group(group, formatted, hdf_file):
    """Estimate file size of all the structure in a group."""
    import os.path

    from aiida_wannier90_workflows.utils.workflows.estimator import (
        WannierFileFormat,
        estimate_structure_group,
        print_estimation,
    )

    if os.path.exists(hdf_file):
        click.confirm(
            f"File {hdf_file} already exists, do you want to overwrite it?", abort=True
        )

    file_format = WannierFileFormat.FORTRAN_UNFORMATTED
    if formatted:
        file_format = WannierFileFormat.FORTRAN_FORMATTED

    estimate_structure_group(group, hdf_file=hdf_file, file_format=file_format)
    echo.echo("")
    print_estimation(hdf_file)


@cmd_estimate.command("structure")
@arguments_core.DATA(
    type=types.DataParamType(sub_classes=("aiida.data:core.structure",)), nargs=1
)
@format_option
@decorators.with_dbenv()
def cmd_estimate_structure(data, formatted):
    """Estimate file size of a structure."""
    from aiida_wannier90_workflows.utils.workflows.estimator import (
        WannierFileFormat,
        estimate_workflow,
        human_readable_size,
    )

    file_format = WannierFileFormat.FORTRAN_UNFORMATTED
    if formatted:
        file_format = WannierFileFormat.FORTRAN_FORMATTED

    estimation = estimate_workflow(structure=data, file_format=file_format)

    for key, val in zip(estimation._fields, estimation):
        if key in ["amn", "mmn", "eig", "chk", "unk"]:
            val = human_readable_size(val)
        echo.echo(f"{key}: {val}")


@cmd_estimate.command("plot")
@click.argument("hdf-file", type=click.Path(exists=True, readable=True))
def cmd_estimate_plot(hdf_file):
    """Plot a histogram of results in a HDF5 file."""
    from aiida_wannier90_workflows.utils.workflows.estimator import (
        plot_histogram,
        print_estimation,
    )

    print_estimation(hdf_file)
    echo.echo("")
    plot_histogram(hdf_file)
