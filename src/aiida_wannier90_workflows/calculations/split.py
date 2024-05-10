"""Calculations for gw2wannier90.py."""

from aiida import orm
from aiida.common import datastructures
from aiida.engine import CalcJob

from aiida_wannier90.calculations import Wannier90Calculation

from aiida_wannier90_workflows.utils.str import removesuffix


class Wannier90SplitCalculation(CalcJob):
    """AiiDA calculation plugin wrapping the split AMN/MMN/EIG script."""

    _DEFAULT_INPUT_FOLDER = "."
    _DEFAULT_OUTPUT_FILE = "split.out"
    _DEFAULT_OUTPUT_FOLDER_VAL = "val"
    _DEFAULT_OUTPUT_FOLDER_COND = "cond"
    _DEFAULT_OUTPUT_FOLDER_TRUNC = "truncate"

    _REQUIRED_INPUT_SUFFIX = ["win", "mmn", "eig", "chk"]

    @classmethod
    def define(cls, spec):
        """Define inputs and outputs of the calculation."""
        super().define(spec)

        spec.input(
            "num_val",
            valid_type=orm.Int,
            required=True,
            serializer=orm.to_aiida_type,
            help="Number of valence WFs.",
        )
        spec.input(
            "rotate_unk",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            serializer=orm.to_aiida_type,
            help="Number of valence WFs.",
        )

        # set default values for AiiDA options
        spec.inputs["metadata"]["options"]["resources"].default = {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
        }
        spec.inputs["metadata"]["options"][
            "parser_name"
        ].default = "wannier90_workflows.split"

        # new ports
        spec.input(
            "metadata.options.output_filename",
            valid_type=str,
            default=cls._DEFAULT_OUTPUT_FILE,
        )
        spec.input(
            "parent_folder",
            valid_type=orm.RemoteData,
            required=False,
            help="Remote folder containing win/amn/mmn/eig/unk files.",
        )
        spec.output(
            "output_parameters",
            valid_type=orm.Dict,
            help="Output parameters.",
        )
        spec.output(
            "remote_folder_val",
            valid_type=orm.RemoteData,
            help="Remote folder for valence.",
        )
        spec.output(
            "remote_folder_cond",
            valid_type=orm.RemoteData,
            help="Remote folder for conduction.",
        )

        spec.exit_code(
            300,
            "ERROR_MISSING_OUTPUT_FILES",
            message="Calculation did not produce all expected output files.",
        )
        spec.exit_code(
            301,
            "ERROR_NO_RETRIEVED_TEMPORARY_FOLDER",
            message="The retrieved temporary folder could not be accessed.",
        )

    def prepare_for_submission(self, folder):
        """Create input files.

        :param folder: an `aiida.common.folders.Folder` where the plugin should temporarily place all files
            needed by the calculation.
        :return: `aiida.common.datastructures.CalcInfo` instance
        """
        # The input amn/mmn/eig are in `./` folder,
        # The output amn/mmn/eig are in the `val`/`cond` dirs.
        w90_default_seedname = removesuffix(
            Wannier90Calculation._DEFAULT_INPUT_FILE,  # pylint: disable=protected-access
            Wannier90Calculation._REQUIRED_INPUT_SUFFIX,  # pylint: disable=protected-access
        )  # actually = aiida

        # Prepare a `CalcInfo` to be returned to the engine
        calcinfo = datastructures.CalcInfo()
        calcinfo.codes_info = []

        cmdline_params = [
            "--nval",
            f"{self.inputs['num_val'].value}",  # num_val
        ]
        if self.inputs["rotate_unk"]:
            cmdline_params.append("--rotate-unk")
        cmdline_params.extend(
            [
                # "--valence_dir",
                # f"{self._DEFAULT_OUTPUT_FOLDER_VAL}",
                # "--conduction_dir",
                # f"{self._DEFAULT_OUTPUT_FOLDER_COND}",
                f"{self._DEFAULT_INPUT_FOLDER}/{w90_default_seedname}",
            ]
        )
        codeinfo = datastructures.CodeInfo()
        codeinfo.cmdline_params = cmdline_params
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.stdout_name = self.metadata.options.output_filename
        # codeinfo.withmpi = self.metadata.options.withmpi
        codeinfo.withmpi = False

        calcinfo.codes_info.append(codeinfo)

        calcinfo.retrieve_list = [
            self.metadata.options.output_filename,
        ]

        calcinfo.remote_symlink_list = []

        parent_folder = self.inputs.parent_folder
        computer_uuid = parent_folder.computer.uuid
        parent_folder_path = parent_folder.get_remote_path()

        for suffix in self._REQUIRED_INPUT_SUFFIX:
            entry = (
                computer_uuid,
                f"{parent_folder_path}/{w90_default_seedname}.{suffix}",
                f"{self._DEFAULT_INPUT_FOLDER}/{w90_default_seedname}.{suffix}",
            )
            calcinfo.remote_symlink_list.append(entry)

        if self.inputs.rotate_unk:
            for fname in parent_folder.listdir():
                if fname.startswith("UNK"):
                    entry = (
                        computer_uuid,
                        f"{parent_folder_path}/{fname}",
                        fname,
                    )
                    calcinfo.remote_symlink_list.append(entry)

        return calcinfo
