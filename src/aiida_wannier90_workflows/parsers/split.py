"""Parsers.

Register parsers via the "aiida.parsers" entry point in setup.json.
"""

import typing as ty

from aiida import orm
from aiida.common import exceptions
from aiida.engine import ExitCode
from aiida.parsers.parser import Parser

from aiida_wannier90_workflows.calculations.split import Wannier90SplitCalculation


class Wannier90SplitParser(Parser):
    """Parser class for parsing output of split."""

    def __init__(self, node):
        """Initialize Parser instance.

        Checks that the ProcessNode being passed was produced by a Gw2wannier90Calculation.

        :param node: ProcessNode of calculation
        :param type node: :class:`aiida.orm.ProcessNode`
        """
        super().__init__(node)
        if not issubclass(node.process_class, Wannier90SplitCalculation):
            raise exceptions.ParsingError("Can only parse Wannier90SplitCalculation")

    def parse(self, **kwargs):
        """Parse outputs, store results in database.

        :returns: an exit code, if parsing fails (or nothing if parsing succeeds)
        """
        output_filename = self.node.get_option("output_filename")

        # Check that folder content is as expected
        files_retrieved = self.retrieved.list_object_names()
        files_expected = [
            Wannier90SplitCalculation._DEFAULT_OUTPUT_FILE,  # pylint: disable=protected-access
        ]
        # Note: set(A) <= set(B) checks whether A is a subset of B
        if not set(files_expected) <= set(files_retrieved):
            self.logger.error(
                f"Found files '{files_retrieved}', expected to find '{files_expected}'"
            )
            return self.exit_codes.ERROR_MISSING_OUTPUT_FILES

        # parse `split.out`
        self.logger.info(f"Parsing '{output_filename}'")
        with self.retrieved.open(output_filename, "r") as handle:
            output_node = parse_out(handle.readlines())
        self.out("output_parameters", output_node)

        remote_folder = self.node.outputs["remote_folder"]
        remote_path = remote_folder.get_remote_path()
        # For py <= 3.8, no removesuffix
        # seedname = Wannier90Calculation._DEFAULT_INPUT_FILE.removesuffix(
        #     Wannier90Calculation._REQUIRED_INPUT_SUFFIX
        # )  # = aiida
        seedname = "aiida"
        required_files = [f"{seedname}.amn", f"{seedname}.mmn", f"{seedname}.eig"]
        for k in ["val", "cond"]:
            remote_folder_k = orm.RemoteData(
                computer=remote_folder.computer,
                remote_path=f"{remote_path}/{k}",
            )
            if not set(required_files) <= set(remote_folder_k.listdir()):
                self.logger.error(
                    f"In remote folder for {k}: {remote_folder_k.get_remote_path()}, "
                    f"found files '{remote_folder_k.listdir()}', expected to find '{required_files}'"
                )
                return self.exit_codes.ERROR_MISSING_OUTPUT_FILES
            self.out(f"remote_folder_{k}", remote_folder_k)

        return ExitCode(0)


def parse_out(filecontent: ty.List[str]) -> orm.Dict:  # pylint: disable=unused-argument
    """Parse `split.out`."""
    parameters = {}

    # regexs = {
    #     "timestamp_started": re.compile(r"Started on\s*(.+)"),
    #     "num_kpoints": re.compile(r"Kpoints number:\s*([0-9]+)"),
    # }

    # for line in filecontent:
    #     for key, reg in regexs.items():
    #         match = reg.match(line.strip())
    #         if match:
    #             parameters[key] = match.group(1)
    #             regexs.pop(key, None)
    #             break

    # parameters["num_kpoints"] = int(parameters["num_kpoints"])

    return orm.Dict(parameters)
