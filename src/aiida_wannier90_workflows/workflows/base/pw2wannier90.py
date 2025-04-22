"""Wrapper workchain for Pw2wannier90Calculation to automatically handle several errors."""

import pathlib
import typing as ty

from aiida import orm
from aiida.common import AttributeDict
from aiida.common.lang import type_check
from aiida.engine import process_handler
from aiida.engine.processes.builder import ProcessBuilder
from aiida.orm.nodes.data.base import to_aiida_type

from aiida_quantumespresso.calculations.pw2wannier90 import Pw2wannier90Calculation
from aiida_quantumespresso.common.types import ElectronicType
from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin

from aiida_wannier90_workflows.common.types import WannierProjectionType

from .qebaserestart import QeBaseRestartWorkChain

__all__ = ["validate_inputs_base", "validate_inputs", "Pw2wannier90BaseWorkChain"]


def validate_inputs_base(  # pylint: disable=unused-argument
    inputs: AttributeDict, ctx=None
) -> None:
    """Validate the inputs of the entire input namespace."""
    return


def validate_inputs(  # pylint: disable=unused-argument,inconsistent-return-statements
    inputs: AttributeDict, ctx=None
) -> None:
    """Validate the inputs of the entire input namespace of `Pw2wannier90BaseWorkChain`."""

    result = validate_inputs_base(inputs, ctx)  # pylint: disable=assignment-from-none
    if result:
        return result

    calc_inputs = AttributeDict(inputs[Pw2wannier90BaseWorkChain._inputs_namespace])
    calc_parameters = calc_inputs["parameters"].get_dict().get("inputpp", {})

    scdm_proj = calc_parameters.get("scdm_proj", False)
    scdm_entanglement = calc_parameters.get("scdm_entanglement", "isolated")

    fit_scdm = False
    if scdm_proj and scdm_entanglement != "isolated":
        scdm_mu = calc_parameters.get("scdm_mu", None)
        scdm_sigma = calc_parameters.get("scdm_sigma", None)
        if scdm_mu is None or scdm_sigma is None:
            fit_scdm = True

    if fit_scdm:
        # Check bands and bands_projections are provided
        if any(_ not in inputs for _ in ("bands_projections", "bands")):
            return "`scdm_proj` is True but `bands_projections` or `bands` is empty"

        # Check `bands` and `bands_projections` are consistent
        bands_num_kpoints, bands_num_bands = inputs["bands"].attributes["array|bands"]
        projections_num_kpoints, projections_num_bands = inputs[
            "bands_projections"
        ].base.attributes.all["array|proj_array_0"]
        if bands_num_kpoints != projections_num_kpoints:
            return (
                "`bands` and `bands_projections` have different number of kpoints: "
                f"{bands_num_kpoints} != {projections_num_kpoints}"
            )
        if bands_num_bands != projections_num_bands:
            return (
                "`bands` and `bands_projections` have different number of bands: "
                f"{bands_num_bands} != {projections_num_bands}"
            )

    atom_proj = calc_parameters.get("atom_proj", False)
    atom_proj_ext = calc_parameters.get("atom_proj_ext", False)
    atom_proj_dir = calc_parameters.get("atom_proj_dir", None)
    if atom_proj and atom_proj_ext and not atom_proj_dir:
        return "`atom_proj_dir` must be specified when using external projectors."


class Pw2wannier90BaseWorkChain(ProtocolMixin, QeBaseRestartWorkChain):
    """Workchain to run a pw2wannier90 calculation with automated error handling and restarts."""

    _process_class = Pw2wannier90Calculation
    _inputs_namespace = "pw2wannier90"

    @classmethod
    def define(cls, spec) -> None:
        """Define the process spec."""
        super().define(spec)

        spec.input(
            "scdm_sigma_factor",
            valid_type=orm.Float,
            default=lambda: orm.Float(3.0),
            serializer=to_aiida_type,
            help="The `sigma` factor of occupation function for SCDM projection.",
        )
        spec.input(
            "bands",
            valid_type=orm.BandsData,
            required=False,
            help="Bands to calculate SCDM `mu`, `sigma`.",
        )
        spec.input(
            "bands_projections",
            valid_type=orm.ProjectionData,
            required=False,
            help="Bands projectability to calculate SCDM `mu`, `sigma`.",
        )
        spec.inputs.validator = validate_inputs

        spec.exit_code(
            400,
            "ERROR_SCDM_FITTING",
            message="Error when fitting `scdm_mu` and `scdm_sigma`.",
        )

    @classmethod
    def get_protocol_filepath(cls) -> pathlib.Path:
        """Return the ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files

        from .. import protocols

        return files(protocols) / "base" / "pw2wannier90.yaml"

    @classmethod
    def get_builder_from_protocol(
        cls,
        code: ty.Union[orm.Code, str, int],
        *,
        protocol: str = None,
        overrides: dict = None,
        electronic_type: ElectronicType = ElectronicType.METAL,
        projection_type: WannierProjectionType = WannierProjectionType.ATOMIC_PROJECTORS_QE,
        exclude_projectors: list = None,
        external_projectors_path: str = None,
        external_projectors_list: dict = None,
        external_projectors_froz: list = None,
    ) -> ProcessBuilder:
        """Return a builder prepopulated with inputs selected according to the chosen protocol.

        :param code: [description]
        :type code: ty.Union[orm.Code, str, int]
        :param structure: [description]
        :type structure: orm.StructureData
        :param protocol: [description], defaults to None
        :type protocol: str, optional
        :param overrides: [description], defaults to None
        :type overrides: dict, optional
        :return: [description]
        :rtype: ProcessBuilder
        """
        from aiida_quantumespresso.workflows.protocols.utils import recursive_merge

        if isinstance(code, (int, str)):
            code = orm.load_code(code)

        type_check(code, orm.Code)
        type_check(electronic_type, ElectronicType)
        type_check(projection_type, WannierProjectionType)

        # Update the parameters based on the protocol inputs
        inputs = cls.get_protocol_inputs(protocol, overrides)
        parameters = inputs[cls._inputs_namespace]["parameters"]["inputpp"]
        metadata = inputs[cls._inputs_namespace]["metadata"]

        # Set projection
        if projection_type == WannierProjectionType.SCDM:
            parameters["scdm_proj"] = True

            if electronic_type == ElectronicType.INSULATOR:
                parameters["scdm_entanglement"] = "isolated"
            else:
                parameters["scdm_entanglement"] = "erfc"
                # scdm_mu, scdm_sigma will be set at runtime
        elif projection_type in [
            WannierProjectionType.ATOMIC_PROJECTORS_QE,
            WannierProjectionType.ATOMIC_PROJECTORS_EXTERNAL,
        ]:
            parameters["atom_proj"] = True
            if exclude_projectors is not None and len(exclude_projectors) > 0:
                parameters["atom_proj_exclude"] = list(exclude_projectors)
            if projection_type == WannierProjectionType.ATOMIC_PROJECTORS_EXTERNAL:
                parameters["atom_proj_ext"] = True
                if external_projectors_path is None:
                    raise ValueError(
                        f"Must specify `external_projectors_path` when using {projection_type}"
                    )
                parameters["atom_proj_dir"] = "external_projectors/"
                if (
                    external_projectors_froz is not None
                    and len(external_projectors_froz) > 0
                ):
                    parameters["atom_proj_frozen"] = list(external_projectors_froz)

        parameters = {"inputpp": parameters}

        # If overrides are provided, they take precedence over default protocol
        if overrides:
            parameter_overrides = overrides.get(cls._inputs_namespace, {}).get(
                "parameters", {}
            )
            parameters = recursive_merge(parameters, parameter_overrides)
            metadata_overrides = overrides.get(cls._inputs_namespace, {}).get(
                "metadata", {}
            )
            metadata = recursive_merge(metadata, metadata_overrides)

        # pylint: disable=no-member
        builder = cls.get_builder()
        builder[cls._inputs_namespace]["code"] = code
        builder[cls._inputs_namespace]["parameters"] = orm.Dict(parameters)
        builder[cls._inputs_namespace]["metadata"] = metadata
        if "settings" in inputs[cls._inputs_namespace]:
            builder[cls._inputs_namespace]["settings"] = orm.Dict(
                dict=inputs[cls._inputs_namespace]["settings"]
            )
        if "settings" in inputs:
            builder["settings"] = orm.Dict(inputs["settings"])
        if projection_type == WannierProjectionType.ATOMIC_PROJECTORS_EXTERNAL:
            builder[cls._inputs_namespace]["external_projectors_path"] = orm.RemoteData(
                remote_path=external_projectors_path, computer=code.computer
            )
            builder[cls._inputs_namespace]["external_projectors_list"] = orm.Dict(
                external_projectors_list
            )
        builder.clean_workdir = orm.Bool(inputs["clean_workdir"])
        # pylint: enable=no-member

        return builder

    def setup(self) -> None:
        """Call the `setup` of the `BaseRestartWorkChain` and then create the inputs dictionary in `self.ctx.inputs`.

        This `self.ctx.inputs` dictionary will be used by the `BaseRestartWorkChain` to submit
        the calculations in the internal loop.
        """
        super().setup()

        self.ctx.inputs = self.prepare_inputs()

    def prepare_inputs(self) -> AttributeDict:
        """Prepare `Pw2wannier90Calculation` inputs according to workchain input sepc.

        Different from `get_builder_from_protocol', this function is executed at runtime.
        """
        from aiida_wannier90_workflows.utils.scdm import fit_scdm_mu_sigma

        inputs = AttributeDict(
            self.exposed_inputs(Pw2wannier90Calculation, self._inputs_namespace)
        )
        parameters = inputs["parameters"].get_dict().get("inputpp", {})

        scdm_proj = parameters.get("scdm_proj", False)
        scdm_entanglement = parameters.get("scdm_entanglement", None)
        scdm_mu = parameters.get("scdm_mu", None)
        scdm_sigma = parameters.get("scdm_sigma", None)

        fit_scdm = (
            scdm_proj
            and scdm_entanglement == "erfc"
            and (scdm_mu is None or scdm_sigma is None)
        )

        if scdm_entanglement == "gaussian":
            if scdm_mu is None or scdm_sigma is None:
                raise NotImplementedError(
                    "scdm_entanglement = gaussian but scdm_mu or scdm_sigma is empty."
                )

        if fit_scdm:
            # pylint: disable=unbalanced-tuple-unpacking
            try:
                mu_new, sigma_new = fit_scdm_mu_sigma(
                    self.inputs.bands,
                    self.inputs.bands_projections,
                    self.inputs.scdm_sigma_factor,
                )
            except ValueError:
                # raise ValueError(f'SCDM mu/sigma fitting failed! {exc.args}') from exc
                return self.exit_codes.ERROR_SCDM_FITTING

            # If `scdm_mu` and/or `scdm_sigma` is present in the input parameters,
            # the workchain will directly use them, only the missing one will be populated.
            if "scdm_mu" not in parameters:
                parameters["scdm_mu"] = mu_new
            if "scdm_sigma" not in parameters:
                parameters["scdm_sigma"] = sigma_new
            inputs["parameters"] = orm.Dict({"inputpp": parameters})

        return inputs

    @process_handler(
        exit_codes=[_process_class.exit_codes.ERROR_OUTPUT_STDOUT_INCOMPLETE]
    )
    def handle_output_stdout_incomplete(self, calculation):
        """Overide parent function."""
        return super().handle_output_stdout_incomplete(calculation)
