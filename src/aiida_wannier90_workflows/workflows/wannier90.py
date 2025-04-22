"""Base class for Wannierisation workflow."""

# pylint: disable=protected-access
import pathlib
import typing as ty

from aiida import orm
from aiida.common import AttributeDict
from aiida.common.lang import type_check
from aiida.engine.processes import ProcessBuilder, ToContext, WorkChain, if_
from aiida.orm.nodes.data.base import to_aiida_type

from aiida_quantumespresso.common.types import ElectronicType, SpinType
from aiida_quantumespresso.utils.mapping import prepare_process_inputs
from aiida_quantumespresso.workflows.protocols.utils import (
    ProtocolMixin,
    recursive_merge,
)
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain

from aiida_wannier90_workflows.common.types import (
    WannierDisentanglementType,
    WannierFrozenType,
    WannierProjectionType,
)

from .base.projwfc import ProjwfcBaseWorkChain
from .base.pw2wannier90 import Pw2wannier90BaseWorkChain
from .base.wannier90 import Wannier90BaseWorkChain

__all__ = ["validate_inputs", "Wannier90WorkChain"]


def validate_inputs(  # pylint: disable=unused-argument,inconsistent-return-statements
    inputs, ctx=None
):
    """Validate the inputs of the entire input namespace of `Wannier90WorkChain`."""
    # If no scf inputs, the nscf must have a `parent_folder`
    if "scf" not in inputs:
        if "nscf" in inputs and "parent_folder" not in inputs["nscf"]["pw"]:
            return "If skipping scf step, nscf inputs must have a `parent_folder`"

    # Cannot specify both `auto_energy_windows` and `scdm_proj`
    pw2wannier_parameters = inputs["pw2wannier90"]["pw2wannier90"][
        "parameters"
    ].get_dict()
    auto_energy_windows = inputs["wannier90"].get("auto_energy_windows", False)
    scdm_proj = pw2wannier_parameters["inputpp"].get("scdm_proj", False)
    if auto_energy_windows and scdm_proj:
        return "`auto_energy_windows` is incompatible with SCDM"

    # Cannot specify both `auto_energy_windows` and `shift_energy_windows`
    shift_energy_windows = inputs["wannier90"].get("shift_energy_windows", False)
    if auto_energy_windows and shift_energy_windows:
        return "`auto_energy_windows` and `shift_energy_windows` are incompatible"


# pylint: disable=fixme,too-many-lines
class Wannier90WorkChain(
    ProtocolMixin, WorkChain
):  # pylint: disable=too-many-public-methods
    """Workchain to obtain maximally localised Wannier functions (MLWF).

    Run the following steps:
        scf -> nscf -> projwfc -> wannier90 postproc -> pw2wannier90 -> wannier90
    """

    @classmethod
    def define(cls, spec):
        """Define the process spec."""
        from .base.pw2wannier90 import (
            validate_inputs_base as validate_inputs_base_pw2wannier90,
        )
        from .base.wannier90 import (
            validate_inputs_base as validate_inputs_base_wannier90,
        )

        super().define(spec)

        spec.input(
            "structure", valid_type=orm.StructureData, help="The input structure."
        )
        spec.input(
            "clean_workdir",
            valid_type=orm.Bool,
            serializer=to_aiida_type,
            default=lambda: orm.Bool(False),
            help=(
                "If True, work directories of all called calculation will be cleaned "
                "at the end of execution."
            ),
        )
        spec.expose_inputs(
            PwBaseWorkChain,
            namespace="scf",
            exclude=("clean_workdir", "pw.structure"),
            namespace_options={
                "required": False,
                "populate_defaults": False,
                "help": "Inputs for the `PwBaseWorkChain` for the SCF calculation.",
            },
        )
        spec.expose_inputs(
            PwBaseWorkChain,
            namespace="nscf",
            exclude=("clean_workdir", "pw.structure", "pw.parent_folder"),
            namespace_options={
                "required": False,
                "populate_defaults": False,
                "help": "Inputs for the `PwBaseWorkChain` for the NSCF calculation.",
            },
        )
        spec.expose_inputs(
            ProjwfcBaseWorkChain,
            namespace="projwfc",
            exclude=("clean_workdir", "projwfc.parent_folder"),
            namespace_options={
                "required": False,
                "populate_defaults": False,
                "help": "Inputs for the `ProjwfcBaseWorkChain`.",
            },
        )
        spec.expose_inputs(
            Pw2wannier90BaseWorkChain,
            namespace="pw2wannier90",
            exclude=(
                "clean_workdir",
                "pw2wannier90.parent_folder",
                "pw2wannier90.nnkp_file",
            ),
            namespace_options={"help": "Inputs for the `Pw2wannier90BaseWorkChain`."},
        )
        spec.inputs["pw2wannier90"].validator = validate_inputs_base_pw2wannier90
        spec.expose_inputs(
            Wannier90BaseWorkChain,
            namespace="wannier90",
            exclude=("clean_workdir", "wannier90.structure"),
            namespace_options={"help": "Inputs for the `Wannier90BaseWorkChain`."},
        )
        spec.inputs["wannier90"].validator = validate_inputs_base_wannier90

        spec.inputs.validator = validate_inputs

        spec.outline(
            cls.setup,
            if_(cls.should_run_scf)(
                cls.run_scf,
                cls.inspect_scf,
            ),
            if_(cls.should_run_nscf)(
                cls.run_nscf,
                cls.inspect_nscf,
            ),
            if_(cls.should_run_projwfc)(
                cls.run_projwfc,
                cls.inspect_projwfc,
            ),
            cls.run_wannier90_pp,
            cls.inspect_wannier90_pp,
            cls.run_pw2wannier90,
            cls.inspect_pw2wannier90,
            cls.run_wannier90,
            cls.inspect_wannier90,
            cls.results,
        )

        spec.expose_outputs(
            PwBaseWorkChain, namespace="scf", namespace_options={"required": False}
        )
        spec.expose_outputs(
            PwBaseWorkChain, namespace="nscf", namespace_options={"required": False}
        )
        spec.expose_outputs(
            ProjwfcBaseWorkChain,
            namespace="projwfc",
            namespace_options={"required": False},
        )
        spec.expose_outputs(Pw2wannier90BaseWorkChain, namespace="pw2wannier90")
        spec.expose_outputs(Wannier90BaseWorkChain, namespace="wannier90_pp")
        spec.expose_outputs(Wannier90BaseWorkChain, namespace="wannier90")

        spec.exit_code(
            420,
            "ERROR_SUB_PROCESS_FAILED_SCF",
            message="the scf PwBaseWorkChain sub process failed",
        )
        spec.exit_code(
            430,
            "ERROR_SUB_PROCESS_FAILED_NSCF",
            message="the nscf PwBaseWorkChain sub process failed",
        )
        spec.exit_code(
            440,
            "ERROR_SUB_PROCESS_FAILED_PROJWFC",
            message="the ProjwfcBaseWorkChain sub process failed",
        )
        spec.exit_code(
            450,
            "ERROR_SUB_PROCESS_FAILED_WANNIER90PP",
            message="the postproc Wannier90BaseWorkChain sub process failed",
        )
        spec.exit_code(
            460,
            "ERROR_SUB_PROCESS_FAILED_PW2WANNIER90",
            message="the Pw2wannier90BaseWorkChain sub process failed",
        )
        spec.exit_code(
            470,
            "ERROR_SUB_PROCESS_FAILED_WANNIER90",
            message="the Wannier90BaseWorkChain sub process failed",
        )
        spec.exit_code(
            480, "ERROR_SANITY_CHECK_FAILED", message="outputs sanity check failed"
        )

    @classmethod
    def get_protocol_filepath(cls) -> pathlib.Path:
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files

        from . import protocols

        return files(protocols) / "wannier90.yaml"

    @classmethod
    def get_protocol_overrides(cls) -> dict:
        """Get the ``overrides`` for various input arguments of the ``get_builder_from_protocol()`` method."""
        from importlib_resources import files
        import yaml

        from . import protocols

        path = files(protocols) / "overrides" / "wannier90.yaml"
        with path.open() as file:
            return yaml.safe_load(file)

    @classmethod
    def get_builder_from_protocol(  # pylint: disable=unused-argument
        cls,
        codes: ty.Mapping[str, ty.Union[str, int, orm.Code]],
        structure: orm.StructureData,
        *,
        protocol: str = None,
        overrides: dict = None,
        pseudo_family: str = None,
        electronic_type: ElectronicType = ElectronicType.METAL,
        spin_type: SpinType = SpinType.NONE,
        initial_magnetic_moments: dict = None,
        projection_type: WannierProjectionType = WannierProjectionType.SCDM,
        disentanglement_type: WannierDisentanglementType = None,
        frozen_type: WannierFrozenType = None,
        exclude_semicore: bool = True,
        external_projectors_path: str = None,
        external_projectors: dict = None,
        plot_wannier_functions: bool = False,
        retrieve_hamiltonian: bool = False,
        retrieve_matrices: bool = False,
        print_summary: bool = True,
        summary: dict = None,
    ) -> ProcessBuilder:
        """Return a builder prepopulated with inputs selected according to the chosen protocol.

        The builder can be submitted directly by `aiida.engine.submit(builder)`.

        :param codes: a dictionary of ``Code`` instance for pw.x, pw2wannier90.x, wannier90.x, (optionally) projwfc.x.
        :type codes: dict
        :param structure: the ``StructureData`` instance to use.
        :type structure: orm.StructureData
        :param protocol: protocol to use, if not specified, the default will be used.
        :type protocol: str
        :param overrides: optional dictionary of inputs to override the defaults of the protocol.
        :param electronic_type: indicate the electronic character of the system through ``ElectronicType`` instance.
        :param spin_type: indicate the spin polarization type to use through a ``SpinType`` instance.
        :param initial_magnetic_moments: optional dictionary that maps the initial magnetic moment of
        each kind to a desired value for a spin polarized calculation.
        Note that for ``spin_type == SpinType.COLLINEAR`` an initial guess for the magnetic moment
        is automatically set in case this argument is not provided.
        :param projection_type: indicate the Wannier initial projection type of the system
        through ``WannierProjectionType`` instance.
        Default to SCDM.
        :param disentanglement_type: indicate the Wannier disentanglement type of the system through
        ``WannierDisentanglementType`` instance. Default to None, which will choose the best type
        based on `projection_type`:
            For WannierProjectionType.SCDM, use WannierDisentanglementType.NONE
            For other WannierProjectionType, use WannierDisentanglementType.SMV
        :param frozen_type: indicate the Wannier disentanglement type of the system
        through ``WannierFrozenType`` instance. Default to None, which will choose
        the best frozen type based on `electronic_type` and `projection_type`.
            for ElectronicType.INSULATOR, use WannierFrozenType.NONE
            for metals or insulators with conduction bands:
                for WannierProjectionType.ANALYTIC/RANDOM, use WannierFrozenType.ENERGY_FIXED
                for WannierProjectionType.ATOMIC_PROJECTORS_QE/OPENMX, use WannierFrozenType.FIXED_PLUS_PROJECTABILITY
                for WannierProjectionType.SCDM, use WannierFrozenType.NONE
        :param maximal_localisation: if true do maximal localisation of Wannier functions.
        :param exclude_semicores: if True do not Wannierise semicore states.
        :param plot_wannier_functions: if True plot Wannier functions as xsf files.
        :param retrieve_hamiltonian: if True retrieve Wannier Hamiltonian.
        :param retrieve_matrices: if True retrieve amn/mmn/eig/chk/spin files.
        :param print_summary: if True print a summary of key input parameters
        :param summary: A dict containing key input parameters and can be printed out
        when the `get_builder_from_protocol` returns, to let user have a quick check of the
        generated inputs. Since in python dict is pass-by-reference, the input dict can be
        modified in the method and used by the invoking function. This allows printing the
        summary only by the last overriding method.
        :return: a process builder instance with all inputs defined and ready for launch.
        :rtype: ProcessBuilder
        """
        from aiida_wannier90_workflows.utils.pseudo import (
            get_frozen_list_ext,
            get_pseudo_orbitals,
            get_semicore_list,
            get_semicore_list_ext,
        )
        from aiida_wannier90_workflows.utils.workflows.builder.projections import (
            guess_wannier_projection_types,
        )
        from aiida_wannier90_workflows.utils.workflows.builder.submit import check_codes

        # Check function arguments
        codes = check_codes(codes)
        type_check(electronic_type, ElectronicType)
        type_check(spin_type, SpinType)
        type_check(projection_type, WannierProjectionType)
        if disentanglement_type:
            type_check(disentanglement_type, WannierDisentanglementType)
        if frozen_type:
            type_check(frozen_type, WannierFrozenType)

        if electronic_type not in [ElectronicType.METAL, ElectronicType.INSULATOR]:
            raise NotImplementedError(
                f"electronic type `{electronic_type}` is not supported."
            )

        if spin_type not in [
            SpinType.NONE,
            SpinType.SPIN_ORBIT,
            SpinType.NON_COLLINEAR,
        ]:
            raise NotImplementedError(f"spin type `{spin_type}` is not supported.")

        if initial_magnetic_moments and spin_type == SpinType.NONE:
            raise ValueError(
                f"`initial_magnetic_moments` is specified but spin type `{spin_type}` is incompatible."
            )

        (
            projection_type,
            disentanglement_type,
            frozen_type,
        ) = guess_wannier_projection_types(
            electronic_type=electronic_type,
            projection_type=projection_type,
            disentanglement_type=disentanglement_type,
            frozen_type=frozen_type,
        )

        if projection_type == WannierProjectionType.ATOMIC_PROJECTORS_EXTERNAL:
            if external_projectors_path is None:
                raise ValueError(
                    f"Must specify `external_projectors_path` when using {projection_type}"
                )
            type_check(external_projectors_path, str)
            if external_projectors is None:
                raise ValueError(
                    f"Must specify `external_projectors` when using {projection_type}"
                )
            type_check(external_projectors, dict)

        # Adapt overrides based on input arguments
        # Note: if overrides are specified, they take precedence!
        protocol_overrides = cls.get_protocol_overrides()

        # If recursive_merge get an arg = None, the arg.copy() will raise an error.
        # When overrides is not given (default value None), it should be set to an empty dict.
        if overrides is None:
            overrides = {}

        if plot_wannier_functions:
            overrides = recursive_merge(
                protocol_overrides["plot_wannier_functions"], overrides
            )

        if retrieve_hamiltonian:
            overrides = recursive_merge(
                protocol_overrides["retrieve_hamiltonian"], overrides
            )

        if retrieve_matrices:
            overrides = recursive_merge(
                protocol_overrides["retrieve_matrices"], overrides
            )

        if pseudo_family is None:
            if spin_type == SpinType.SPIN_ORBIT:
                # Use fully relativistic PseudoDojo for SOC
                pseudo_family = "PseudoDojo/0.4/PBE/FR/standard/upf"
            else:
                # Use the one used in Wannier90BaseWorkChain
                pseudo_family = (
                    pseudo_family
                    or Wannier90BaseWorkChain.get_protocol_inputs(protocol=protocol)[
                        "meta_parameters"
                    ]["pseudo_family"]
                )

        # As PwBaseWorkChain.get_builder_from_protocol() does not support SOC, we have to pass the
        # desired parameters through the overrides. In this case we need to set the `pw.x`
        # spin_type to SpinType.NONE, otherwise the builder will raise an error.
        # This block should be removed once SOC is supported in PwBaseWorkChain.
        spin_orbit_coupling = spin_type == SpinType.SPIN_ORBIT
        spin_non_collinear = (
            spin_type == SpinType.NON_COLLINEAR
        ) or spin_orbit_coupling

        if spin_type == SpinType.NON_COLLINEAR:
            overrides = recursive_merge(
                protocol_overrides["spin_noncollinear"], overrides
            )
        if spin_type == SpinType.SPIN_ORBIT:
            overrides = recursive_merge(protocol_overrides["spin_orbit"], overrides)
            if (initial_magnetic_moments is not None) or any(
                "magmom" in kind for kind in structure.base.attributes.all["kinds"]
            ):
                pw_spin_type = SpinType.NON_COLLINEAR
            else:
                pw_spin_type = SpinType.NONE
        else:
            pw_spin_type = spin_type

        inputs = cls.get_protocol_inputs(protocol=protocol, overrides=overrides)

        builder = cls.get_builder()
        builder.structure = structure
        builder.clean_workdir = orm.Bool(inputs.get("clean_workdir"))

        # Prepare wannier90 builder
        wannier_overrides = inputs.get("wannier90", {})
        wannier_overrides.setdefault("meta_parameters", {})
        wannier_overrides["meta_parameters"].setdefault(
            "exclude_semicore", exclude_semicore
        )
        wannier_builder = Wannier90BaseWorkChain.get_builder_from_protocol(
            code=codes["wannier90"],
            structure=structure,
            protocol=protocol,
            overrides=wannier_overrides,
            electronic_type=electronic_type,
            spin_type=spin_type,
            projection_type=projection_type,
            disentanglement_type=disentanglement_type,
            frozen_type=frozen_type,
            pseudo_family=pseudo_family,
            external_projectors=external_projectors,
        )
        # Remove workchain excluded inputs
        wannier_builder["wannier90"].pop("structure", None)
        wannier_builder.pop("clean_workdir", None)
        builder.wannier90 = wannier_builder._inputs(prune=True)

        # Prepare SCF builder
        scf_overrides = inputs.get("scf", {})
        scf_overrides["pseudo_family"] = pseudo_family
        scf_builder = PwBaseWorkChain.get_builder_from_protocol(
            code=codes["pw"],
            structure=structure,
            protocol=protocol,
            overrides=scf_overrides,
            electronic_type=electronic_type,
            spin_type=pw_spin_type,
        )
        # Remove workchain excluded inputs
        scf_builder["pw"].pop("structure", None)
        scf_builder.pop("clean_workdir", None)
        builder.scf = scf_builder._inputs(prune=True)

        # Prepare NSCF builder
        nscf_overrides = inputs.get("nscf", {})
        nscf_overrides["pseudo_family"] = pseudo_family

        num_bands = wannier_builder["wannier90"]["parameters"]["num_bands"]
        exclude_bands = (
            wannier_builder["wannier90"]["parameters"]
            .get_dict()
            .get("exclude_bands", [])
        )
        nscf_overrides["pw"]["parameters"]["SYSTEM"]["nbnd"] = num_bands + len(
            exclude_bands
        )

        nscf_builder = PwBaseWorkChain.get_builder_from_protocol(
            code=codes["pw"],
            structure=structure,
            protocol=protocol,
            overrides=nscf_overrides,
            electronic_type=electronic_type,
            spin_type=pw_spin_type,
        )
        # Use explicit list of kpoints generated by wannier builder.
        # Since the QE auto generated kpoints might be different from wannier90, here we explicitly
        # generate a list of kpoint coordinates to avoid discrepancies.
        nscf_builder.pop("kpoints_distance", None)
        nscf_builder.kpoints = wannier_builder["wannier90"]["kpoints"]

        # Remove workchain excluded inputs
        nscf_builder["pw"].pop("structure", None)
        nscf_builder.pop("clean_workdir", None)
        builder.nscf = nscf_builder._inputs(prune=True)

        # Prepare projwfc builder
        if projection_type == WannierProjectionType.SCDM:
            run_projwfc = True
        else:
            if (  # pylint: disable=simplifiable-if-statement
                frozen_type == WannierFrozenType.ENERGY_AUTO
            ):
                run_projwfc = True
            else:
                run_projwfc = False
        if run_projwfc:
            projwfc_overrides = inputs.get("projwfc", {})
            projwfc_builder = ProjwfcBaseWorkChain.get_builder_from_protocol(
                code=codes["projwfc"], protocol=protocol, overrides=projwfc_overrides
            )
            # Remove workchain excluded inputs
            projwfc_builder.pop("clean_workdir", None)
            builder.projwfc = projwfc_builder._inputs(prune=True)

        # Prepare pw2wannier90 builder
        exclude_projectors = None
        external_projectors_list = None
        external_projectors_froz = None

        if projection_type == WannierProjectionType.ATOMIC_PROJECTORS_EXTERNAL:
            external_projectors_list = {
                kind.name: kind.symbol for kind in structure.kinds
            }
            external_projectors_froz = get_frozen_list_ext(
                structure=structure,
                external_projectors=external_projectors,
                spin_non_collinear=spin_non_collinear,
            )
        if exclude_semicore:
            pseudo_orbitals = get_pseudo_orbitals(builder["scf"]["pw"]["pseudos"])
            if projection_type == WannierProjectionType.ATOMIC_PROJECTORS_EXTERNAL:
                exclude_projectors = get_semicore_list_ext(
                    structure, external_projectors, pseudo_orbitals, spin_non_collinear
                )
            else:
                exclude_projectors = get_semicore_list(
                    structure, pseudo_orbitals, spin_non_collinear
                )
        pw2wannier90_overrides = inputs.get("pw2wannier90", {})
        pw2wannier90_builder = Pw2wannier90BaseWorkChain.get_builder_from_protocol(
            code=codes["pw2wannier90"],
            protocol=protocol,
            overrides=pw2wannier90_overrides,
            electronic_type=electronic_type,
            projection_type=projection_type,
            exclude_projectors=exclude_projectors,
            external_projectors_path=external_projectors_path,
            external_projectors_list=external_projectors_list,
            external_projectors_froz=external_projectors_froz,
        )
        # Remove workchain excluded inputs
        pw2wannier90_builder.pop("clean_workdir", None)
        builder.pw2wannier90 = pw2wannier90_builder._inputs(prune=True)

        # A dictionary containing key info of Wannierisation and will be printed when the function returns.
        if summary is None:
            summary = {}
        summary["Formula"] = structure.get_formula()
        summary["PseudoFamily"] = pseudo_family
        summary["ElectronicType"] = electronic_type.name
        summary["SpinType"] = spin_type.name
        summary["WannierProjectionType"] = projection_type.name
        summary["WannierDisentanglementType"] = disentanglement_type.name
        summary["WannierFrozenType"] = frozen_type.name

        params = builder["wannier90"]["wannier90"]["parameters"].get_dict()
        summary["num_bands"] = params["num_bands"]
        summary["num_wann"] = params["num_wann"]
        if "exclude_bands" in params:
            summary["exclude_bands"] = params["exclude_bands"]
        summary["mp_grid"] = params["mp_grid"]

        notes = summary.get("notes", [])
        summary["notes"] = notes

        if print_summary:
            cls.print_summary(summary)

        return builder

    @classmethod
    def print_summary(cls, summary: ty.Dict) -> None:
        """Try to pretty print the summary when the `get_builder_from_protocol` returns."""
        notes = summary.pop("notes", [])

        print("Summary of key input parameters:")
        for key, val in summary.items():
            print(f"  {key}: {val}")
        print("")

        if len(notes) == 0:
            return

        print("Notes:")
        for note in notes:
            print(f"  * {note}")

    def setup(self) -> None:
        """Define the current structure in the context to be the input structure."""
        self.ctx.current_structure = self.inputs.structure

        if not self.should_run_scf():
            if self.should_run_nscf():
                self.ctx.current_folder = self.inputs["nscf"]["pw"]["parent_folder"]
            elif self.should_run_projwfc():
                self.ctx.current_folder = self.inputs["projwfc"]["projwfc"][
                    "parent_folder"
                ]
            else:
                self.ctx.current_folder = self.inputs["pw2wannier90"]["pw2wannier90"][
                    "parent_folder"
                ]

    def should_run_scf(self) -> bool:
        """If the 'scf' input namespace was specified, run the scf workchain."""
        return "scf" in self.inputs

    def run_scf(self):
        """Run the `PwBaseWorkChain` in scf mode on the current structure."""
        inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace="scf"))
        inputs.pw.structure = self.ctx.current_structure
        inputs.metadata.call_link_label = "scf"

        inputs = prepare_process_inputs(PwBaseWorkChain, inputs)
        running = self.submit(PwBaseWorkChain, **inputs)
        self.report(f"launching {running.process_label}<{running.pk}> in scf mode")

        return ToContext(workchain_scf=running)

    def inspect_scf(self):  # pylint: disable=inconsistent-return-statements
        """Verify that the `PwBaseWorkChain` for the scf run successfully finished."""
        workchain = self.ctx.workchain_scf

        if not workchain.is_finished_ok:
            self.report(
                f"scf {workchain.process_label} failed with exit status {workchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_SCF

        self.ctx.current_folder = workchain.outputs.remote_folder

    def should_run_nscf(self) -> bool:
        """If the `nscf` input namespace was specified, run the nscf workchain."""
        return "nscf" in self.inputs

    def run_nscf(self):
        """Run the PwBaseWorkChain in nscf mode."""
        inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace="nscf"))
        inputs.pw.structure = self.ctx.current_structure
        inputs.pw.parent_folder = self.ctx.current_folder
        inputs.metadata.call_link_label = "nscf"

        inputs = prepare_process_inputs(PwBaseWorkChain, inputs)
        running = self.submit(PwBaseWorkChain, **inputs)
        self.report(f"launching {running.process_label}<{running.pk}> in nscf mode")

        return ToContext(workchain_nscf=running)

    def inspect_nscf(self):  # pylint: disable=inconsistent-return-statements
        """Verify that the `PwBaseWorkChain` for the nscf run successfully finished."""
        workchain = self.ctx.workchain_nscf

        if not workchain.is_finished_ok:
            self.report(
                f"nscf {workchain.process_label} failed with exit status {workchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_NSCF

        self.ctx.current_folder = workchain.outputs.remote_folder

    def should_run_projwfc(self) -> bool:
        """If the 'projwfc' input namespace was specified, run the projwfc calculation."""
        return "projwfc" in self.inputs

    def run_projwfc(self):
        """Projwfc step."""
        inputs = AttributeDict(
            self.exposed_inputs(ProjwfcBaseWorkChain, namespace="projwfc")
        )
        inputs.projwfc.parent_folder = self.ctx.current_folder
        inputs.metadata.call_link_label = "projwfc"

        inputs = prepare_process_inputs(ProjwfcBaseWorkChain, inputs)
        running = self.submit(ProjwfcBaseWorkChain, **inputs)
        self.report(f"launching {running.process_label}<{running.pk}>")

        return ToContext(workchain_projwfc=running)

    def inspect_projwfc(self):  # pylint: disable=inconsistent-return-statements
        """Verify that the `ProjwfcCalculation` for the projwfc run successfully finished."""
        workchain = self.ctx.workchain_projwfc

        if not workchain.is_finished_ok:
            self.report(
                f"{workchain.process_label} failed with exit status {workchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_PROJWFC

    def prepare_wannier90_pp_inputs(self):  # pylint: disable=too-many-statements
        """Prepare the inputs of wannier90 calculation before submission.

        This method will be called by the workchain at runtime, to fill some parameters such as
        Fermi energy which can only be retrieved after scf step.
        Moreover, this allows overriding the method in derived classes to further modify the inputs.
        """
        from aiida_wannier90_workflows.utils.workflows.pw import (
            get_fermi_energy,
            get_fermi_energy_from_nscf,
        )

        base_inputs = AttributeDict(
            self.exposed_inputs(Wannier90BaseWorkChain, namespace="wannier90")
        )
        inputs = base_inputs["wannier90"]
        inputs.structure = self.ctx.current_structure
        parameters = inputs.parameters.get_dict()

        # Add Fermi energy
        if "workchain_scf" in self.ctx:
            scf_output_parameters = self.ctx.workchain_scf.outputs.output_parameters
            fermi_energy = get_fermi_energy(scf_output_parameters)
        elif "workchain_nscf" in self.ctx:
            fermi_energy = get_fermi_energy_from_nscf(self.ctx.workchain_nscf)
        else:
            if "fermi_energy" in parameters:
                fermi_energy = parameters["fermi_energy"]
            else:
                raise ValueError("Cannot retrieve Fermi energy from scf or nscf output")
        parameters["fermi_energy"] = fermi_energy

        inputs.parameters = orm.Dict(parameters)

        # Add `postproc_setup`
        if "settings" in inputs:
            settings = inputs["settings"].get_dict()
        else:
            settings = {}
        settings["postproc_setup"] = True
        inputs["settings"] = settings

        # I should not stash files in postproc, otherwise there is a RemoteStashFolderData in outputs
        inputs["metadata"]["options"].pop("stash", None)

        base_inputs["wannier90"] = inputs

        if base_inputs["shift_energy_windows"] and "bands" not in base_inputs:
            if "workchain_scf" in self.ctx:
                output_band = self.ctx.workchain_scf.outputs.output_band
            elif "workchain_nscf" in self.ctx:
                output_band = self.ctx.workchain_nscf.outputs.output_band
            else:
                raise ValueError("No output scf or nscf bands")
            base_inputs.bands = output_band

        if base_inputs["auto_energy_windows"]:
            if "bands" not in base_inputs:
                base_inputs.bands = self.ctx.workchain_projwfc.outputs.bands
            if "bands_projections" not in base_inputs:
                base_inputs.bands_projections = (
                    self.ctx.workchain_projwfc.outputs.projections
                )

        base_inputs["clean_workdir"] = orm.Bool(False)

        return base_inputs

    def run_wannier90_pp(self):
        """Wannier90 post processing step."""
        inputs = self.prepare_wannier90_pp_inputs()
        inputs["metadata"] = {"call_link_label": "wannier90_pp"}

        inputs = prepare_process_inputs(Wannier90BaseWorkChain, inputs)
        running = self.submit(Wannier90BaseWorkChain, **inputs)
        self.report(f"launching {running.process_label}<{running.pk}> in postproc mode")

        return ToContext(workchain_wannier90_pp=running)

    def inspect_wannier90_pp(self):  # pylint: disable=inconsistent-return-statements
        """Verify that the `Wannier90Calculation` for the wannier90 run successfully finished."""
        workchain = self.ctx.workchain_wannier90_pp

        if not workchain.is_finished_ok:
            self.report(
                f"wannier90 postproc {workchain.process_label} failed with exit status {workchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_WANNIER90PP

    def prepare_pw2wannier90_inputs(self):
        """Prepare the inputs of `Pw2wannier90BaseWorkChain` before submission.

        This method will be called by the workchain at runtime, so it can dynamically add/modify inputs
        based on outputs of previous calculations, e.g. add bands and projections for calculating
        scdm_mu/sigma from projectability, etc.
        Moreover, it can be overridden in derived classes.
        """
        base_inputs = AttributeDict(
            self.exposed_inputs(Pw2wannier90BaseWorkChain, namespace="pw2wannier90")
        )
        inputs = base_inputs["pw2wannier90"]
        parameters = inputs.parameters.get_dict().get("inputpp", {})

        scdm_proj = parameters.get("scdm_proj", False)
        scdm_entanglement = parameters.get("scdm_entanglement", None)
        scdm_mu = parameters.get("scdm_mu", None)
        scdm_sigma = parameters.get("scdm_sigma", None)

        fit_scdm = (
            scdm_proj
            and scdm_entanglement == "erfc"
            and (scdm_mu is None or scdm_sigma is None)
        )

        if fit_scdm:
            if "workchain_projwfc" not in self.ctx:
                raise ValueError("Needs to run projwfc for SCDM projection")
            base_inputs["bands"] = self.ctx.workchain_projwfc.outputs.bands
            base_inputs["bands_projections"] = (
                self.ctx.workchain_projwfc.outputs.projections
            )

        inputs["parent_folder"] = self.ctx.current_folder
        inputs["nnkp_file"] = self.ctx.workchain_wannier90_pp.outputs.nnkp_file

        base_inputs["pw2wannier90"] = inputs

        return base_inputs

    def run_pw2wannier90(self):
        """Run the pw2wannier90 step."""
        inputs = self.prepare_pw2wannier90_inputs()
        inputs.metadata.call_link_label = "pw2wannier90"

        inputs = prepare_process_inputs(Pw2wannier90BaseWorkChain, inputs)
        running = self.submit(Pw2wannier90BaseWorkChain, **inputs)
        self.report(f"launching {running.process_label}<{running.pk}>")

        return ToContext(workchain_pw2wannier90=running)

    def inspect_pw2wannier90(self):  # pylint: disable=inconsistent-return-statements
        """Verify that the Pw2wannier90BaseWorkChain for the pw2wannier90 run successfully finished."""
        workchain = self.ctx.workchain_pw2wannier90

        if not workchain.is_finished_ok:
            self.report(
                f"{workchain.process_label} failed with exit status {workchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_PW2WANNIER90

        self.ctx.current_folder = workchain.outputs.remote_folder

    def prepare_wannier90_inputs(self):  # pylint: disable=too-many-statements
        """Prepare the inputs of wannier90 calculation before submission.

        This method will be called by the workchain at runtime, to fill some parameters such as
        Fermi energy which can only be retrieved after scf step.
        Moreover, this allows overriding the method in derived classes to further modify the inputs.
        """
        from copy import deepcopy

        from aiida_wannier90_workflows.utils.workflows import get_last_calcjob

        base_inputs = AttributeDict(
            self.exposed_inputs(Wannier90BaseWorkChain, namespace="wannier90")
        )

        # I need to disable Fermi energy shifting since this is done in postproc step,
        # otherwise it will be shifted twice!
        base_inputs.pop("shift_energy_windows", None)
        base_inputs.pop("auto_energy_windows", None)
        base_inputs.pop("auto_energy_windows_threshold", None)
        base_inputs.pop("bands", None)
        base_inputs.pop("bands_projections", None)

        inputs = base_inputs["wannier90"]

        # I should stash files, which was removed from metadata in the postproc step
        stash = None
        if "stash" in inputs["metadata"]["options"]:
            stash = deepcopy(inputs["metadata"]["options"]["stash"])

        # Use the Wannier90BaseWorkChain-corrected parameters
        last_calc = get_last_calcjob(self.ctx.workchain_wannier90_pp)
        # copy postproc inputs, especially the `kmesh_tol` might have been corrected
        for key in last_calc.inputs:
            inputs[key] = last_calc.inputs[key]

        inputs["remote_input_folder"] = self.ctx.current_folder

        if "settings" in inputs:
            settings = inputs.settings.get_dict()
        else:
            settings = {}
        settings["postproc_setup"] = False

        inputs.settings = settings

        # Restore stash files
        if stash:
            options = deepcopy(inputs["metadata"]["options"])
            options["stash"] = stash
            inputs["metadata"]["options"] = options

        base_inputs["wannier90"] = inputs
        base_inputs["clean_workdir"] = orm.Bool(False)

        return base_inputs

    def run_wannier90(self):
        """Wannier90 step for MLWF."""
        inputs = self.prepare_wannier90_inputs()
        inputs["metadata"] = {"call_link_label": "wannier90"}

        inputs = prepare_process_inputs(Wannier90BaseWorkChain, inputs)
        running = self.submit(Wannier90BaseWorkChain, **inputs)
        self.report(f"launching {running.process_label}<{running.pk}>")

        return ToContext(workchain_wannier90=running)

    def inspect_wannier90(self):  # pylint: disable=inconsistent-return-statements
        """Verify that the `Wannier90BaseWorkChain` for the wannier90 run successfully finished."""
        workchain = self.ctx.workchain_wannier90

        if not workchain.is_finished_ok:
            self.report(
                f"{workchain.process_label} failed with exit status {workchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_WANNIER90

        self.ctx.current_folder = workchain.outputs.remote_folder

    def results(self):  # pylint: disable=inconsistent-return-statements
        """Attach the desired output nodes directly as outputs of the workchain."""

        if "workchain_scf" in self.ctx:
            self.out_many(
                self.exposed_outputs(
                    self.ctx.workchain_scf, PwBaseWorkChain, namespace="scf"
                )
            )

        if "workchain_nscf" in self.ctx:
            self.out_many(
                self.exposed_outputs(
                    self.ctx.workchain_nscf, PwBaseWorkChain, namespace="nscf"
                )
            )

        if "workchain_projwfc" in self.ctx:
            self.out_many(
                self.exposed_outputs(
                    self.ctx.workchain_projwfc,
                    ProjwfcBaseWorkChain,
                    namespace="projwfc",
                )
            )

        self.out_many(
            self.exposed_outputs(
                self.ctx.workchain_pw2wannier90,
                Pw2wannier90BaseWorkChain,
                namespace="pw2wannier90",
            )
        )
        self.out_many(
            self.exposed_outputs(
                self.ctx.workchain_wannier90_pp,
                Wannier90BaseWorkChain,
                namespace="wannier90_pp",
            )
        )
        self.out_many(
            self.exposed_outputs(
                self.ctx.workchain_wannier90,
                Wannier90BaseWorkChain,
                namespace="wannier90",
            )
        )

        result = self.sanity_check()
        if result:
            return result

        self.report(f"{self.get_name()} successfully completed")

    def sanity_check(self):  # pylint: disable=inconsistent-return-statements
        """Sanity checks for final outputs.

        Not necessary but it is good to check it.
        """
        from aiida_wannier90_workflows.utils.pseudo import (
            get_number_of_electrons,
            get_number_of_projections,
        )

        # If using external atomic projectors, disable sanity check
        p2w_params = self.ctx.workchain_pw2wannier90.inputs["pw2wannier90"][
            "parameters"
        ].get_dict()["inputpp"]
        atom_proj = p2w_params.get("atom_proj", False)
        atom_proj_ext = p2w_params.get("atom_proj_ext", False)
        if atom_proj and atom_proj_ext:
            return

        # 1. the calculated number of projections is consistent with QE projwfc.x
        check_num_projs = True
        if self.should_run_scf():
            pseudos = self.inputs["scf"]["pw"]["pseudos"]
            spin_orbit_coupling = (
                self.inputs["scf"]["pw"]["parameters"]
                .get_dict()["SYSTEM"]
                .get("lspinorb", False)
            )
        elif self.should_run_nscf():
            pseudos = self.inputs["nscf"]["pw"]["pseudos"]
            spin_orbit_coupling = (
                self.inputs["nscf"]["pw"]["parameters"]
                .get_dict()["SYSTEM"]
                .get("lspinorb", False)
            )
        else:
            check_num_projs = False
            pseudos = None
            spin_orbit_coupling = None
        if check_num_projs:
            args = {
                "structure": self.ctx.current_structure,
                # The type of `self.inputs['scf']['pw']['pseudos']` is AttributesFrozendict,
                # we need to convert it to dict, otherwise get_number_of_projections will fail.
                "pseudos": dict(pseudos),
            }
            if "workchain_projwfc" in self.ctx:
                num_proj = len(
                    self.ctx.workchain_projwfc.outputs["projections"].get_orbitals()
                )
                params = self.ctx.workchain_wannier90.inputs["wannier90"][
                    "parameters"
                ].get_dict()
                spin_non_collinear = params.get("spinors", False)
                number_of_projections = get_number_of_projections(
                    **args,
                    spin_non_collinear=spin_non_collinear,
                    spin_orbit_coupling=spin_orbit_coupling,
                )
                if number_of_projections != num_proj:
                    self.report(
                        f"number of projections {number_of_projections} != projwfc.x output {num_proj}"
                    )
                    return self.exit_codes.ERROR_SANITY_CHECK_FAILED

        # 2. the number of electrons is consistent with QE output
        # only check num electrons when we already know pseudos in the check num projectors step
        check_num_elecs = check_num_projs
        if "workchain_scf" in self.ctx:
            num_elec = self.ctx.workchain_scf.outputs["output_parameters"][
                "number_of_electrons"
            ]
        elif "workchain_nscf" in self.ctx:
            num_elec = self.ctx.workchain_nscf.outputs["output_parameters"][
                "number_of_electrons"
            ]
        else:
            check_num_elecs = False
            num_elec = None  # to avoid pylint errors
        if check_num_elecs:
            number_of_electrons = get_number_of_electrons(**args)
            if number_of_electrons != num_elec:
                self.report(
                    f"number of electrons {number_of_electrons} != QE output {num_elec}"
                )
                return self.exit_codes.ERROR_SANITY_CHECK_FAILED

    def on_terminated(self):
        """Clean the working directories of all child calculations if `clean_workdir=True` in the inputs."""
        super().on_terminated()

        if not self.inputs.clean_workdir:
            self.report("remote folders will not be cleaned")
            return

        cleaned_calcs = []

        for called_descendant in self.node.called_descendants:
            if isinstance(called_descendant, orm.CalcJobNode):
                try:
                    called_descendant.outputs.remote_folder._clean()  # pylint: disable=protected-access
                    cleaned_calcs.append(called_descendant.pk)
                except (OSError, KeyError):
                    pass

        if cleaned_calcs:
            self.report(
                f"cleaned remote folders of calculations: {' '.join(map(str, cleaned_calcs))}"
            )
