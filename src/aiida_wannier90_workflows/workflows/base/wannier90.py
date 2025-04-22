"""Wrapper workchain for `Wannier90Calculation` to automatically handle several errors."""

import pathlib
import typing as ty

from aiida import orm
from aiida.common import AttributeDict
from aiida.common.lang import type_check
from aiida.engine import (
    BaseRestartWorkChain,
    ProcessHandlerReport,
    process_handler,
    while_,
)
from aiida.engine.processes.builder import ProcessBuilder
from aiida.orm.nodes.data.base import to_aiida_type

from aiida_quantumespresso.common.types import ElectronicType, SpinType
from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin

from aiida_wannier90.calculations import Wannier90Calculation

from aiida_wannier90_workflows.common.types import (
    WannierDisentanglementType,
    WannierFrozenType,
    WannierProjectionType,
)

__all__ = ["validate_inputs_base", "validate_inputs", "Wannier90BaseWorkChain"]

# pylint: disable=inconsistent-return-statements


def validate_inputs_base(inputs: AttributeDict, ctx=None) -> None:
    """Validate the inputs of the entire input namespace."""
    # pylint: disable=inconsistent-return-statements,unused-argument

    # Check `settings`
    if "settings" in inputs:
        settings = inputs["settings"].get_dict()
        valid_keys = ("remote_symlink_files",)
        for key in settings:
            if key not in valid_keys:
                return f"Invalid settings: `{key}`, valid keys are: {valid_keys}"


def validate_inputs(inputs: AttributeDict, ctx=None) -> None:
    """Validate the inputs of the entire input namespace."""
    # pylint: disable=too-many-return-statements,unused-argument

    result = validate_inputs_base(inputs, ctx)
    if result:
        return result

    # pylint: disable=protected-access
    calc_inputs = AttributeDict(inputs[Wannier90BaseWorkChain._inputs_namespace])
    calc_parameters = calc_inputs.parameters.get_dict()

    # Check existence of `fermi_energy`
    if inputs["shift_energy_windows"]:
        if "fermi_energy" not in calc_parameters:
            return "`shift_energy_windows` is requested but no `fermi_energy` in input parameters"
        if "bands" not in inputs:
            return "`shift_energy_windows` is requested but no `bands` in inputs"

    # Check `bands`
    if any(_ in inputs for _ in ("bands", "bands_projections")) and all(
        _ not in inputs for _ in ("shift_energy_windows", "auto_energy_windows")
    ):
        return (
            "`bands` and/or `bands_projections` are provided but both `shift_energy_windows` "
            "and `auto_energy_windows` are False?"
        )

    # Check `auto_energy_windows`
    if inputs["auto_energy_windows"]:
        if inputs["shift_energy_windows"]:
            return "No need to shift energy windows when auto set energy windows"

        if any(_ not in inputs for _ in ("bands_projections", "bands")):
            return "`auto_energy_windows` is requested but `bands_projections` or `bands` is empty"

        # Check bands and bands_projections are consistent
        bands_num_kpoints, bands_num_bands = inputs["bands"].base.attributes.all[
            "array|bands"
        ]
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


class Wannier90BaseWorkChain(ProtocolMixin, BaseRestartWorkChain):
    """Workchain to run a `Wannier90Calculation` with automated error handling and restarts."""

    _process_class = Wannier90Calculation
    _inputs_namespace = "wannier90"

    _WANNIER90_DEFAULT_KMESH_TOL = 1e-6
    _WANNIER90_DEFAULT_DIS_PROJ_MIN = 0.1
    _WANNIER90_DEFAULT_DIS_PROJ_MAX = 0.9
    _WANNIER90_DEFAULT_WANNIER_PLOT_SUPERCELL = 2

    @classmethod
    def define(cls, spec) -> None:
        """Define the process spec."""
        from aiida_wannier90.calculations.wannier90 import (
            validate_inputs_base as validate_inputs_base_wannier90,
        )

        super().define(spec)

        spec.expose_inputs(Wannier90Calculation, namespace=cls._inputs_namespace)
        spec.inputs[cls._inputs_namespace]["metadata"]["options"][
            "resources"
        ].default = {
            "num_machines": 1,
            # 'num_mpiprocs_per_machine': 1,
        }

        spec.inputs["wannier90"].validator = validate_inputs_base_wannier90

        spec.input(
            "shift_energy_windows",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            serializer=to_aiida_type,
            help=(
                "If True the `dis_froz_min`, `dis_froz_max`, `dis_win_min`, `dis_win_max` will be shifted by "
                "Fermi enerngy. False is the default behaviour of wannier90."
            ),
        )
        spec.input(
            "auto_energy_windows",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            serializer=to_aiida_type,
            help=(
                "If True use the energy corresponding to projectability = auto_energy_windows_threshold "
                "as `dis_froz_max` for wannier90."
            ),
        )
        spec.input(
            "auto_energy_windows_threshold",
            valid_type=orm.Float,
            default=lambda: orm.Float(0.9),
            serializer=to_aiida_type,
            help="Threshold for auto_energy_windows.",
        )
        spec.input(
            "bands",
            valid_type=orm.BandsData,
            required=False,
            help=(
                "For shift_energy_windows, if provided the energy windows will be shifted by Fermi energy "
                "for metals or minimum of lowest-unoccupied bands for insulators. "
                "The bands should be along a kpath to better estimate the band gap. "
                "For auto_energy_windows, the bands is used to find out the energy corresponds to "
                "projectability = auto_energy_windows_threshold, the energy is used as `dis_froz_max`. "
                "In this case the bands should be on a nscf kmesh."
            ),
        )
        spec.input(
            "bands_projections",
            valid_type=orm.ProjectionData,
            required=False,
            help="Projectability of bands to auto set `dis_froz_max`.",
        )
        spec.input(
            "guiding_centres_projections",
            valid_type=(orm.OrbitalData, orm.Dict, orm.List),
            required=False,
            help="Projections block for `guiding_centres = True`.",
        )
        spec.input(
            "settings",
            valid_type=orm.Dict,
            required=False,
            serializer=to_aiida_type,
            help="Additional settings.",
        )
        spec.inputs.validator = validate_inputs

        spec.outline(
            cls.setup,
            while_(cls.should_run_process)(
                cls.run_process,
                cls.inspect_process,
            ),
            cls.results,
        )

        spec.expose_outputs(Wannier90Calculation)

        spec.exit_code(401, "ERROR_BVECTORS", message="Unrecoverable bvectors error.")
        spec.exit_code(
            402,
            "ERROR_DISENTANGLEMENT_NOT_ENOUGH_STATES",
            message="Unrecoverable disentanglement error.",
        )
        spec.exit_code(
            403, "ERROR_PLOT_WF_CUBE", message="Unrecoverable cube format error."
        )
        spec.exit_code(
            404,
            "ERROR_OUTPUT_STDOUT_INCOMPLETE",
            message="The stdout output file was incomplete probably because the calculation got interrupted.",
        )

    @classmethod
    def get_protocol_filepath(cls) -> pathlib.Path:
        """Return the ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files

        from .. import protocols

        return files(protocols) / "base" / "wannier90.yaml"

    @classmethod
    def get_builder_from_protocol(
        cls,
        code: ty.Union[orm.Code, str, int],
        *,
        structure: orm.StructureData,
        protocol: str = None,
        overrides: dict = None,
        pseudo_family: str = None,
        external_projectors: dict = None,
        electronic_type: ElectronicType = ElectronicType.METAL,
        spin_type: SpinType = SpinType.NONE,
        projection_type: WannierProjectionType = WannierProjectionType.ATOMIC_PROJECTORS_QE,
        disentanglement_type: WannierDisentanglementType = WannierDisentanglementType.SMV,
        frozen_type: WannierFrozenType = WannierFrozenType.FIXED_PLUS_PROJECTABILITY,
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
        # pylint: disable=too-many-statements,too-many-locals,too-many-branches
        from aiida_quantumespresso.workflows.protocols.utils import recursive_merge

        from aiida_wannier90_workflows.utils.kpoints import (
            create_kpoints_from_distance,
            get_explicit_kpoints,
        )
        from aiida_wannier90_workflows.utils.pseudo import (
            get_number_of_projections,
            get_number_of_projections_ext,
            get_pseudo_and_cutoff,
            get_pseudo_orbitals,
            get_semicore_list,
            get_semicore_list_ext,
            get_wannier_number_of_bands,
            get_wannier_number_of_bands_ext,
        )

        if isinstance(code, (int, str)):
            code = orm.load_code(code)

        type_check(code, orm.Code)
        type_check(structure, orm.StructureData)
        type_check(electronic_type, ElectronicType)
        type_check(spin_type, SpinType)
        type_check(projection_type, WannierProjectionType)
        type_check(disentanglement_type, WannierDisentanglementType)
        type_check(frozen_type, WannierFrozenType)

        if electronic_type in [ElectronicType.AUTOMATIC]:
            raise NotImplementedError("`ElectronicType.AUTOMATIC` not implemented")

        # if spin_type == SpinType.COLLINEAR:
        #     raise NotImplementedError("`SpinType.COLLINEAR` not implemented")
        if spin_type == SpinType.SPIN_ORBIT and pseudo_family is None:
            raise ValueError(
                "Need to explicitly specify `pseudo_family` for SOC calculation"
            )

        inputs = cls.get_protocol_inputs(protocol, overrides)

        # Update the parameters based on the protocol inputs
        parameters = inputs[cls._inputs_namespace]["parameters"]
        metadata = inputs[cls._inputs_namespace]["metadata"]

        meta_parameters = inputs.pop("meta_parameters")
        num_atoms = len(structure.sites)
        parameters["conv_tol"] = num_atoms * meta_parameters["conv_tol_per_atom"]
        parameters["dis_conv_tol"] = (
            num_atoms * meta_parameters["dis_conv_tol_per_atom"]
        )

        # Set `num_bands`, `num_wann`, also take care of semicore states
        only_valence = electronic_type == ElectronicType.INSULATOR
        spin_polarized = spin_type == SpinType.COLLINEAR
        spin_orbit_coupling = spin_type == SpinType.SPIN_ORBIT
        spin_non_collinear = spin_orbit_coupling or spin_type == SpinType.NON_COLLINEAR
        # need add check for non_collinear, nproj/numbands also should * 2
        # some check for spin_orbit_coupling should also work for non-collinear

        if pseudo_family is None:
            pseudo_family = meta_parameters["pseudo_family"]
        pseudos, _, _ = get_pseudo_and_cutoff(pseudo_family, structure)

        if external_projectors is not None:
            if projection_type in [
                WannierProjectionType.ATOMIC_PROJECTORS_QE,
                WannierProjectionType.RANDOM,
            ]:
                raise ValueError(
                    f"Can not specify `external_projectors` when using {projection_type}"
                )
            num_bands = get_wannier_number_of_bands_ext(
                structure=structure,
                pseudos=pseudos,
                external_projectors=external_projectors,
                factor=meta_parameters["num_bands_factor"],
                only_valence=only_valence,
                spin_polarized=spin_polarized,
                spin_non_collinear=spin_non_collinear,
                spin_orbit_coupling=spin_orbit_coupling,
            )
            num_projs = get_number_of_projections_ext(
                structure=structure,
                external_projectors=external_projectors,
                spin_non_collinear=spin_non_collinear,
                spin_orbit_coupling=spin_orbit_coupling,
            )
        else:  # external_projector is None
            if projection_type == WannierProjectionType.ATOMIC_PROJECTORS_EXTERNAL:
                raise ValueError(
                    f"Must specify `external_projectors` when using {projection_type}"
                )
            num_bands = get_wannier_number_of_bands(
                structure=structure,
                pseudos=pseudos,
                factor=meta_parameters["num_bands_factor"],
                only_valence=only_valence,
                spin_polarized=spin_polarized,
                spin_non_collinear=spin_non_collinear,
                spin_orbit_coupling=spin_orbit_coupling,
            )
            num_projs = get_number_of_projections(
                structure=structure,
                pseudos=pseudos,
                spin_non_collinear=spin_non_collinear,
                spin_orbit_coupling=spin_orbit_coupling,
            )

        if electronic_type == ElectronicType.INSULATOR:
            num_wann = num_bands
        else:
            num_wann = num_projs

        if meta_parameters["exclude_semicore"]:
            pseudo_orbitals = get_pseudo_orbitals(pseudos)
            if projection_type == WannierProjectionType.ATOMIC_PROJECTORS_EXTERNAL:
                semicore_list = get_semicore_list_ext(
                    structure, external_projectors, pseudo_orbitals, spin_non_collinear
                )
            elif (
                projection_type == WannierProjectionType.ANALYTIC
                and external_projectors is not None
            ):
                semicore_list = get_semicore_list_ext(
                    structure, external_projectors, pseudo_orbitals, spin_non_collinear
                )
            else:
                semicore_list = get_semicore_list(
                    structure, pseudo_orbitals, spin_non_collinear
                )
            num_excludes = len(semicore_list)
            # TODO I assume all the semicore bands are the lowest  # pylint: disable=fixme
            exclude_pswfcs = range(1, num_excludes + 1)
            if num_excludes != 0:
                parameters["exclude_bands"] = exclude_pswfcs
                num_wann -= num_excludes
                num_bands -= num_excludes

        if num_wann <= 0:
            raise ValueError(f"Wrong num_wann {num_wann}")
        parameters["num_wann"] = num_wann
        parameters["num_bands"] = num_bands

        # Spinor
        if spin_type in [SpinType.NON_COLLINEAR, SpinType.SPIN_ORBIT]:
            parameters["spinors"] = True

        # Set initial projections
        if projection_type in [
            WannierProjectionType.SCDM,
            WannierProjectionType.ATOMIC_PROJECTORS_QE,
            WannierProjectionType.ATOMIC_PROJECTORS_EXTERNAL,
        ]:
            parameters["auto_projections"] = True
        elif projection_type == WannierProjectionType.ANALYTIC:
            pseudo_orbitals = get_pseudo_orbitals(pseudos)
            projections = []
            if external_projectors is None:
                for kind in structure.kinds:
                    for orb in pseudo_orbitals[kind.symbol]["pswfcs"]:
                        if meta_parameters["exclude_semicore"]:
                            if orb in pseudo_orbitals[kind.symbol]["semicores"]:
                                continue
                        projections.append(f"{kind.name}:{orb[-1].lower()}")
            else:  # external_projectors is not None
                for kind in structure.kinds:
                    for orb in external_projectors[kind.symbol]:
                        if spin_orbit_coupling and orb.get("j", 0.0) < orb["l"]:
                            continue  # avoid repeated counting
                        if meta_parameters["exclude_semicore"]:
                            if (
                                orb["label"].upper()
                                in pseudo_orbitals[kind.symbol]["semicores"]
                            ):
                                continue
                        projections.append(f"{kind.name}:{orb['label'][-1].lower()}")
            inputs[cls._inputs_namespace]["projections"] = orm.List(list=projections)
        elif projection_type == WannierProjectionType.RANDOM:
            settings = inputs[cls._inputs_namespace].get("settings", {})
            settings.update({"random_projections": True})
            inputs[cls._inputs_namespace]["settings"] = settings
        else:
            raise ValueError(f"Unrecognized projection type {projection_type}")

        # Set disentanglement
        if disentanglement_type == WannierDisentanglementType.NONE:
            parameters["dis_num_iter"] = 0
            parameters.pop("dis_froz_min", None)
            parameters.pop("dis_froz_max", None)
            parameters.pop("dis_win_min", None)
            parameters.pop("dis_win_max", None)
        elif disentanglement_type == WannierDisentanglementType.SMV:
            if frozen_type == WannierFrozenType.ENERGY_FIXED:
                inputs["shift_energy_windows"] = True
                parameters.update(
                    {
                        # Here +2 means fermi_energy + 2 eV, however Fermi energy is calculated at runtime
                        # inside Wannier90WorkChain, so it will add Fermi energy with this
                        # dis_froz_max dynamically.
                        "dis_froz_max": +2.0,
                    }
                )
            elif frozen_type == WannierFrozenType.ENERGY_AUTO:
                # ENERGY_AUTO needs projectability, will be set dynamically when workchain is running
                inputs["auto_energy_windows"] = True
            elif frozen_type == WannierFrozenType.PROJECTABILITY:
                parameters.update(
                    {
                        "dis_proj_min": 0.01,
                        "dis_proj_max": 0.95,
                    }
                )
            elif frozen_type == WannierFrozenType.FIXED_PLUS_PROJECTABILITY:
                inputs["shift_energy_windows"] = True
                parameters.update(
                    {
                        "dis_proj_min": 0.01,
                        "dis_proj_max": 0.95,
                        "dis_froz_max": +2.0,  # relative to fermi_energy
                    }
                )
            elif frozen_type == WannierFrozenType.NONE:
                parameters.pop("dis_froz_min", None)
                parameters.pop("dis_froz_max", None)
                parameters.pop("dis_win_min", None)
                parameters.pop("dis_win_max", None)
            else:
                raise ValueError(f"Not supported frozen type: {frozen_type}")
        else:
            raise ValueError(
                f"Not supported disentanglement type: {disentanglement_type}"
            )

        # Set kpoints
        # If inputs.kpoints is a kmesh, mp_grid will be auto-set by `Wannier90Calculation`,
        # otherwise we need to set it manually. If use open_grid, kpoints will be set dynamically
        # after open_grid calculation.
        kpoints = create_kpoints_from_distance(
            structure,
            distance=meta_parameters["kpoints_distance"],
            force_parity=meta_parameters["kpoints_force_parity"],
        )
        inputs["kpoints"] = get_explicit_kpoints(kpoints)
        parameters["mp_grid"] = kpoints.get_kpoints_mesh()[0]

        # If overrides are provided, they take precedence over default values
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
        builder[cls._inputs_namespace]["structure"] = structure
        builder[cls._inputs_namespace]["kpoints"] = inputs["kpoints"]
        builder[cls._inputs_namespace]["parameters"] = orm.Dict(parameters)
        builder[cls._inputs_namespace]["metadata"] = metadata
        if "projections" in inputs[cls._inputs_namespace]:
            builder[cls._inputs_namespace]["projections"] = inputs[
                cls._inputs_namespace
            ]["projections"]
        if "settings" in inputs[cls._inputs_namespace]:
            builder[cls._inputs_namespace]["settings"] = orm.Dict(
                dict=inputs[cls._inputs_namespace]["settings"]
            )
        if "settings" in inputs:
            builder["settings"] = orm.Dict(inputs["settings"])
        builder["clean_workdir"] = orm.Bool(inputs["clean_workdir"])
        builder["shift_energy_windows"] = orm.Bool(inputs["shift_energy_windows"])
        builder["auto_energy_windows"] = orm.Bool(inputs["auto_energy_windows"])
        builder["auto_energy_windows_threshold"] = orm.Float(
            inputs["auto_energy_windows_threshold"]
        )
        # pylint: enable=no-member

        return builder

    def setup(self) -> None:
        """Call the `setup` of the `BaseRestartWorkChain` and then create the inputs dictionary in `self.ctx.inputs`.

        This `self.ctx.inputs` dictionary will be used by the `BaseRestartWorkChain` to submit
        the calculations in the internal loop.
        """
        super().setup()

        self.ctx.inputs = self.prepare_inputs()

        self.ctx.kmeshtol_new = [self._WANNIER90_DEFAULT_KMESH_TOL, 1e-8, 1e-4]
        self.ctx.disprojmin_multipliers = [0.5, 0.25, 0.125, 0]
        self.ctx.wannier_plot_supercell_new = [4, 6, 8, 10]

    def prepare_inputs(
        self,
    ) -> AttributeDict:
        """Prepare `Wannier90Calculation` inputs according to workchain input sepc.

        Different from `get_builder_from_protocol', this function is executed at runtime.
        """
        # pylint: disable=too-many-statements,too-many-locals,too-many-branches
        import numpy as np

        from aiida_wannier90_workflows.utils.bands import (
            get_homo_lumo,
            remove_exclude_bands,
        )
        from aiida_wannier90_workflows.utils.scdm import get_energy_of_projectability

        inputs = AttributeDict(
            self.exposed_inputs(Wannier90Calculation, self._inputs_namespace)
        )
        parameters = inputs.parameters.get_dict()

        if self.inputs.shift_energy_windows:
            fermi_energy = parameters["fermi_energy"]
            # For metal, we shift the four parameters by Fermi energy.
            shift_energy = fermi_energy
            if "bands" in self.inputs:
                # Check the system is metal or insulator.
                # For insulator, we shift them by the minimum of LUMO.
                bands = self.inputs.bands.get_bands()
                homo, lumo = get_homo_lumo(bands, fermi_energy)
                bandgap = lumo - homo
                if bandgap > 1e-3:
                    shift_energy = lumo

            keys = ("dis_froz_min", "dis_froz_max", "dis_win_min", "dis_win_max")
            for key in keys:
                if key in parameters:
                    parameters[key] += shift_energy

        # Auto set `dis_froz_max`
        if self.inputs.auto_energy_windows:
            dis_froz_max = get_energy_of_projectability(
                bands=self.inputs.bands,
                projections=self.inputs.bands_projections,
                thresholds=self.inputs.auto_energy_windows_threshold.value,
            )
            parameters["dis_froz_max"] = dis_froz_max

        # Prevent error:
        #   dis_windows: More states in the frozen window than target WFs
        if "dis_froz_max" in parameters and "bands" in self.inputs:
            bands = self.inputs.bands.get_bands()
            if parameters.get("exclude_bands", None):
                # Index of parameters['exclude_bands'] starts from 1,
                # I need to change it to 0-based
                exclude_bands = [_ - 1 for _ in parameters["exclude_bands"]]
                bands = remove_exclude_bands(bands=bands, exclude_bands=exclude_bands)
            highest_band = bands[:, min(parameters["num_wann"], bands.shape[1]) - 1]
            # This is the energy to freeze all the num_wann bands
            max_froz_energy = np.max(highest_band)
            # Cannot freeze more bands than num_wann,
            # this sets the upper limit of `dis_froz_max`.
            if bands.shape[1] > parameters["num_wann"]:
                min_next_band = np.min(bands[:, parameters["num_wann"]])
                # I subtract a small value for safety
                min_next_band -= 1e-4
                max_froz_energy = min(max_froz_energy, min_next_band)
            # `dis_froz_max` should be smaller than this max_froz_energy
            # to allow doing disentanglement
            dis_froz_max = min(max_froz_energy, parameters["dis_froz_max"])
            parameters["dis_froz_max"] = dis_froz_max

        # quick and hacky way to enable guiding_centres while still using
        # auto_projections = True in the nnkp step
        # - in nnkp generation step, we ignore guiding_centres totally, and
        #   use whatever user provided in inputs, can be analytic projections or
        #   auto_projections
        # - in Wannierization step, we check if guiding_centres_projections is
        #   provided or not: if yes, pop auto_projections from parameters and
        #   set guiding_centres = True, and set projections from guiding_centres_projections;
        #   if not, we change nothing.
        w90_settings = self.inputs.wannier90.get("settings", orm.Dict()).get_dict()
        postproc_setup = w90_settings.get("postproc_setup", False)
        if not postproc_setup and "guiding_centres_projections" in self.inputs:
            parameters.pop("auto_projections", None)
            parameters["guiding_centres"] = True
            inputs.projections = self.inputs.guiding_centres_projections

        inputs.parameters = orm.Dict(parameters)

        if "remote_input_folder" in inputs and "settings" in self.inputs:
            # Note there is an `additional_remote_symlink_list` in Wannier90Calculation.inputs.settings,
            # however it requires user providing a list of
            #   (computer_uuid, remote_input_folder_abs_path, dest_path)
            # This is impossible if we launch a Wannier90Calculation inside a workflow since we don't
            # know the remote_input_folder when setting the inputs of the workflow.
            # Thus I add an `inputs.settings['remote_symlink_files']` to Wannier90BaseWorkChain,
            # which only accepts a list of filenames and generate the full
            # `additional_remote_symlink_list` here.
            remote_input_folder = inputs["remote_input_folder"]
            remote_input_folder_path = pathlib.Path(
                remote_input_folder.get_remote_path()
            )
            workflow_settings = self.inputs.settings.get_dict()
            if "settings" in inputs:
                calc_settings = inputs.settings.get_dict()
            else:
                calc_settings = {}
            remote_symlink_list = calc_settings.get(
                "additional_remote_symlink_list", []
            )
            existed_symlinks = [_[-1] for _ in remote_symlink_list]
            for filename in workflow_settings.get("remote_symlink_files", []):
                if filename in existed_symlinks:
                    continue
                remote_symlink_list.append(
                    (
                        remote_input_folder.computer.uuid,
                        str(remote_input_folder_path / filename),
                        filename,
                    )
                )
            calc_settings["additional_remote_symlink_list"] = remote_symlink_list
            inputs.settings = orm.Dict(calc_settings)

        return inputs

    def report_error_handled(self, calculation, action) -> None:
        """Report an action taken for a calculation that has failed.

        This should be called in a registered error handler if its condition is met and an action was taken.

        :param calculation: the failed calculation node
        :param action: a string message with the action taken
        """
        message = f"{calculation.process_label}<{calculation.pk}> failed"
        message += (
            f" with exit status {calculation.exit_status}: {calculation.exit_message}"
        )
        self.report(message)
        self.report(f"Action taken: {action}")

    @process_handler(exit_codes=[Wannier90Calculation.exit_codes.ERROR_BVECTORS])
    def handle_bvectors(self, calculation) -> ProcessHandlerReport:
        """Try to fix Wannier90 bvectors errors by tunning `kmesh_tol`.

        The handler will try to use kmesh_tol = 1e-6, 1e-8, 1e-4.
        """
        parameters = self.ctx.inputs.parameters.get_dict()

        # If the user has specified `kmesh_tol` in the input parameters and it is different
        # from the default, we will first try to use the default `kmesh_tol`.
        current_kmeshtol = parameters.get(
            "kmesh_tol", self._WANNIER90_DEFAULT_KMESH_TOL
        )
        if current_kmeshtol in self.ctx.kmeshtol_new:
            self.ctx.kmeshtol_new.remove(current_kmeshtol)

        if len(self.ctx.kmeshtol_new) == 0:
            action = "Unrecoverable bvectors error after several trials of kmesh_tol"
            self.report_error_handled(calculation, action)
            return ProcessHandlerReport(True, self.exit_codes.ERROR_BVECTORS)

        new_kmeshtol = self.ctx.kmeshtol_new.pop(0)
        parameters["kmesh_tol"] = new_kmeshtol
        action = f"Bvectors error, current kmesh_tol = {current_kmeshtol}, new kmesh_tol = {new_kmeshtol}"
        self.report_error_handled(calculation, action)
        self.ctx.inputs.parameters = orm.Dict(parameters)

        return ProcessHandlerReport(True)

    @process_handler(
        exit_codes=[
            Wannier90Calculation.exit_codes.ERROR_DISENTANGLEMENT_NOT_ENOUGH_STATES
        ]
    )
    def handle_disentanglement_not_enough_states(
        self, calculation
    ) -> ProcessHandlerReport:
        """Try to fix Wannier90 wout error message related to projectability disentanglement.

        The error message is: 'Energy window contains fewer states than number of target WFs,
        consider reducing dis_proj_min/increasing dis_win_max?'.

        The handler will try to use decrease 'dis_proj_min' to allow for more states for disentanglement.
        """
        parameters = self.ctx.inputs.parameters.get_dict()
        if "dis_proj_min" not in parameters and "dis_proj_max" not in parameters:
            # If neither is present, I should never encounter this exit_code
            action = "Unrecoverable bvectors error: the error handler is only for projectability disentanglement"
            self.report_error_handled(calculation, action)
            return ProcessHandlerReport(
                True, self.exit_codes.ERROR_DISENTANGLEMENT_NOT_ENOUGH_STATES
            )

        if len(self.ctx.disprojmin_multipliers) == 0:
            action = "Unrecoverable error after several trials of dis_proj_min"
            self.report_error_handled(calculation, action)
            return ProcessHandlerReport(
                True, self.exit_codes.ERROR_DISENTANGLEMENT_NOT_ENOUGH_STATES
            )

        current_disprojmin = parameters.get(
            "dis_proj_min", self._WANNIER90_DEFAULT_KMESH_TOL
        )
        multiplier = self.ctx.disprojmin_multipliers.pop(0)
        new_disprojmin = current_disprojmin * multiplier
        parameters["dis_proj_min"] = new_disprojmin

        action = "Not enough states for disentanglement, "
        action += f"current dis_proj_min = {current_disprojmin}, new dis_proj_min = {new_disprojmin}"
        self.report_error_handled(calculation, action)
        self.ctx.inputs.parameters = orm.Dict(parameters)

        return ProcessHandlerReport(True)

    @process_handler(exit_codes=[Wannier90Calculation.exit_codes.ERROR_PLOT_WF_CUBE])
    def handle_plot_wf_cube(self, calculation) -> ProcessHandlerReport:
        """Try to fix Wannier90 wout error message related to cube format.

        The error message is: 'Error plotting WF cube. Try one of the following:'.

        The handler will try to increase 'wannier_plot_supercell'.
        """
        parameters = self.ctx.inputs.parameters.get_dict()

        current_supercell = parameters.get(
            "wannier_plot_supercell", self._WANNIER90_DEFAULT_WANNIER_PLOT_SUPERCELL
        )
        # Remove sizes which are smaller equal than current supercell size
        self.ctx.wannier_plot_supercell_new = [
            _ for _ in self.ctx.wannier_plot_supercell_new if _ > current_supercell
        ]

        if len(self.ctx.wannier_plot_supercell_new) == 0:
            action = (
                "Unrecoverable error after several trials of wannier_plot_supercell"
            )
            self.report_error_handled(calculation, action)
            return ProcessHandlerReport(True, self.exit_codes.ERROR_PLOT_WF_CUBE)

        new_supercell = self.ctx.wannier_plot_supercell_new.pop(0)
        parameters["wannier_plot_supercell"] = new_supercell

        action = "Error plotting WFs in cube format, "
        action += f"current wannier_plot_supercell = {current_supercell}, new wannier_plot_supercell = {new_supercell}"
        self.report_error_handled(calculation, action)
        self.ctx.inputs.parameters = orm.Dict(parameters)

        return ProcessHandlerReport(True)

    @process_handler(
        exit_codes=[Wannier90Calculation.exit_codes.ERROR_OUTPUT_STDOUT_INCOMPLETE]
    )
    def handle_output_stdout_incomplete(self, calculation) -> ProcessHandlerReport:
        """Try to fix incomplete stdout error by reducing the number of cores.

        Often the ERROR_OUTPUT_STDOUT_INCOMPLETE is due to out-of-memory.
        The handler will try to set `num_mpiprocs_per_machine` to 1.
        """
        import re

        regex_oom = re.compile(r"Detected \d+ oom-kill event\(s\) in step")
        regex_bus = re.compile(
            r"Program received signal SIGBUS: Access to an undefined portion of a memory object"
        )
        # srun: error: nid002144: tasks 0-63: Killed
        regex_killed = re.compile(r"srun: error: \w+: tasks? [\w\-,]+: Killed")
        scheduler_stderr = calculation.get_scheduler_stderr()
        for line in scheduler_stderr.split("\n"):
            if (
                regex_oom.search(line)
                or regex_bus.search(line)
                or regex_killed.search(line)
                or "Out Of Memory" in line
            ):
                break
        else:
            action = "Unrecoverable incomplete stdout error"
            self.report_error_handled(calculation, action)
            return ProcessHandlerReport(
                True, self.exit_codes.ERROR_OUTPUT_STDOUT_INCOMPLETE
            )

        metadata = self.ctx.inputs["metadata"]
        current_num_mpiprocs_per_machine = metadata["options"]["resources"].get(
            "num_mpiprocs_per_machine", 1
        )
        # num_mpiprocs_per_machine = calculation.base.attributes.all['resources'].get('num_mpiprocs_per_machine', 1)

        if current_num_mpiprocs_per_machine == 1:
            action = "Unrecoverable out-of-memory error after setting num_mpiprocs_per_machine to 1"
            self.report_error_handled(calculation, action)
            return ProcessHandlerReport(
                True, self.exit_codes.ERROR_OUTPUT_STDOUT_INCOMPLETE
            )

        new_num_mpiprocs_per_machine = current_num_mpiprocs_per_machine // 2
        metadata["options"]["resources"][
            "num_mpiprocs_per_machine"
        ] = new_num_mpiprocs_per_machine
        action = f"Out-of-memory error, current num_mpiprocs_per_machine = {current_num_mpiprocs_per_machine}"
        action += f", new num_mpiprocs_per_machine = {new_num_mpiprocs_per_machine}"
        self.report_error_handled(calculation, action)
        self.ctx.inputs["metadata"] = metadata

        return ProcessHandlerReport(True)
