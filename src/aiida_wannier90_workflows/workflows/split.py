"""Workchain to automatically optimize dis_proj_min/max for projectability disentanglement."""

import copy
import pathlib
import typing as ty

import numpy as np

from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import ProcessBuilder, ToContext, WorkChain, if_
from aiida.orm.nodes.data.base import to_aiida_type

from aiida_quantumespresso.common.types import ElectronicType, SpinType
from aiida_quantumespresso.utils.mapping import prepare_process_inputs

from aiida_wannier90_workflows.calculations.split import Wannier90SplitCalculation
from aiida_wannier90_workflows.common.types import (
    WannierDisentanglementType,
    WannierFrozenType,
    WannierProjectionType,
)
from aiida_wannier90_workflows.utils.bands import get_homo_lumo
from aiida_wannier90_workflows.utils.workflows.plot.bands import (
    get_workchain_fermi_energy,
)
from aiida_wannier90_workflows.workflows.base.wannier90 import Wannier90BaseWorkChain
from aiida_wannier90_workflows.workflows.optimize import Wannier90OptimizeWorkChain

__all__ = ["validate_inputs", "Wannier90SplitWorkChain"]


def validate_inputs(  # pylint: disable=unused-argument,inconsistent-return-statements
    inputs, ctx=None
):
    """Validate the inputs of the entire input namespace of `Wannier90OptimizeWorkChain`."""

    parameters = inputs["valcond"]["wannier90"]["wannier90"]["parameters"].get_dict()
    num_wann = parameters["num_wann"]

    num_val = inputs["split"].get("num_val", None)

    if num_val is not None:
        if num_val <= 0 or num_val >= num_wann:
            return f"num_valence must be between 0 and {num_wann=}"


class Wannier90SplitWorkChain(WorkChain):  # pylint: disable=too-many-public-methods
    """Workchain to split valence+conduction into two calculations for val and cond, respectively."""

    @classmethod
    def define(cls, spec):
        """Define the process spec."""
        super().define(spec)

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
            Wannier90OptimizeWorkChain,
            namespace="valcond",
            exclude=("clean_workdir",),
            namespace_options={
                "required": False,
                "populate_defaults": False,
                "help": "Inputs for the `Wannier90OptimizeWorkChain` for the val+cond Wannierization.",
            },
        )
        spec.expose_inputs(
            Wannier90SplitCalculation,
            namespace="split",
            exclude=("clean_workdir",),
            namespace_options={
                "required": False,
                "populate_defaults": False,
                "help": "Inputs for the `Wannier90SplitCalculation`.",
            },
        )
        # Make it optional, can be calculated on the fly.
        spec.input(
            "split.num_val",
            valid_type=orm.Int,
            required=False,
            serializer=orm.to_aiida_type,
            help="Number of valence WFs.",
        )
        spec.expose_inputs(
            Wannier90BaseWorkChain,
            namespace="val",
            exclude=("clean_workdir",),
            namespace_options={
                "required": False,
                "populate_defaults": False,
                "help": "Inputs for the `Wannier90BaseWorkChain` for the val Wannierization.",
            },
        )
        spec.expose_inputs(
            Wannier90BaseWorkChain,
            namespace="cond",
            exclude=("clean_workdir",),
            namespace_options={
                "required": False,
                "populate_defaults": False,
                "help": "Inputs for the `Wannier90BaseWorkChain` for the cond Wannierization.",
            },
        )
        spec.inputs.validator = validate_inputs

        spec.outline(
            cls.setup,
            if_(cls.should_run_valcond)(
                cls.run_valcond,
                cls.inspect_valcond,
            ),
            if_(cls.should_run_split)(
                cls.run_split,
                cls.inspect_split,
            ),
            if_(cls.should_run_val)(
                cls.run_val,
                cls.inspect_val,
            ),
            if_(cls.should_run_cond)(
                cls.run_cond,
                cls.inspect_cond,
            ),
            cls.results,
        )

        spec.expose_outputs(
            Wannier90OptimizeWorkChain,
            namespace="valcond",
            namespace_options={"required": False},
        )
        spec.expose_outputs(
            Wannier90SplitCalculation,
            namespace="split",
            namespace_options={"required": False},
        )
        spec.expose_outputs(
            Wannier90BaseWorkChain,
            namespace="val",
            namespace_options={"required": False},
        )
        spec.expose_outputs(
            Wannier90BaseWorkChain,
            namespace="cond",
            namespace_options={"required": False},
        )

        spec.output(
            "bands_distance.valcond_to_ref",
            valid_type=orm.Float,
            required=False,
            help="Bands distances between reference bands and val+cond Wannier interpolated bands.",
        )
        spec.output(
            "bands_distance.val_to_ref",
            valid_type=orm.Float,
            required=False,
            help="Bands distances between reference bands and val Wannier interpolated bands.",
        )
        spec.output(
            "bands_distance.val_to_valcond",
            valid_type=orm.Float,
            required=False,
            help="Bands distances between val+cond Wannier bands and val Wannier bands.",
        )
        spec.output(
            "bands_distance.cond_to_ref",
            valid_type=orm.Float,
            required=False,
            help="Bands distances between reference bands and cond Wannier interpolated bands.",
        )
        spec.output(
            "bands_distance.cond_to_valcond",
            valid_type=orm.Float,
            required=False,
            help="Bands distances between val+cond Wannier bands and cond Wannier bands.",
        )

        spec.exit_code(
            500,
            "ERROR_SUB_PROCESS_FAILED_VALCOND",
            message="All the trials on dis_proj_min/max have failed, cannot compare bands distance",
        )
        spec.exit_code(
            501,
            "ERROR_SUB_PROCESS_FAILED_SPLIT",
            message="the split calculation has failed",
        )
        spec.exit_code(
            502,
            "ERROR_SUB_PROCESS_FAILED_VAL",
            message="the valence Wannier90Calculation has failed",
        )
        spec.exit_code(
            503,
            "ERROR_SUB_PROCESS_FAILED_COND",
            message="the conduction Wannier90Calculation has failed",
        )
        spec.exit_code(
            520,
            "ERROR_SUB_PROCESS_FAILED_ALL_OCCUPIED",
            message="All the orbitals are occupied, cannot split into valence and conduction",
        )

    @classmethod
    def get_protocol_filepath(cls) -> pathlib.Path:
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files

        from . import protocols

        return files(protocols) / "split.yaml"

    @classmethod
    def get_builder_from_protocol(  # pylint: disable=too-many-locals,too-many-statements
        cls,
        codes: ty.Mapping[str, ty.Union[str, int, orm.Code]],
        structure: orm.StructureData,
        *,
        num_val: ty.Optional[int] = None,
        bands_distance_threshold: float = 1e-2,  # unit is eV
        **kwargs,
    ) -> ProcessBuilder:
        """Return a builder prepopulated with inputs selected according to the specified arguments.

        :params: num_val: number of valence electrons, default is num_wann/2
        :return: [description]
        :rtype: ProcessBuilder
        """
        from aiida_wannier90_workflows.utils.workflows.builder.submit import (
            recursive_merge_builder,
        )

        projection_type = kwargs.pop("projection_type", None)
        bands_kpoints = kwargs.pop("bands_kpoints", None)
        plot_wannier_functions = kwargs.pop("plot_wannier_functions", False)
        reference_bands = kwargs.pop("reference_bands", None)
        exclude_semicore = kwargs.pop("exclude_semicore", True)

        # Prepare workchain builder
        valcond_builder = Wannier90OptimizeWorkChain.get_builder_from_protocol(
            codes,
            structure,
            reference_bands=reference_bands,
            bands_distance_threshold=bands_distance_threshold,
            projection_type=projection_type,
            bands_kpoints=bands_kpoints,
            plot_wannier_functions=plot_wannier_functions,
            exclude_semicore=exclude_semicore,
            **kwargs,
        )
        valcond_inputs = valcond_builder._inputs(  # pylint: disable=protected-access
            prune=True
        )

        parameters = valcond_inputs["pw2wannier90"]["pw2wannier90"][
            "parameters"
        ].get_dict()
        parameters["inputpp"]["wvfn_formatted"] = True
        parameters["inputpp"]["spn_formatted"] = True
        valcond_inputs["pw2wannier90"]["pw2wannier90"]["parameters"] = orm.Dict(
            dict=parameters
        )

        parameters = valcond_inputs["wannier90"]["wannier90"]["parameters"].get_dict()
        parameters["wvfn_formatted"] = True
        parameters["spn_formatted"] = True
        valcond_inputs["wannier90"]["wannier90"]["parameters"] = orm.Dict(
            dict=parameters
        )

        num_wann = parameters["num_wann"]

        val_builder = Wannier90BaseWorkChain.get_builder_from_protocol(  # pylint: disable=too-many-function-args
            codes["wannier90"],
            structure=structure,
            electronic_type=ElectronicType.INSULATOR,
            spin_type=SpinType.NONE,
            projection_type=WannierProjectionType.RANDOM,  # no need projection
            disentanglement_type=WannierDisentanglementType.NONE,
            frozen_type=WannierFrozenType.NONE,
            **kwargs,
        )
        val_inputs = val_builder._inputs(prune=True)  # pylint: disable=protected-access
        parameters = val_inputs["wannier90"]["parameters"].get_dict()
        if num_val is None:
            parameters.pop("num_wann", None)
        else:
            parameters["num_wann"] = num_val
        parameters.pop("num_bands", None)
        parameters.pop("guiding_centres", None)
        parameters.pop("exclude_bands", None)

        # stricter convergence
        # parameters["conv_tol"] = 1e-9
        # parameters["num_iter"] = 5000

        parameters["bands_plot"] = True
        val_inputs["wannier90"]["bands_kpoints"] = bands_kpoints

        # Plot WF
        if plot_wannier_functions:
            parameters["wannier_plot"] = True

        parameters["wvfn_formatted"] = True
        parameters["spn_formatted"] = True

        val_inputs["wannier90"]["parameters"] = orm.Dict(parameters)

        # explicit shallow copy, deepcopy will duplicate orm.Code and
        # aiida engine will store them as new code!
        cond_inputs = copy.copy(val_inputs)
        parameters = cond_inputs["wannier90"]["parameters"].get_dict()
        if num_val is None:
            parameters.pop("num_wann", None)
        else:
            parameters["num_wann"] = num_wann - num_val
        cond_inputs["wannier90"]["parameters"] = orm.Dict(parameters)

        split_inputs = {
            "code": codes["split"],
        }
        if num_val is not None:
            split_inputs["num_val"] = num_val
        if plot_wannier_functions:
            split_inputs["rotate_unk"] = True

        valcond_inputs.pop("clean_workdir", None)
        val_inputs.pop("clean_workdir", None)
        cond_inputs.pop("clean_workdir", None)
        split_inputs.pop("clean_workdir", None)

        inputs = {
            "valcond": valcond_inputs,
            "split": split_inputs,
            "val": val_inputs,
            "cond": cond_inputs,
        }

        builder = cls.get_builder()
        builder = recursive_merge_builder(builder, inputs)

        return builder

    def setup(self):
        """Define the current structure in the context to be the input structure."""

        self.ctx.ref_bands = self.inputs["valcond"].get(
            "optimize_reference_bands", None
        )

    def should_run_valcond(self):
        """Whether should optimize dis_proj_min/max."""
        if "valcond" in self.inputs:
            return True

        return False

    def run_valcond(self):
        """Run val+cond Wannier90."""
        inputs = AttributeDict(self.inputs["valcond"])

        metadata = inputs.get("metadata", {})
        metadata["call_link_label"] = "valcond"
        inputs["metadata"] = metadata
        inputs = prepare_process_inputs(Wannier90OptimizeWorkChain, inputs)
        running = self.submit(Wannier90OptimizeWorkChain, **inputs)
        self.report(f"launching {running.process_label}<{running.pk}>")

        return ToContext(workchain_valcond=running)

    def inspect_valcond(self):  # pylint: disable=inconsistent-return-statements
        """Overide parent."""

        workchain = self.ctx.workchain_valcond

        if not workchain.is_finished_ok:
            self.report(
                f"{workchain.process_label} failed with exit status {workchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_VALCOND

        if "primitive_structure" in workchain.outputs:
            self.ctx.current_structure = workchain.outputs.primitive_structure

        fermi_energy = get_workchain_fermi_energy(workchain)
        self.ctx.fermi_energy = fermi_energy

        # Prioritize reference bands, since Wannier-interpolated bands might have
        # interpolation error
        if self.ctx.ref_bands is None:
            bands: orm.BandsData = workchain.outputs.band_structure
        else:
            bands: orm.BandsData = self.ctx.ref_bands
        bands_arr = bands.get_bands()
        try:
            homo, lumo = get_homo_lumo(bands_arr, fermi_energy)
        except ValueError as exc:
            self.report(f"{exc}")
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_ALL_OCCUPIED
        band_gap = lumo - homo
        gap_threshold = 1e-2
        if band_gap < gap_threshold:
            self.report(f"{homo=}, {lumo=}, {band_gap=}, seems metal?")

        num_wann = workchain.inputs.wannier90.wannier90.parameters["num_wann"]
        num_val = np.count_nonzero(np.all(bands_arr <= fermi_energy, axis=0))
        self.ctx.num_val = num_val
        self.ctx.num_cond = num_wann - num_val

    def should_run_split(self):
        """Whether to run split val cond."""
        if "valcond" in self.inputs:
            return True

        return False

    def prepare_split_inputs(self):
        """Prepare inputs for split run."""
        inputs = AttributeDict(self.inputs["split"])

        if self.should_run_valcond():
            workchain_valcond = self.ctx.workchain_valcond
            if workchain_valcond.inputs["separate_plotting"]:
                parent_folder = workchain_valcond.outputs["wannier90_plot"][
                    "remote_folder"
                ]
            else:
                if workchain_valcond.inputs["optimize_disproj"]:
                    parent_folder = workchain_valcond.outputs["wannier90_optimal"][
                        "remote_folder"
                    ]
                else:
                    parent_folder = workchain_valcond.outputs["wannier90"][
                        "remote_folder"
                    ]
            inputs["parent_folder"] = parent_folder

            if "num_val" in inputs:
                input_num_val = inputs["num_val"].value
                if input_num_val != self.ctx.num_val:
                    self.report(
                        f"input num_val={input_num_val} != num_val={self.ctx.num_val}"
                        "from val+cond band structure"
                    )
            else:
                inputs["num_val"] = orm.Int(self.ctx.num_val)

        return inputs

    def run_split(self):
        """Run split."""
        inputs = self.prepare_split_inputs()

        metadata = inputs.get("metadata", {})
        metadata["call_link_label"] = "split"
        inputs["metadata"] = metadata
        inputs = prepare_process_inputs(Wannier90SplitCalculation, inputs)
        running = self.submit(Wannier90SplitCalculation, **inputs)
        self.report(f"launching {running.process_label}<{running.pk}>")

        return ToContext(workchain_split=running)

    def inspect_split(self):  # pylint: disable=inconsistent-return-statements
        """Inspect."""

        workchain = self.ctx.workchain_split

        if not workchain.is_finished_ok:
            self.report(
                f"{workchain.process_label} failed with exit status {workchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_SPLIT

    def should_run_val(self):
        """Whether to run val."""
        if "val" in self.inputs:
            return True

        return False

    def prepare_val_inputs(self):
        """Prepare inputs for val run."""
        base_inputs = AttributeDict(self.inputs["val"])
        inputs = base_inputs["wannier90"]

        if "current_structure" in self.ctx:
            inputs["structure"] = self.ctx["current_structure"]

        if self.should_run_split():
            inputs["remote_input_folder"] = (
                self.ctx.workchain_split.outputs.remote_folder_val
            )

        if self.should_run_valcond():
            parameters = inputs["parameters"].get_dict()
            parameters["fermi_energy"] = self.ctx.fermi_energy

            if "num_wann" in parameters:
                input_num_wann = parameters["num_wann"]
                if input_num_wann != self.ctx.num_val:
                    self.report(
                        f"input num_wann={input_num_wann} "
                        f"!= num_val={self.ctx.num_val} "
                        "from val+cond band structure?"
                    )
            else:
                parameters["num_wann"] = self.ctx.num_val

            inputs["parameters"] = orm.Dict(parameters)

        base_inputs["wannier90"] = inputs
        base_inputs["clean_workdir"] = orm.Bool(False)

        return base_inputs

    def run_val(self):
        """Run valence."""
        inputs = self.prepare_val_inputs()

        metadata = inputs.get("metadata", {})
        metadata["call_link_label"] = "val"
        inputs["metadata"] = metadata
        inputs = prepare_process_inputs(Wannier90BaseWorkChain, inputs)
        running = self.submit(Wannier90BaseWorkChain, **inputs)
        self.report(f"launching {running.process_label}<{running.pk}>")

        return ToContext(workchain_val=running)

    def inspect_val(self):  # pylint: disable=inconsistent-return-statements
        """Inspect valence."""
        workchain = self.ctx.workchain_val
        if self.ctx.ref_bands is not None:
            bandsdist = self._get_bands_distance(
                self.ctx.ref_bands,
                workchain.outputs["interpolated_bands"],
                is_val=True,
                is_ref_dft=True,
            )
            self.ctx.bandsdist_val_to_ref = bandsdist
        if self.should_run_valcond():
            bandsdist = self._get_bands_distance(
                self.ctx.workchain_valcond.outputs.band_structure,
                workchain.outputs["interpolated_bands"],
                is_val=True,
                is_ref_dft=False,
            )
            self.ctx.bandsdist_val_to_valcond = bandsdist

        if not workchain.is_finished_ok:
            self.report(
                f"{workchain.process_label} failed with exit status {workchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_VAL

    def should_run_cond(self):
        """Whether to run cond."""
        if "cond" in self.inputs:
            return True

        return False

    def prepare_cond_inputs(self):
        """Prepare inputs for cond run."""
        base_inputs = AttributeDict(self.inputs["cond"])
        inputs = base_inputs["wannier90"]

        if "current_structure" in self.ctx:
            inputs["structure"] = self.ctx["current_structure"]

        if self.should_run_split():
            inputs["remote_input_folder"] = (
                self.ctx.workchain_split.outputs.remote_folder_cond
            )

        if self.should_run_valcond():
            parameters = inputs["parameters"].get_dict()
            parameters["fermi_energy"] = self.ctx.fermi_energy

            if "num_wann" in parameters:
                input_num_wann = parameters["num_wann"]
                if input_num_wann != self.ctx.num_cond:
                    self.report(
                        f"input num_wann={input_num_wann} "
                        f"!= num_cond={self.ctx.num_cond} "
                        "from val+cond band structure?"
                    )
            else:
                parameters["num_wann"] = self.ctx.num_cond

            inputs["parameters"] = orm.Dict(parameters)

        base_inputs["wannier90"] = inputs
        base_inputs["clean_workdir"] = orm.Bool(False)

        return base_inputs

    def run_cond(self):
        """Run cond."""
        inputs = self.prepare_cond_inputs()

        metadata = inputs.get("metadata", {})
        metadata["call_link_label"] = "cond"
        inputs["metadata"] = metadata
        inputs = prepare_process_inputs(Wannier90BaseWorkChain, inputs)
        running = self.submit(Wannier90BaseWorkChain, **inputs)
        self.report(f"launching {running.process_label}<{running.pk}>")

        return ToContext(workchain_cond=running)

    def inspect_cond(self):  # pylint: disable=inconsistent-return-statements
        """Inspect cond."""

        workchain = self.ctx.workchain_cond
        if self.ctx.ref_bands is not None:
            bandsdist = self._get_bands_distance(
                self.ctx.ref_bands,
                workchain.outputs["interpolated_bands"],
                is_val=False,
                is_ref_dft=True,
            )
            self.ctx.bandsdist_cond_to_ref = bandsdist
        if self.should_run_valcond():
            bandsdist = self._get_bands_distance(
                self.ctx.workchain_valcond.outputs.band_structure,
                workchain.outputs["interpolated_bands"],
                is_val=False,
                is_ref_dft=False,
            )
            self.ctx.bandsdist_cond_to_valcond = bandsdist

        if not workchain.is_finished_ok:
            self.report(
                f"{workchain.process_label} failed with exit status {workchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_COND

    def results(self):
        """Attach the relevant output nodes from the band calculation to the workchain outputs for convenience."""

        if self.should_run_valcond():
            self.out_many(
                self.exposed_outputs(
                    self.ctx.workchain_valcond,
                    Wannier90OptimizeWorkChain,
                    namespace="valcond",
                )
            )
            if "bands_distance" in self.outputs["valcond"]:
                bands_distance = self.outputs["valcond"]["bands_distance"]
                self.out("bands_distance.valcond_to_ref", bands_distance)

        if self.should_run_split():
            self.out_many(
                self.exposed_outputs(
                    self.ctx.workchain_split,
                    Wannier90SplitCalculation,
                    namespace="split",
                )
            )

        if self.should_run_val():
            self.out_many(
                self.exposed_outputs(
                    self.ctx.workchain_val,
                    Wannier90BaseWorkChain,
                    namespace="val",
                )
            )
            if "bandsdist_val_to_ref" in self.ctx:
                bands_distance = orm.Float(self.ctx["bandsdist_val_to_ref"])
                bands_distance.store()
                self.out("bands_distance.val_to_ref", bands_distance)
            if "bandsdist_val_to_valcond" in self.ctx:
                bands_distance = orm.Float(self.ctx["bandsdist_val_to_valcond"])
                bands_distance.store()
                self.out("bands_distance.val_to_valcond", bands_distance)

        if self.should_run_cond():
            self.out_many(
                self.exposed_outputs(
                    self.ctx.workchain_cond,
                    Wannier90BaseWorkChain,
                    namespace="cond",
                )
            )
            if "bandsdist_cond_to_ref" in self.ctx:
                bands_distance = orm.Float(self.ctx["bandsdist_cond_to_ref"])
                bands_distance.store()
                self.out("bands_distance.cond_to_ref", bands_distance)
            if "bandsdist_cond_to_valcond" in self.ctx:
                bands_distance = orm.Float(self.ctx["bandsdist_cond_to_valcond"])
                bands_distance.store()
                self.out("bands_distance.cond_to_valcond", bands_distance)

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

    def _get_bands_distance_consts(self) -> ty.Tuple[int, int, float]:
        """Get the num_semicore, num_val, fermi_energy."""

        wan_valcond_workchain = self.ctx.workchain_valcond
        w90_calc = wan_valcond_workchain.outputs.band_structure.creator
        wan_valcond_params = w90_calc.inputs["parameters"].get_dict()

        fermi_energy = wan_valcond_params.get("fermi_energy")

        exclude_list_dft = wan_valcond_params.get("exclude_bands", [])
        num_semicore = len(exclude_list_dft)

        num_val = self.ctx.workchain_split.inputs["num_val"].value

        return num_semicore, num_val, fermi_energy

    def _get_bands_distance(
        self,
        ref_bands: orm.BandsData,
        cmp_bands: orm.BandsData,
        is_val: bool,
        is_ref_dft: bool,
    ) -> float:
        num_semicore, num_val, fermi_energy = self._get_bands_distance_consts()
        # self.report(
        #     f"Computing bands distance for {ref_bands.pk} and {cmp_bands.pk}, "
        #     f"{is_val=} {is_ref_dft=}, {num_semicore=} {num_val=} {fermi_energy=}"
        # )

        return _get_bands_distance_raw(
            ref_bands,
            cmp_bands,
            is_val,
            is_ref_dft,
            num_semicore=num_semicore,
            num_val=num_val,
            fermi_energy=fermi_energy,
        )


def _get_bands_distance_raw(
    ref_bands: orm.BandsData,
    cmp_bands: orm.BandsData,
    is_val: bool,
    is_ref_dft: bool,
    *,
    num_semicore: int,
    num_val: int,
    fermi_energy: float,
) -> dict:
    """Get bands distance for isolated or Fermi energy + 2eV.

    For valence wan_bands, always compute eta_isolated: val_to_valcond, val_to_ref
    For conduction wan_bands, compute eta_2: cond_to_ref, compute eta_isolated: cond_to_valcond
    For valence+conduction wan_bands, compute eta_2: valcond_to_ref

    :param is_val: calc valence or conduction
    :param isolated: calc Ef+2 or isolated bands
    :param num_semicore: number of semicore bands in the DFT calculation
    :param num_val: number of valence bands (excluding semicores) in the DFT calculation
    :param fermi_energy: Fermi energy of the DFT calculation
    """
    from aiida_wannier90_workflows.utils.bands.distance import (
        bands_distance,
        bands_distance_isolated,
    )

    if is_ref_dft:
        isolated = False
        if is_val:
            isolated = True
    else:
        # when comparing with valcond, the semicore are already excluded
        num_semicore = 0
        isolated = True

    ref_bands_arr = ref_bands.get_bands()
    num_bands_ref = ref_bands_arr.shape[1]

    cmp_bands_arr = cmp_bands.get_bands()
    num_bands_cmp = cmp_bands_arr.shape[1]

    exclude_list_dft = []
    if num_semicore > 0:
        exclude_list_dft.extend(list(range(num_semicore)))

    if is_val:
        exclude_list_dft.extend(list(range(num_semicore + num_val, num_bands_ref)))
    else:
        exclude_list_dft.extend(list(range(num_semicore, num_semicore + num_val)))
        if not is_ref_dft:
            # cond_to_valcond
            exclude_list_dft.extend(
                list(range(num_semicore + num_val + num_bands_cmp, num_bands_ref))
            )
    # switch to 1 indexed
    if len(exclude_list_dft) > 0:
        exclude_list_dft = [_ + 1 for _ in exclude_list_dft]

    print("exclude_list_dft", exclude_list_dft)
    print("isolated", isolated)
    print("fermi_energy", fermi_energy)

    if isolated:
        bandsdist = bands_distance_isolated(
            ref_bands.get_bands(), cmp_bands.get_bands(), exclude_list_dft
        )
        # Only return average distance, not max distance
        bandsdist = bandsdist[0]
    else:
        # Bands distance from Ef to Ef+5
        bandsdist = bands_distance(ref_bands, cmp_bands, fermi_energy, exclude_list_dft)
        # Only return average distance, not max distance
        bandsdist = bandsdist[:, 1]
        # Return Ef+2
        bandsdist = bandsdist[2]

    return bandsdist


def compute_band_distance(wkc: Wannier90SplitWorkChain) -> dict:
    """Compute band distance and max distance."""
    from copy import deepcopy

    from aiida_wannier90_workflows.utils.bands.distance import (
        bands_distance,
        bands_distance_isolated,
    )

    # pylint: disable=unused-variable

    results = {
        "bands_distance.cond_to_valcond": None,
        "bands_distance.cond_to_ref": None,
        "bands_distance.val_to_valcond": None,
        "bands_distance.val_to_ref": None,
        "bands_distance.valcond_to_ref": None,
        "bands_max_distance.cond_to_valcond": None,
        "bands_max_distance.cond_to_ref": None,
        "bands_max_distance.val_to_valcond": None,
        "bands_max_distance.val_to_ref": None,
        "bands_max_distance.valcond_to_ref": None,
    }

    # valcond_bands = wkc.outputs.valcond.band_structure
    #
    # workaround for the bug in the output band_structure
    # some old workchains outputs the 1st w90calc as the band_structure,
    # this is wrong, this is not the optimal band_structure!
    # Somehow in the future, I should remove this workaround.
    if "wannier90_plot" in wkc.outputs.valcond:
        valcond = wkc.outputs.valcond.wannier90_plot.interpolated_bands
    else:
        valcond = wkc.outputs.valcond.wannier90_optimal.interpolated_bands
    val = wkc.outputs.val.interpolated_bands
    cond = wkc.outputs.cond.interpolated_bands
    fermi = val.creator.inputs.parameters.get_dict()["fermi_energy"]

    calc_split = wkc.base.links.get_outgoing(link_label_filter="split").one().node
    num_val = calc_split.inputs.num_val.value

    num_bands_valcond = valcond.get_bands().shape[1]
    exclude_list_dft = list(
        range(num_val + 1, num_bands_valcond + 1)
    )  # exclude conduction
    dist = bands_distance_isolated(
        valcond.get_bands(), val.get_bands(), exclude_list_dft=exclude_list_dft
    )
    bands_dist, max_dist, max_dist_2, max_dist_loc, max_dist_2_loc = dist
    results["bands_max_distance.val_to_valcond"] = max_dist_2
    results["bands_distance.val_to_valcond"] = bands_dist

    exclude_list_dft = list(range(1, num_val + 1))  # exclude valence, idx starts from 1
    dist = bands_distance_isolated(
        valcond.get_bands(), cond.get_bands(), exclude_list_dft=exclude_list_dft
    )
    bands_dist, max_dist, max_dist_2, max_dist_loc, max_dist_2_loc = dist
    # print(valcond, cond, exclude_list_dft, num_val, num_bands_valcond)
    results["bands_max_distance.cond_to_valcond"] = max_dist_2
    results["bands_distance.cond_to_valcond"] = bands_dist

    ref = wkc.inputs.valcond.optimize_reference_bands
    semicore_list = wkc.inputs.valcond.wannier90.wannier90.parameters.get_dict().get(
        "exclude_bands", []
    )
    exclude_list_dft = deepcopy(semicore_list)  # exclude semicore
    # print(ref, valcond, fermi, exclude_list_dft, semicore_list, num_val, num_bands_valcond)
    dist = bands_distance(ref, valcond, fermi, exclude_list_dft=exclude_list_dft)
    mu, bands_dist, max_dist, max_dist_2 = dist[2, :]
    assert abs(mu - fermi - 2) < 1e-6
    results["bands_max_distance.valcond_to_ref"] = max_dist_2
    results["bands_distance.valcond_to_ref"] = bands_dist

    exclude_list_dft = deepcopy(semicore_list)
    # exclude semicore + conduction
    exclude_list_dft.extend(
        range(
            len(semicore_list) + num_val + 1, len(semicore_list) + num_bands_valcond + 1
        )
    )
    # print(ref, val, exclude_list_dft, semicore_list, num_val, num_bands_valcond)
    #
    # dist = bands_distance_isolated(ref.get_bands(), val.get_bands(), exclude_list_dft=exclude_list_dft)
    # bands_dist, max_dist, max_dist_2, max_dist_loc, max_dist_2_loc = dist
    #
    dist = bands_distance(ref, val, fermi, exclude_list_dft=exclude_list_dft)
    mu, bands_dist, max_dist, max_dist_2 = dist[2, :]
    #
    results["bands_max_distance.val_to_ref"] = max_dist_2
    results["bands_distance.val_to_ref"] = bands_dist

    exclude_list_dft = deepcopy(semicore_list)
    # exclude semicore + valence
    exclude_list_dft.extend(
        range(len(semicore_list) + 1, len(semicore_list) + num_val + 1)
    )
    # print(ref, cond, exclude_list_dft, semicore_list, num_val, num_bands_valcond)
    dist = bands_distance(ref, cond, fermi, exclude_list_dft=exclude_list_dft)
    mu, bands_dist, max_dist, max_dist_2 = dist[2, :]
    assert abs(mu - fermi - 2) < 1e-6
    results["bands_max_distance.cond_to_ref"] = max_dist_2
    results["bands_distance.cond_to_ref"] = bands_dist

    exclude_list_dft = deepcopy(semicore_list)  # exclude semicore
    # valence of valcond
    val_valcond = valcond.get_bands()[:, :num_val]
    dist = bands_distance_isolated(ref, val_valcond, exclude_list_dft=exclude_list_dft)
    bands_dist, max_dist, max_dist_2, max_dist_loc, max_dist_2_loc = dist
    results["bands_max_distance.val_valcond_to_ref"] = max_dist_2
    results["bands_distance.val_valcond_to_ref"] = bands_dist

    cond_valcond = valcond.get_bands()[:, num_val:]
    exclude_list_dft = deepcopy(semicore_list)  # exclude semicore
    exclude_list_dft.extend(
        range(len(semicore_list) + 1, len(semicore_list) + num_val + 1)
    )
    dist = bands_distance(ref, cond_valcond, fermi, exclude_list_dft=exclude_list_dft)
    mu, bands_dist, max_dist, max_dist_2 = dist[2, :]
    assert abs(mu - fermi - 2) < 1e-6
    results["bands_max_distance.cond_valcond_to_ref"] = max_dist_2
    results["bands_distance.cond_valcond_to_ref"] = bands_dist

    return results
