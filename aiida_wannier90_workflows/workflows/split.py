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
from aiida_wannier90_workflows.utils.workflows.plot import get_workchain_fermi_energy
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
        from aiida_wannier90_workflows.utils.workflows.builder import (
            recursive_merge_builder,
        )

        projection_type = kwargs.pop("projection_type", None)
        bands_kpoints = kwargs.pop("bands_kpoints", None)
        plot_wannier_functions = kwargs.pop("plot_wannier_functions", False)
        reference_bands = kwargs.pop("reference_bands", None)

        # Prepare workchain builder
        valcond_builder = Wannier90OptimizeWorkChain.get_builder_from_protocol(
            codes,
            structure,
            reference_bands=reference_bands,
            bands_distance_threshold=bands_distance_threshold,
            projection_type=projection_type,
            bands_kpoints=bands_kpoints,
            plot_wannier_functions=plot_wannier_functions,
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

        # stricter convergence
        parameters["conv_tol"] = 1e-9
        parameters["num_iter"] = 5000

        # exclude_bands = parameters.pop('exclude_bands', [])

        parameters["bands_plot"] = True
        val_inputs["wannier90"]["bands_kpoints"] = bands_kpoints

        # Plot WF
        if plot_wannier_functions:
            parameters["wannier_plot"] = True

        parameters["wvfn_formatted"] = True
        parameters["spn_formatted"] = True

        val_inputs["wannier90"]["parameters"] = orm.Dict(dict=parameters)

        # explicit shallow copy, deepcopy will duplicate orm.Code and
        # aiida engine will store them as new code!
        cond_inputs = copy.copy(val_inputs)
        parameters = cond_inputs["wannier90"]["parameters"].get_dict()
        if num_val is None:
            parameters.pop("num_wann", None)
        else:
            parameters["num_wann"] = num_wann - num_val
        cond_inputs["wannier90"]["parameters"] = orm.Dict(dict=parameters)

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

        bands: orm.BandsData = workchain.outputs.band_structure
        bands_arr = bands.get_bands()
        homo, lumo = get_homo_lumo(bands_arr, fermi_energy)
        band_gap = lumo - homo
        gap_threshold = 1e-2
        if band_gap < gap_threshold:
            self.report(f"{homo=}, {lumo=}, {band_gap=}, seems metal?")

        num_wann = bands.creator.inputs.parameters.get_dict()["num_wann"]
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
            inputs["parent_folder"] = self.ctx.workchain_valcond.outputs["wannier90"][
                "remote_folder"
            ]
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
            inputs[
                "remote_input_folder"
            ] = self.ctx.workchain_split.outputs.remote_folder_val

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

            inputs["parameters"] = orm.Dict(dict=parameters)

        base_inputs["wannier90"] = inputs
        base_inputs["clean_workdir"] = orm.Bool(False)

        return base_inputs

    def run_val(self):
        """Overide parent."""
        inputs = self.prepare_val_inputs()

        metadata = inputs.get("metadata", {})
        metadata["call_link_label"] = "val"
        inputs["metadata"] = metadata
        inputs = prepare_process_inputs(Wannier90BaseWorkChain, inputs)
        running = self.submit(Wannier90BaseWorkChain, **inputs)
        self.report(f"launching {running.process_label}<{running.pk}>")

        return ToContext(workchain_val=running)

    def inspect_val(self):  # pylint: disable=inconsistent-return-statements
        """Overide parent."""

        workchain = self.ctx.workchain_val
        if self.ctx.ref_bands is not None:
            bandsdist = self._get_bands_distance(
                self.ctx.ref_bands,
                workchain.outputs["interpolated_bands"],
                is_val=True,
                isolated=False,
            )
            self.ctx.bandsdist_val_to_ref = bandsdist
        if self.should_run_valcond():
            bandsdist = self._get_bands_distance(
                self.ctx.workchain_valcond.outputs.band_structure,
                workchain.outputs["interpolated_bands"],
                is_val=True,
                isolated=True,
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
            inputs[
                "remote_input_folder"
            ] = self.ctx.workchain_split.outputs.remote_folder_cond

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

            inputs["parameters"] = orm.Dict(dict=parameters)

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
                isolated=False,
            )
            self.ctx.bandsdist_cond_to_ref = bandsdist
        if self.should_run_valcond():
            bandsdist = self._get_bands_distance(
                self.ctx.workchain_valcond.outputs.band_structure,
                workchain.outputs["interpolated_bands"],
                is_val=False,
                isolated=True,
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
        isolated: bool,
    ) -> float:
        """Get bands distance for isolated or Fermi energy + 2eV.

        :param is_val: calc valence or conduction
        :param isolated: calc Ef+2 or isolated bands
        """
        num_semicore, num_val, fermi_energy = self._get_bands_distance_consts()

        if isolated:
            # when comparing with valcond, the semicore are already excluded
            num_semicore = 0

        bandsdist = get_bands_distance(
            ref_bands,
            cmp_bands,
            is_val=is_val,
            isolated=isolated,
            num_semicore=num_semicore,
            num_val=num_val,
            fermi_energy=fermi_energy,
        )

        return bandsdist


def get_bands_distance(
    dft_bands: orm.BandsData,
    wan_bands: orm.BandsData,
    *,
    is_val: bool,
    isolated: bool,  # or Ef2
    num_semicore: int,
    num_val: int,
    fermi_energy: float,
) -> float:
    """Get bands distance for isolated group of bands."""
    from aiida_wannier90_workflows.utils.bands.distance import (
        bands_distance,
        bands_distance_isolated,
    )

    dft_bands_arr = dft_bands.get_bands()
    num_bands_dft = dft_bands_arr.shape[1]

    wan_bands_arr = wan_bands.get_bands()
    num_bands_wan = wan_bands_arr.shape[1]

    exclude_list_dft = []
    if num_semicore > 0:
        exclude_list_dft.extend(list(range(num_semicore)))

    if is_val:
        exclude_list_dft.extend(list(range(num_semicore + num_val, num_bands_dft)))
    else:
        exclude_list_dft.extend(list(range(num_semicore, num_semicore + num_val)))
        exclude_list_dft.extend(
            list(range(num_semicore + num_val + num_bands_wan, num_bands_dft))
        )
    # switch to 1 indexed
    if len(exclude_list_dft) > 0:
        exclude_list_dft = [_ + 1 for _ in exclude_list_dft]

    if isolated:
        bandsdist = bands_distance_isolated(
            dft_bands_arr, wan_bands_arr, exclude_list_dft
        )
        # Only return average distance, not max distance
        bandsdist = bandsdist[0]
    else:
        # Bands distance from Ef to Ef+5
        bandsdist = bands_distance(dft_bands, wan_bands, fermi_energy, exclude_list_dft)
        # Only return average distance, not max distance
        bandsdist = bandsdist[:, 1]
        # Return Ef+2
        bandsdist = bandsdist[2]

    return bandsdist