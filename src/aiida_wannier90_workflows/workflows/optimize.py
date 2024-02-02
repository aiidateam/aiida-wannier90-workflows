"""Workchain to automatically optimize dis_proj_min/max for projectability disentanglement."""

import pathlib
import typing as ty
import warnings

import numpy as np

from aiida import orm
from aiida.engine import ProcessBuilder, ToContext, append_, if_, while_
from aiida.orm.nodes.data.base import to_aiida_type

from aiida_quantumespresso.utils.mapping import prepare_process_inputs

from aiida_wannier90_workflows.common.types import WannierProjectionType
from aiida_wannier90_workflows.utils.workflows import get_last_calcjob

from .bands import Wannier90BandsWorkChain
from .base.wannier90 import Wannier90BaseWorkChain

__all__ = ["validate_inputs", "Wannier90OptimizeWorkChain"]


def validate_inputs(inputs, ctx=None):  # pylint: disable=unused-argument
    """Validate the inputs of the entire input namespace of `Wannier90OptimizeWorkChain`."""
    from .bands import validate_inputs as parent_validate_inputs

    # Call parent validator
    result = parent_validate_inputs(inputs)
    if result is not None:
        return result

    parameters = inputs["wannier90"]["wannier90"]["parameters"].get_dict()

    optimize_disproj = inputs.get("optimize_disproj", True)
    if optimize_disproj:
        if all(_ not in parameters for _ in ("dis_proj_min", "dis_proj_max")):
            return "Trying to optimize dis_proj_min/max but no dis_proj_min/max in wannier90 parameters?"

    if "optimize_reference_bands" in inputs and not optimize_disproj:
        warnings.warn(
            "`optimize_reference_bands` is provided but `optimize_disproj = False`?"
        )

    if (
        "optimize_bands_distance_threshold" in inputs
        and "optimize_reference_bands" not in inputs
    ):
        return "No `optimize_reference_bands` but `optimize_bands_distance_threshold` is set?"

    separate_plotting = inputs.get("separate_plotting", False)
    plot_inputs = [
        parameters.get(_, False)
        for _ in Wannier90OptimizeWorkChain._WANNIER90_PLOT_INPUTS  # pylint: disable=protected-access
    ]
    if separate_plotting:
        if not any(plot_inputs):
            warnings.warn(
                "Trying to separate plotting routines but no "
                f"{'/'.join(Wannier90OptimizeWorkChain._WANNIER90_PLOT_INPUTS)} in wannier90 parameters. "  # pylint: disable=protected-access
                "The unnecessary plotting step can be avoided "
                "by setting `builder.separate_plotting = False` and resubmitting the workchain."
            )
    else:
        if optimize_disproj and parameters.get("wannier_plot", False):
            return (
                "Detected wannier_plot = True in wannier90 parameters, "
                "but `separate_plotting = False`. For optimizing projectability "
                "disentanglement (`optimize_disproj = True` ), it is highly recommended "
                "to plot Wannier functions in a separate step to reduce computational time. "
                "To do so, set `builder.separate_plotting = True` before submitting the workchain."
            )
        if optimize_disproj and any(plot_inputs):
            warnings.warn(
                f"Detected a plotting input ({'/'.join(Wannier90OptimizeWorkChain._WANNIER90_PLOT_INPUTS)}) "
                "in wannier90 parameters, but `separate_plotting = False`. For optimizing projectability "
                "disentanglement (`optimize_disproj = True` ), it is recommended to run the plotting mode "
                "in a separate step to reduce computational time. "
                "This can be done by setting `builder.separate_plotting = True` and resubmitting the workchain."
            )

    return None


# pylint: disable=too-many-lines
class Wannier90OptimizeWorkChain(
    Wannier90BandsWorkChain
):  # pylint: disable=too-many-public-methods
    """Workchain to optimize dis_proj_min/max for projectability disentanglement."""

    # The following keys are for wannier90.x plotting, i.e. they can be restarted from
    # chk file by setting `restart = plot` in wannier90.win.
    # the `bands_plot` is commented out since it is rather cheap to compute,
    # and also we want to check the band distance during each iteration of optimizing dis_proj_min/max
    # even if `separate_plotting = True`
    _WANNIER90_PLOT_INPUTS = (
        "wannier_plot",
        # "bands_plot",
        "write_tb",
        "write_hr",
        "write_hhmn",
        "write_hkmn",
        "write_hvmn",
        "write_hdmn",
        "fermi_surface_plot",
    )

    @classmethod
    def define(cls, spec):
        """Define the process spec."""
        super().define(spec)

        spec.input(
            "separate_plotting",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            serializer=to_aiida_type,
            help=(
                "If True separate the maximal localisation and the plotting of bands/Wannier function in two steps. "
                "This allows reusing the chk file to restart plotting if it were crashed due to memory issue."
            ),
        )
        spec.input(
            "optimize_disproj",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(True),
            serializer=to_aiida_type,
            help=(
                "If True iterate dis_proj_min/max to find the best MLWFs for projectability disentanglement."
            ),
        )
        spec.input(
            "optimize_disprojmax_range",
            valid_type=orm.List,
            default=lambda: orm.List(list=list(np.linspace(0.99, 0.85, 15))),
            serializer=to_aiida_type,
            help=(
                "The range to iterate dis_proj_max. `None` means disabling projectability disentanglement."
            ),
        )
        spec.input(
            "optimize_disprojmin_range",
            valid_type=orm.List,
            default=lambda: orm.List(list=list(np.linspace(0.01, 0.02, 2))),
            serializer=to_aiida_type,
            help=(
                "The range to iterate dis_proj_min. `None` means disabling projectability disentanglement."
            ),
        )
        spec.input(
            "optimize_reference_bands",
            valid_type=orm.BandsData,
            required=False,
            help=(
                "If provided, during the iteration of dis_proj_min/max, the BandsData will be the reference "
                "for calculating bands distance, the final optimal MLWFs will be selected based on both spreads "
                "and bands distance. If not provided, spreads will be the criterion for selecting optimal MLWFs. "
                "The bands distance is calculated for bands below Fermi energy + 2eV."
            ),
        )
        spec.input(
            "optimize_bands_distance_threshold",
            valid_type=orm.Float,
            required=False,
            serializer=to_aiida_type,
            help=(
                "If provided, during the iteration of dis_proj_min/max, if the bands distance is smaller "
                "than this threshold, the optimization will stop. Unit is eV."
            ),
        )
        spec.input(
            "optimize_spreads_imbalence_threshold",
            valid_type=orm.Float,
            required=False,
            serializer=to_aiida_type,
            help=(
                "If provided, during the iteration of dis_proj_min/max, if the spreads imbalence is smaller "
                "than this threshold, the optimization will stop."
            ),
        )
        spec.inputs.validator = validate_inputs

        spec.outline(
            cls.setup,
            if_(cls.should_run_seekpath)(
                cls.run_seekpath,
            ),
            if_(cls.should_run_scf)(
                cls.run_scf,
                cls.inspect_scf,
            ),
            if_(cls.should_run_nscf)(
                cls.run_nscf,
                cls.inspect_nscf,
            ),
            if_(cls.should_run_open_grid)(
                cls.run_open_grid,
                cls.inspect_open_grid,
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
            while_(cls.should_run_wannier90_optimize)(
                cls.run_wannier90_optimize,
                cls.inspect_wannier90_optimize,
            ),
            cls.inspect_wannier90_optimize_final,
            if_(cls.should_run_wannier90_plot)(
                cls.run_wannier90_plot,
                cls.inspect_wannier90_plot,
            ),
            cls.results,
        )

        spec.expose_outputs(
            Wannier90BaseWorkChain,
            namespace="wannier90_optimal",
            namespace_options={"required": False},
        )
        spec.expose_outputs(
            Wannier90BaseWorkChain,
            namespace="wannier90_plot",
            namespace_options={"required": False},
        )
        # for spin_collinear condition:
        spec.expose_outputs(
            Wannier90BaseWorkChain,
            namespace="wannier90_optimal_up",
            namespace_options={"required": False},
        )
        spec.expose_outputs(
            Wannier90BaseWorkChain,
            namespace="wannier90_plot_up",
            namespace_options={"required": False},
        )
        spec.expose_outputs(
            Wannier90BaseWorkChain,
            namespace="wannier90_optimal_down",
            namespace_options={"required": False},
        )
        spec.expose_outputs(
            Wannier90BaseWorkChain,
            namespace="wannier90_plot_down",
            namespace_options={"required": False},
        )

        spec.output(
            "bands_distance",
            valid_type=orm.Float,
            required=False,
            help="Bands distances between reference bands and Wannier interpolated bands for Ef to Ef+5eV.",
        )

        spec.exit_code(
            500,
            "ERROR_SUB_PROCESS_FAILED_WANNIER90_OPTIMIZE",
            message="All the trials on dis_proj_min/max have failed, cannot compare bands distance",
        )
        spec.exit_code(
            501,
            "ERROR_SUB_PROCESS_FAILED_WANNIER90_OPTIMIZE",
            message="All the trials on dis_proj_min/max have failed, cannot compare spreads",
        )
        spec.exit_code(
            500,
            "ERROR_SUB_PROCESS_FAILED_WANNIER90_PLOT",
            message="the Wannier90Calculation plotting sub process failed",
        )

    @classmethod
    def get_protocol_filepath(cls) -> pathlib.Path:
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files

        from . import protocols

        return files(protocols) / "optimize.yaml"

    @classmethod
    def get_builder_from_protocol(  # pylint: disable=arguments-differ
        cls,
        codes: ty.Mapping[str, ty.Union[str, int, orm.Code]],
        structure: orm.StructureData,
        *,
        reference_bands: orm.BandsData = None,
        bands_distance_threshold: float = 1e-2,  # unit is eV
        **kwargs,
    ) -> ProcessBuilder:
        """Return a builder prepopulated with inputs selected according to the specified arguments.

        :return: [description]
        :rtype: ProcessBuilder
        """
        from copy import deepcopy

        from aiida_wannier90_workflows.utils.workflows.bands import (
            has_overlapping_semicore,
        )

        kwargs.setdefault("projection_type", WannierProjectionType.ATOMIC_PROJECTORS_QE)
        if reference_bands and kwargs.get("bands_kpoints", None) is None:
            warnings.warn(
                "It is recommended to provide both `reference_bands` and `bands_kpoints` so that"
                " the seekpath step can be skipped."
            )

        inputs = cls.get_protocol_inputs(
            protocol=kwargs.get("protocol", None),
            overrides=kwargs.pop("overrides", None),
        )

        parent_builder = Wannier90BandsWorkChain.get_builder_from_protocol(
            codes, structure, overrides=deepcopy(inputs), **kwargs
        )

        if reference_bands is not None:
            exclude_semicore = kwargs.get("exclude_semicore", True)
            if exclude_semicore:
                params = parent_builder.wannier90.wannier90.parameters.get_dict()
                exclude_bands = params.get("exclude_bands", None)
                overlapping_semicore = has_overlapping_semicore(
                    reference_bands, exclude_bands
                )
                if overlapping_semicore:
                    warnings.warn(
                        "The reference bands has overlapping semicore bands, "
                        "the exclude_semicore option is set to False."
                    )
                    kwargs["exclude_semicore"] = False
                    parent_builder = Wannier90BandsWorkChain.get_builder_from_protocol(
                        codes, structure, overrides=inputs, **kwargs
                    )

        # Prepare workchain builder
        builder = cls.get_builder()
        builder._data = parent_builder._data  # pylint: disable=protected-access

        # Inputs for optimizing dis_proj_min/max
        if reference_bands:
            builder.optimize_disproj = True
            builder.optimize_reference_bands = reference_bands
            builder.optimize_bands_distance_threshold = bands_distance_threshold

        if builder.optimize_disproj:
            # If optimizing dis_proj_min/max,
            # make sure Wannier functions are plotted in a separate step since it is heavy
            if (
                builder["wannier90"]["wannier90"]["parameters"]
                .get_dict()
                .get("wannier_plot", False)
            ):
                builder.separate_plotting = True
        return builder

    def setup(self):
        """Define the current structure in the context to be the input structure."""
        super().setup()

        dis_proj_min = self.inputs["optimize_disprojmin_range"].get_list()
        dis_proj_max = self.inputs["optimize_disprojmax_range"].get_list()
        # dis_proj_max changes the fastest
        self.ctx.optimize_minmax_new = [
            (i, j) for i in dis_proj_min for j in dis_proj_max
        ]

        # Arrays to save calculated results
        self.ctx.optimize_minmax = []
        self.ctx.optimize_bandsdist = []
        self.ctx.optimize_spreads_imbalence = []
        # The optimal wannier90 workchain
        self.ctx.optimize_best = None
        # Store the spinor bands for spin_collinear calculation
        if self.ctx.spin_collinear:
            self.ctx.optimize_spinor_bands = []

        # For separate_plotting, restore these inputs when running plotting calc.
        self.ctx.saved_parameters = {}
        if self.inputs["separate_plotting"]:
            parameters = self.inputs.wannier90.wannier90["parameters"].get_dict()
            # I convert the tuple to list so it can be changed
            excluded_inputs = list(Wannier90OptimizeWorkChain._WANNIER90_PLOT_INPUTS)
            for key in excluded_inputs:
                plot_input = parameters.get(key, False)
                if plot_input:
                    self.ctx.saved_parameters[key] = plot_input

    def should_run_wannier90_optimize(self):
        """Whether should optimize dis_proj_min/max."""
        if not self.inputs["optimize_disproj"]:
            return False

        if "optimize_bands_distance_threshold" in self.inputs:
            threshold = self.inputs["optimize_bands_distance_threshold"]
            if self.ctx.workchain_wannier90_bandsdist <= threshold:
                # Stop if the initial bands distance is already good enough
                self.ctx.optimize_minmax_new = []
            else:
                # Replace `None` by a huge number to avoid np.min error:
                # TypeError: '<=' not supported between instances of 'float' and 'NoneType'
                opt_dist = [_ if _ else 1e5 for _ in self.ctx.optimize_bandsdist]
                if len(opt_dist) > 0 and np.min(opt_dist) <= threshold:
                    self.ctx.optimize_minmax_new = []
        elif "optimize_spreads_imbalence_threshold" in self.inputs:
            threshold = self.inputs["optimize_spreads_imbalence_threshold"]
            if self.ctx.workchain_wannier90_spreads_imbalence <= threshold:
                # Stop if the initial spreads are already good enough
                self.ctx.optimize_minmax_new = []
            elif (
                len(self.ctx.optimize_spreads_imbalence) > 0
                and np.min(self.ctx.optimize_spreads_imbalence) <= threshold
            ):
                self.ctx.optimize_minmax_new = []

        if len(self.ctx.optimize_minmax_new) == 0:
            return False

        return True

    def has_run_wannier90_optimize(self):
        """Whether the optimization loop has been invoked."""
        if not self.ctx.spin_collinear:
            result = "workchain_wannier90_optimize" in self.ctx
        else:
            result = (
                "workchain_wannier90_optimize_up" in self.ctx
                and "workchain_wannier90_optimize_down" in self.ctx
            )
        return result

    def should_run_wannier90_plot(self):
        """Whether to run wannier90 maximal localisation and plotting in two steps or in one step."""
        return self.inputs["separate_plotting"]

    def prepare_wannier90_pp_inputs(self):
        """Override parent method.

        :return: the inputs port
        :rtype: InputPort
        """
        base_inputs = super().prepare_wannier90_pp_inputs()
        inputs = base_inputs["wannier90"]

        parameters = inputs.parameters.get_dict()

        # Do not run plotting subroutines in the Wannier90BandsWorkChain, they will be run in a separate step
        if self.should_run_wannier90_plot():
            for key in self.ctx.saved_parameters:
                parameters.pop(key, None)
            inputs.parameters = orm.Dict(parameters)

            if "optimize_reference_bands" not in self.inputs:
                inputs.pop("kpoint_path", None)
                inputs.pop("bands_kpoints", None)

            base_inputs["wannier90"] = inputs

        # This will use the input bands which usually is a bands along kpath,
        # also used to calculate bands distance.
        # However in Wannier90BaseWorkChain, this bands will also be used to
        # set the max of dis_froz_max (in `prepare_inputs()`), however, if
        # the number of bands of this input is too small, this will lead to
        # too small max limit of dis_froz_max, essentially causing a very low
        # dis_froz_max in the actual wannier90 calculation.
        # In such case, it is safer to use the output_band of scf or nscf step
        # inside the workflow, however
        # these are bands on a grid, usually their LUMO is a bit larger than
        # LUMO from bands along kpath, so when shifting dis_froz_max w.r.t. LUMO,
        # it might be a bit inaccurate.
        #
        # The parent class Wannier90BandsWorkChain.inputs.wannier90.bands is empty,
        # so it will use output_band of scf or nscf. Here I explicitly overwrite the
        # wannier90.bands by the reference_bands.
        if self.inputs.optimize_disproj and "optimize_reference_bands" in self.inputs:
            base_inputs.bands = self.inputs.optimize_reference_bands

        return base_inputs

    def run_wannier90(self):
        """Overide parent, pop stash settings."""
        if not self.ctx.spin_collinear:
            inputs = self.prepare_wannier90_inputs()

            # I should not stash files if there is an additional plotting step,
            # otherwise there is a RemoteStashFolderData in outputs
            if self.should_run_wannier90_plot():
                inputs["wannier90"]["metadata"]["options"].pop("stash", None)

            inputs["metadata"] = {"call_link_label": "wannier90"}
            inputs = prepare_process_inputs(Wannier90BaseWorkChain, inputs)
            running = self.submit(Wannier90BaseWorkChain, **inputs)
            self.report(f"launching {running.process_label}<{running.pk}>")

            result_wannier90 = ToContext(workchain_wannier90=running)
        else:
            result_wannier90 = {}
            self.ctx.current_spin = "up"
            result_wannier90.update(self.run_wannier90_up())
            self.ctx.current_spin = "down"
            result_wannier90.update(self.run_wannier90_down())

        return result_wannier90

    def run_wannier90_up(self):
        """Wannier90 step for MLWF."""
        inputs = self.prepare_wannier90_inputs()
        if self.should_run_wannier90_plot():
            inputs["wannier90"]["metadata"]["options"].pop("stash", None)
        inputs["metadata"] = {"call_link_label": "wannier90_up"}

        inputs = prepare_process_inputs(Wannier90BaseWorkChain, inputs)
        running = self.submit(Wannier90BaseWorkChain, **inputs)
        self.report(f"launching {running.process_label}<{running.pk}>")

        return ToContext(workchain_wannier90_up=running)

    def run_wannier90_down(self):
        """Wannier90 step for MLWF."""
        inputs = self.prepare_wannier90_inputs()
        if self.should_run_wannier90_plot():
            inputs["wannier90"]["metadata"]["options"].pop("stash", None)
        inputs["metadata"] = {"call_link_label": "wannier90_down"}

        inputs = prepare_process_inputs(Wannier90BaseWorkChain, inputs)
        running = self.submit(Wannier90BaseWorkChain, **inputs)
        self.report(f"launching {running.process_label}<{running.pk}>")

        return ToContext(workchain_wannier90_down=running)

    def inspect_wannier90(self):
        """Overide parent."""
        super().inspect_wannier90()

        if not self.ctx.spin_collinear:
            workchain = self.ctx.workchain_wannier90
        else:
            workchain = [
                self.ctx.workchain_wannier90_up,
                self.ctx.workchain_wannier90_down,
            ]

        if "optimize_reference_bands" in self.inputs:
            bandsdist = self._get_bands_distance(workchain)
            self.ctx.workchain_wannier90_bandsdist = bandsdist
            if not self.ctx.spin_collinear:
                self.report(
                    f"current workchain<{workchain.pk}> bands distance={bandsdist:.2e}eV"
                )
            else:
                self.report(
                    f"current workchain(up: <{workchain[0].pk}>, down: <{workchain[1].pk}>) "
                    f"bands distance={bandsdist:.2e}eV"
                )

        if not self.ctx.spin_collinear:
            self.ctx.workchain_wannier90_spreads_imbalence = get_spreads_imbalence(
                workchain.outputs.output_parameters["wannier_functions_output"]
            )
        else:
            self.ctx.workchain_wannier90_spreads_imbalence = get_spreads_imbalence(
                workchain[0].outputs.output_parameters["wannier_functions_output"]
                + workchain[1].outputs.output_parameters["wannier_functions_output"]
            )

    def prepare_wannier90_optimize_inputs(self):
        """Prepare inputs for optimize run."""
        base_inputs = super().prepare_wannier90_inputs()
        inputs = base_inputs["wannier90"]

        # I need to save `inputs.wannier90.metadata.options.resources`, somehow it is missing if I
        # copy all the inputs from `self.ctx.workchain_wannier90`.
        # resources = base_inputs['wannier90']['wannier90']['metadata']['options']['resources']

        # Use the Wannier90BaseWorkChain-corrected parameters, especially `num_mpiprocs_per_machine`
        if self.ctx.spin_collinear:
            if self.ctx.current_spin == "up":
                last_calc = get_last_calcjob(self.ctx.workchain_wannier90_up)
            elif self.ctx.current_spin == "down":
                last_calc = get_last_calcjob(self.ctx.workchain_wannier90_down)
        for key in last_calc.inputs:
            inputs[key] = last_calc.inputs[key]

        parameters = inputs.parameters.get_dict()

        dis_proj_min, dis_proj_max = self.ctx.optimize_minmax_new[0]
        parameters["dis_proj_min"] = dis_proj_min
        parameters["dis_proj_max"] = dis_proj_max

        if "optimize_reference_bands" in self.inputs:
            parameters["bands_plot"] = True
            if self.ctx.current_kpoint_path:
                inputs.kpoint_path = self.ctx.current_kpoint_path
            if self.ctx.current_bands_kpoints:
                inputs.bands_kpoints = self.ctx.current_bands_kpoints

        inputs.parameters = orm.Dict(parameters)

        # I should not stash files if there is an additional plotting step,
        # otherwise there is a RemoteStashFolderData in outputs
        if self.should_run_wannier90_plot():
            inputs["metadata"]["options"].pop("stash", None)

        base_inputs["wannier90"] = inputs
        base_inputs["clean_workdir"] = orm.Bool(False)

        return base_inputs

    def run_wannier90_optimize(self):
        """Optimize dis_proj_min/max."""
        if not self.ctx.spin_collinear:
            inputs = self.prepare_wannier90_optimize_inputs()

            iteration = len(self.ctx.optimize_minmax) + 1  # Start from 1
            inputs["metadata"] = {
                "call_link_label": f"wannier90_optimize_iteration{iteration}"
            }

            # Disable the error handler which might modify dis_proj_min
            handler_overrides = {
                "handle_disentanglement_not_enough_states": {"enabled": False}
            }
            inputs["handler_overrides"] = orm.Dict(handler_overrides)

            inputs = prepare_process_inputs(Wannier90BaseWorkChain, inputs)
            running = self.submit(Wannier90BaseWorkChain, **inputs)

            dis_proj_min, dis_proj_max = self.ctx.optimize_minmax_new[0]
            self.report(
                f"launching {running.process_label}<{running.pk}> with dis_proj_min={dis_proj_min} "
                f"dis_proj_max={dis_proj_max}"
            )

            result_wannier90_optmize = ToContext(
                workchain_wannier90_optimize=append_(running)
            )
        else:
            result_wannier90_optmize = {}
            self.ctx.current_spin = "up"
            result_wannier90_optmize.update(self.run_wannier90_optimize_up())
            self.ctx.current_spin = "down"
            result_wannier90_optmize.update(self.run_wannier90_optimize_down())

        return result_wannier90_optmize

    def run_wannier90_optimize_up(self):
        """Optimize dis_proj_min/max."""
        inputs = self.prepare_wannier90_optimize_inputs()

        iteration = len(self.ctx.optimize_minmax) + 1  # Start from 1
        inputs["metadata"] = {
            "call_link_label": f"wannier90_optimize_up_iteration{iteration}"
        }

        # Disable the error handler which might modify dis_proj_min
        handler_overrides = {
            "handle_disentanglement_not_enough_states": {"enabled": False}
        }
        inputs["handler_overrides"] = orm.Dict(handler_overrides)

        inputs = prepare_process_inputs(Wannier90BaseWorkChain, inputs)
        running = self.submit(Wannier90BaseWorkChain, **inputs)

        dis_proj_min, dis_proj_max = self.ctx.optimize_minmax_new[0]
        self.report(
            f"launching {running.process_label}<{running.pk}> with dis_proj_min={dis_proj_min} "
            f"dis_proj_max={dis_proj_max}"
        )

        return ToContext(workchain_wannier90_optimize_up=append_(running))

    def run_wannier90_optimize_down(self):
        """Optimize dis_proj_min/max."""
        inputs = self.prepare_wannier90_optimize_inputs()

        iteration = len(self.ctx.optimize_minmax) + 1  # Start from 1
        inputs["metadata"] = {
            "call_link_label": f"wannier90_optimize_down_iteration{iteration}"
        }

        # Disable the error handler which might modify dis_proj_min
        handler_overrides = {
            "handle_disentanglement_not_enough_states": {"enabled": False}
        }
        inputs["handler_overrides"] = orm.Dict(handler_overrides)

        inputs = prepare_process_inputs(Wannier90BaseWorkChain, inputs)
        running = self.submit(Wannier90BaseWorkChain, **inputs)

        dis_proj_min, dis_proj_max = self.ctx.optimize_minmax_new[0]
        self.report(
            f"launching {running.process_label}<{running.pk}> with dis_proj_min={dis_proj_min} "
            f"dis_proj_max={dis_proj_max}"
        )

        return ToContext(workchain_wannier90_optimize_down=append_(running))

    def inspect_wannier90_optimize(self):
        """Verify that the `Wannier90BaseWorkChain` for the wannier90 optimization run successfully finished."""
        if not self.ctx.spin_collinear:
            workchain = self.ctx.workchain_wannier90_optimize[-1]
            workchain_is_finished_ok = workchain.is_finished_ok
            workchain_optimize_ok = (
                "optimize_reference_bands" in self.inputs
                and "interpolated_bands" in workchain.outputs
            )
        else:
            workchain = [
                self.ctx.workchain_wannier90_optimize_up[-1],
                self.ctx.workchain_wannier90_optimize_down[-1],
            ]
            workchain_is_finished_ok = all(_.is_finished_ok for _ in workchain)
            workchain_optimize_ok = "optimize_reference_bands" in self.inputs and all(
                "interpolated_bands" in _.outputs for _ in workchain
            )

        if workchain_is_finished_ok:
            if not self.ctx.spin_collinear:
                spreads = get_spreads_imbalence(
                    workchain.outputs.output_parameters["wannier_functions_output"]
                )
            else:
                spreads = get_spreads_imbalence(
                    workchain[0].outputs.output_parameters["wannier_functions_output"]
                    + workchain[1].outputs.output_parameters["wannier_functions_output"]
                )
            if workchain_optimize_ok:
                bandsdist = self._get_bands_distance(workchain)
                if not self.ctx.spin_collinear:
                    self.report(
                        f"current workchain<{workchain.pk}> bands distance={bandsdist:.2e}eV"
                    )
                else:
                    self.report(
                        f"current workchain(up: <{workchain[0].pk}>, down: <{workchain[1].pk}>) "
                        f"bands distance={bandsdist:.2e}eV"
                    )
            else:
                bandsdist = None
        else:
            if not self.ctx.collinear:
                self.report(
                    f"{workchain.process_label} failed with exit status {workchain.exit_status}, "
                    "but I will keep launching next iteration"
                )
            else:
                for workchain_spin in workchain:
                    if not workchain_spin.is_finished_ok:
                        self.report(
                            f"{workchain_spin.process_label} failed with exit status {workchain_spin.exit_status}, "
                            "but I will keep launching next iteration"
                        )
            spreads = None
            bandsdist = None

        minmax = self.ctx.optimize_minmax_new.pop(0)
        self.ctx.optimize_minmax.append(minmax)
        self.ctx.optimize_bandsdist.append(bandsdist)
        self.ctx.optimize_spreads_imbalence.append(spreads)
        if self.ctx.spin_collinear and "spinor_bands" in self.ctx:
            self.ctx.optimize_spinor_bands.append(self.ctx.spinor_bands)

    def inspect_wannier90_optimize_final(self):
        """Select the optimal choice for dis_proj_min/max."""
        if not self.has_run_wannier90_optimize():
            return

        if not self.ctx.spin_collinear:
            workchains = self.ctx.workchain_wannier90_optimize
        else:
            workchains = [
                self.ctx.workchain_wannier90_optimize_up,
                self.ctx.workchain_wannier90_optimize_down,
            ]
            # Transpose the list so workchains[i_wc] = [wc_up, wc_down]
            workchains = list(map(list, zip(*workchains)))

        # The optimal wannier90 workchain
        self.ctx.optimize_best = None

        if "optimize_reference_bands" in self.inputs:
            # Usually good bands distance means MLWFs have good spreads
            fake_max = 1e5
            bandsdist = np.array(
                [_ if _ else fake_max for _ in self.ctx.optimize_bandsdist]
            )
            idx = np.argmin(bandsdist)
            if bandsdist[idx] < min(fake_max, self.ctx.workchain_wannier90_bandsdist):
                self.ctx.optimize_best = workchains[idx]
                self.ctx.optimize_bandsdist_best = bandsdist[idx]
                self.ctx.optimize_minmax_best = self.ctx.optimize_minmax[idx]
                if not self.ctx.spin_collinear:
                    try:
                        self.ctx.optimize_band_best = (
                            self.ctx.optimize_best.outputs.interpolated_bands
                        )
                    except KeyError:
                        self.ctx.optimize_band_best = None
                else:
                    self.ctx.optimize_band_best = self.ctx.optimize_spinor_bands[idx]
            else:
                # All optimizations failed, just output the initial w90
                if not self.ctx.spin_collinear:
                    self.ctx.optimize_best = self.ctx.workchain_wannier90
                    self.ctx.optimize_bandsdist_best = (
                        self.ctx.workchain_wannier90_bandsdist
                    )
                    # dis_proj_min/max might be corrected by error handlers,
                    # output the last min/max.
                    last_calc = get_last_calcjob(self.ctx.optimize_best)
                else:
                    self.ctx.optimize_best = [
                        self.ctx.workchain_wannier90_up,
                        self.ctx.workchain_wannier90_down,
                    ]
                    self.ctx.optimize_bandsdist_best = (
                        self.ctx.workchain_wannier90_bandsdist
                    )
                    last_calc = get_last_calcjob(self.ctx.optimize_best[0])
                params = last_calc.inputs.parameters.get_dict()
                self.ctx.optimize_minmax_best = (
                    params.get("dis_proj_min", None),
                    params.get("dis_proj_max", None),
                )
            self.report(
                f"Optimal bands distance={self.ctx.optimize_bandsdist_best:.2e}, "
                f"dis_proj_min={self.ctx.optimize_minmax_best[0]} "
                f"dis_proj_max={self.ctx.optimize_minmax_best[1]}"
            )
        else:
            # I only check the spreads are balenced
            spreads = np.array(
                [_ if _ else 1e5 for _ in self.ctx.optimize_spreads_imbalence]
            )
            idx = np.argmin(spreads)
            self.ctx.optimize_best = workchains[idx]
            self.ctx.optimize_minmax_best = self.ctx.optimize_minmax[idx]
            self.report(
                f"Optimal spreads={spreads[idx]}, "
                f"dis_proj_min={self.ctx.optimize_minmax_best[0]} "
                f"dis_proj_max={self.ctx.optimize_minmax_best[1]}"
            )

        if not self.ctx.spin_collinear:
            self.ctx.current_folder = self.ctx.optimize_best.outputs.remote_folder
        else:
            self.ctx.current_folder_up = self.ctx.optimize_best[0].outputs.remote_folder
            self.ctx.current_folder_down = self.ctx.optimize_best[
                1
            ].outputs.remote_folder

    def prepare_wannier90_plot_inputs(self):
        """Wannier90 plot step, also stash files."""
        # Note with `Wannier90WorkChain.prepare_wannier90_inputs()`, the stash setting
        # has been restored.
        base_inputs = super().prepare_wannier90_inputs()
        inputs = base_inputs["wannier90"]

        # Use the corrected parameters
        if self.has_run_wannier90_optimize():
            # Use the optimal parameters
            optimal_workchain = self.ctx.optimize_best
        else:
            # Just use the base workchain
            optimal_workchain = self.ctx.workchain_wannier90
        # Copy inputs, especially the `dis_proj_min/max` might have been corrected
        if not self.ctx.spin_collinear:
            last_calc = get_last_calcjob(optimal_workchain)
        else:
            if self.ctx.current_spin == "up":
                last_calc = get_last_calcjob(optimal_workchain[0])
            elif self.ctx.current_spin == "down":
                last_calc = get_last_calcjob(optimal_workchain[1])
        for key in last_calc.inputs:
            inputs[key] = last_calc.inputs[key]

        # Use `current_folder` which points to the optimal wannier90 folder, since we need the chk file.
        # However we need to explicitly
        # symlink UNK files in that folder, otherwise the plot calculation would fail.
        if not self.ctx.spin_collinear:
            inputs["remote_input_folder"] = self.ctx["current_folder"]
        else:
            inputs["remote_input_folder"] = self.ctx[
                f"current_folder_{self.ctx.current_spin}"
            ]
        # Maybe in aiida-w90, should let Calculation accepts an optional SinglefileData? for chk,
        # so we don't need to explicitly symlink.
        settings = inputs.settings.get_dict()
        if not self.ctx.spin_collinear:
            remote_input_folder_uuid = (
                self.ctx.workchain_pw2wannier90.outputs.remote_folder.computer.uuid
            )
            remote_input_folder_path = pathlib.Path(
                self.ctx.workchain_pw2wannier90.outputs.remote_folder.get_remote_path()
            )
        else:
            remote_input_folder_uuid = self.ctx[
                "workchain_pw2wannier90"
            ].outputs.remote_folder.computer.uuid
            remote_input_folder_path = pathlib.Path(
                self.ctx[
                    f"workchain_pw2wannier90_{self.ctx.current_spin}"
                ].outputs.remote_folder.get_remote_path()
            )
        additional_remote_symlink_list = settings.get(
            "additional_remote_symlink_list", []
        )
        additional_remote_symlink_list += [
            (remote_input_folder_uuid, str(remote_input_folder_path / "UNK*"), ".")
        ]
        settings["additional_remote_symlink_list"] = additional_remote_symlink_list
        inputs.settings = orm.Dict(settings)

        # Restore plotting related tags
        parameters = inputs.parameters.get_dict()
        for key in self.ctx.saved_parameters:
            parameters[key] = True

        parameters["restart"] = "plot"
        inputs.parameters = orm.Dict(parameters)

        if parameters.get("bands_plot", False):
            if self.ctx.current_kpoint_path:
                inputs.kpoint_path = self.ctx.current_kpoint_path
            if self.ctx.current_bands_kpoints:
                inputs.bands_kpoints = self.ctx.current_bands_kpoints

        base_inputs["wannier90"] = inputs
        base_inputs["clean_workdir"] = orm.Bool(False)

        return base_inputs

    def run_wannier90_plot(self):
        """Wannier90 plot step, also stash files."""
        if not self.ctx.spin_collinear:
            inputs = self.prepare_wannier90_plot_inputs()
            inputs["metadata"] = {"call_link_label": "wannier90_plot"}

            inputs = prepare_process_inputs(Wannier90BaseWorkChain, inputs)
            running = self.submit(Wannier90BaseWorkChain, **inputs)
            self.report(
                f"launching {running.process_label}<{running.pk}> in plotting mode"
            )

            result_wannier90_plot = ToContext(workchain_wannier90_plot=running)
        else:
            result_wannier90_plot = {}
            self.ctx.current_spin = "up"
            result_wannier90_plot.update(self.run_wannier90_plot_up())
            self.ctx.current_spin = "down"
            result_wannier90_plot.update(self.run_wannier90_plot_down())

        return result_wannier90_plot

    def run_wannier90_plot_up(self):
        """Wannier90_up plot step, also stash files."""
        inputs = self.prepare_wannier90_plot_inputs()
        inputs["metadata"] = {"call_link_label": "wannier90_plot_up"}

        inputs = prepare_process_inputs(Wannier90BaseWorkChain, inputs)
        running = self.submit(Wannier90BaseWorkChain, **inputs)
        self.report(f"launching {running.process_label}<{running.pk}> in plotting mode")

        return ToContext(workchain_wannier90_plot_up=running)

    def run_wannier90_plot_down(self):
        """Wannier90_down plot step, also stash files."""
        inputs = self.prepare_wannier90_plot_inputs()
        inputs["metadata"] = {"call_link_label": "wannier90_plot_down"}

        inputs = prepare_process_inputs(Wannier90BaseWorkChain, inputs)
        running = self.submit(Wannier90BaseWorkChain, **inputs)
        self.report(f"launching {running.process_label}<{running.pk}> in plotting mode")

        return ToContext(workchain_wannier90_plot_down=running)

    def inspect_wannier90_plot(self):
        """Verify that the `Wannier90BaseWorkChain` for the wannier90 plotting run successfully finished."""
        if not self.ctx.spin_collinear:
            workchain = self.ctx.workchain_wannier90_plot
            workchain_is_finished_ok = workchain.is_finished_ok
        else:
            workchain = [
                self.ctx.workchain_wannier90_plot_up,
                self.ctx.workchain_wannier90_plot_down,
            ]
            workchain_is_finished_ok = all(_.is_finished_ok for _ in workchain)

        if not workchain_is_finished_ok:
            if not self.ctx.spin_collinear:
                self.report(
                    f"{workchain.process_label} failed with exit status {workchain.exit_status}"
                )
            else:
                for workchain_spin in workchain:
                    if not workchain_spin.is_finished_ok:
                        self.report(
                            f"{workchain_spin.process_label} failed with exit status {workchain_spin.exit_status}, "
                            "but I will keep launching next iteration"
                        )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_WANNIER90_PLOT

        if not self.ctx.spin_collinear:
            self.ctx.current_folder = workchain.outputs.remote_folder
        else:
            self.ctx.current_folder_up = workchain[0].outputs.remote_folder
            self.ctx.current_folder_down = workchain[1].output.remote_folder
        return None

    def results(self):
        """Attach the relevant output nodes from the band calculation to the workchain outputs for convenience."""

        super().results()

        if self.inputs["optimize_disproj"]:
            if self.has_run_wannier90_optimize():
                optimal_workchain = self.ctx.optimize_best
                if "optimize_band_best" in self.ctx:
                    # Optimize succeed in workchain.
                    if self.ctx.optimize_band_best is not None:
                        # Override the output band_structure.
                        # Parent Workchain W90BandsWC output interpolated_bands from initial W90 run.
                        # And bands_plot is removed from separate_plot, so bands output will not be overrided there.
                        # I record the best optimized bands in inspect_wan_optimize_final, and override the output here
                        # if optimize workchain succeed and the best optimized band was recorded
                        self.out("band_structure", self.ctx.optimize_band_best)

            else:
                optimal_workchain = self.ctx.workchain_wannier90
            if not self.ctx.spin_collinear:
                self.out_many(
                    self.exposed_outputs(
                        optimal_workchain,
                        Wannier90BaseWorkChain,
                        namespace="wannier90_optimal",
                    )
                )
            else:
                self.out_many(
                    self.exposed_outputs(
                        optimal_workchain[0],
                        Wannier90BaseWorkChain,
                        namespace="wannier90_optimal_up",
                    )
                )
                self.out_many(
                    self.exposed_outputs(
                        optimal_workchain[1],
                        Wannier90BaseWorkChain,
                        namespace="wannier90_optimal_down",
                    )
                )

        if self.should_run_wannier90_plot():
            if not self.ctx.spin_collinear:
                self.out_many(
                    self.exposed_outputs(
                        self.ctx.workchain_wannier90_plot,
                        Wannier90BaseWorkChain,
                        namespace="wannier90_plot",
                    )
                )
                # Now band_plot has been removed from separate_plot
                # if "interpolated_bands" in self.outputs["wannier90_plot"]:
                #     w90_bands = self.outputs["wannier90_plot"]["interpolated_bands"]
                #     self.out("band_structure", w90_bands)
            else:
                self.out_many(
                    self.exposed_outputs(
                        self.ctx.workchain_wannier90_plot_up,
                        Wannier90BaseWorkChain,
                        namespace="wannier90_plot_up",
                    )
                )
                self.out_many(
                    self.exposed_outputs(
                        self.ctx.workchain_wannier90_plot_down,
                        Wannier90BaseWorkChain,
                        namespace="wannier90_plot_down",
                    )
                )

        if "optimize_reference_bands" in self.inputs:
            if self.has_run_wannier90_optimize():
                optimal_workchain = self.ctx.optimize_best
                bandsdist = self.ctx.optimize_bandsdist_best
            else:
                # Even if I haven't run optimization, I still output bands distance if reference bands is present
                optimal_workchain = self.ctx.workchain_wannier90
                bandsdist = self.ctx.workchain_wannier90_bandsdist
            bandsdist = orm.Float(bandsdist)
            bandsdist.store()
            self.out("bands_distance", bandsdist)

    def _get_bands_distance(
        self, wannier_workchain: ty.Union[Wannier90BaseWorkChain, list]
    ) -> float:
        """Get bands distance for Fermi energy + 2eV."""
        ref_bands = self.inputs["optimize_reference_bands"]
        if self.ctx.spin_collinear:
            bandsdist, spinor_bands = get_bands_distance_ef2(
                ref_bands, wannier_workchain
            )
            self.ctx.spinor_bands = spinor_bands
        else:
            bandsdist = get_bands_distance_ef2(ref_bands, wannier_workchain)[0]

        return bandsdist


def get_bands_distance_ef2(
    ref_bands: orm.BandsData, wannier_workchain: ty.Union[Wannier90BaseWorkChain, list]
) -> ty.Union[float, ty.Tuple[float, orm.BandsData]]:
    """Get bands distance for E <= Fermi energy + 2eV."""
    from aiida_wannier90_workflows.utils.bands.distance import bands_distance
    from aiida_wannier90_workflows.workflows.bands import get_spinor_band_structure

    if isinstance(
        wannier_workchain, list
    ):  # list of Wannier90BaseWorkChain for spin up and down
        wan_bands_up = wannier_workchain[0].outputs["interpolated_bands"]
        wan_bands_down = wannier_workchain[1].outputs["interpolated_bands"]
        structure = wannier_workchain[0].inputs["wannier90"]["structure"]
        wan_bands = get_spinor_band_structure(wan_bands_up, wan_bands_down, structure)
        wan_parameters = (
            wannier_workchain[0].inputs["wannier90"]["parameters"].get_dict()
        )
        #
    else:
        wan_bands = wannier_workchain.outputs["interpolated_bands"]
        wan_parameters = wannier_workchain.inputs["wannier90"]["parameters"].get_dict()

    fermi_energy = wan_parameters.get("fermi_energy")
    exclude_list_dft = wan_parameters.get("exclude_bands", None)

    # Bands distance from Ef to Ef+5
    bandsdist = bands_distance(ref_bands, wan_bands, fermi_energy, exclude_list_dft)
    # Only return average distance, not max distance
    bandsdist = bandsdist[:, 1]
    # Return Ef+2
    bandsdist = bandsdist[2]

    return bandsdist, wan_bands


def get_spreads_imbalence(wannier_functions_output: list) -> float:
    """Calculate the variance of spreads.

    There could be other ways to calculate the spreads imbalence, for now I just use variance.
    :param wannier_functions_output: [description]
    :type wannier_functions_output: dict (in fact list of dicts)
    :return: [description]
    :rtype: float
    """
    spreads = [_["wf_spreads"] for _ in wannier_functions_output]
    var = np.var(spreads)

    # TODO try K-Means clustering?  pylint: disable=fixme
    return var
