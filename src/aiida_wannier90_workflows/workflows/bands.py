"""WorkChain to automatically calculate Wannier band structure."""

import pathlib
import typing as ty

from aiida import orm
from aiida.engine import ProcessBuilder, if_
from aiida.orm.nodes.data.base import to_aiida_type

from .open_grid import Wannier90OpenGridWorkChain
from .wannier90 import Wannier90WorkChain

__all__ = ["validate_inputs", "Wannier90BandsWorkChain"]


def validate_inputs(  # pylint: disable=unused-argument,inconsistent-return-statements
    inputs, ctx=None
):
    """Validate the inputs of the entire input namespace of `Wannier90BandsWorkChain`."""
    from .open_grid import validate_inputs as parent_validate_inputs

    # Call parent validator
    result = parent_validate_inputs(inputs)
    if result is not None:
        return result

    # Cannot specify both `kpoint_path` and `bands_kpoints_distance`
    if (
        sum(
            _ in inputs
            for _ in ["kpoint_path", "bands_kpoints", "bands_kpoints_distance"]
        )
        > 1
    ):
        return "Can only specify one of the `kpoint_path`, `bands_kpoints` and `bands_kpoints_distance`."

    # `kpoint_path` and `bands_kpoints` must contain `labels`
    if "kpoint_path" in inputs:
        if inputs["kpoint_path"].labels is None:
            return "`kpoint_path` must contain `labels`"
    if "bands_kpoints" in inputs:
        if inputs["bands_kpoints"].labels is None:
            return "`bands_kpoints` must contain `labels`"


class Wannier90BandsWorkChain(Wannier90OpenGridWorkChain):
    """WorkChain to automatically compute a Wannier band structure for a given structure."""

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)

        spec.input(
            "kpoint_path",
            valid_type=orm.KpointsData,
            required=False,
            help=(
                "High symmetry kpoints to use for the wannier90 bands interpolation. "
                "If specified, the high symmetry kpoint labels will be used and wannier90 will use the "
                "`bands_num_points` mechanism to auto generate a list of kpoints along the kpath. "
                "If not specified, the workchain will run seekpath to generate "
                "a primitive cell and a bands_kpoints. Specify either this or `bands_kpoints` "
                "or `bands_kpoints_distance`."
            ),
        )
        spec.input(
            "bands_kpoints",
            valid_type=orm.KpointsData,
            required=False,
            help=(
                "Explicit kpoints to use for the wannier90 bands interpolation. "
                "If specified, wannier90 will use this list of kpoints and will not use the "
                "`bands_num_points` mechanism to auto generate a list of kpoints along the kpath. "
                "If not specified, the workchain will run seekpath to generate "
                "a primitive cell and a bands_kpoints. Specify either this or `bands_kpoints` "
                "or `bands_kpoints_distance`. "
                "This ensures the wannier interpolated bands has the exact same number of kpoints "
                "as PW bands, to calculate bands distance."
            ),
        )
        spec.input(
            "bands_kpoints_distance",
            valid_type=orm.Float,
            serializer=to_aiida_type,
            required=False,
            help="Minimum kpoints distance for seekpath to generate a list of kpoints along the path. "
            "Specify either this or `bands_kpoints` or `kpoint_path`.",
        )

        # We expose the in/output of `Wannier90OpenGridWorkChain` since `Wannier90WorkChain` in/output
        # is a subset of `Wannier90OpenGridWorkChain`, this allow us to launch either `Wannier90WorkChain`
        # or `Wannier90OpenGridWorkChain`.
        spec.expose_inputs(
            Wannier90OpenGridWorkChain,
            exclude=(
                "wannier90.wannier90.kpoint_path",
                "wannier90.wannier90.bands_kpoints",
            ),
            namespace_options={"required": True},
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
            cls.results,
        )

        spec.output(
            "primitive_structure",
            valid_type=orm.StructureData,
            required=False,
            help="The normalized and primitivized structure for which the calculations are computed.",
        )
        spec.output(
            "seekpath_parameters",
            valid_type=orm.Dict,
            required=False,
            help="The parameters used in the SeeKpath call to normalize the input or relaxed structure.",
        )
        spec.expose_outputs(
            Wannier90OpenGridWorkChain, namespace_options={"required": True}
        )
        spec.output(
            "band_structure",
            valid_type=orm.BandsData,
            help="The Wannier interpolated band structure.",
        )

    @classmethod
    def get_protocol_filepath(cls) -> pathlib.Path:
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files

        from . import protocols

        return files(protocols) / "bands.yaml"

    @classmethod
    def get_builder_from_protocol(  # pylint: disable=arguments-differ
        cls,
        codes: ty.Mapping[str, ty.Union[str, int, orm.Code]],
        structure: orm.StructureData,
        *,
        kpoint_path: orm.Dict = None,
        bands_kpoints: orm.KpointsData = None,
        bands_kpoints_distance: float = None,
        run_open_grid: bool = False,
        open_grid_only_scf: bool = True,
        **kwargs,
    ) -> ProcessBuilder:
        """Return a builder prepopulated with inputs selected according to the specified arguments.

        :param codes: a dictionary of codes for pw.x, pw2wannier90.x, wannier90.x, and optionally
        projwfc.x, open_grid.x.
        :type codes: ty.Mapping[str, ty.Union[str, int, orm.Code]]
        :param structure: the ``StructureData`` instance to use.
        :type structure: orm.StructureData
        :param kpoint_path: Explicit kpoints to use for the Wannier bands interpolation.
        If `kpoint_path` or `bands_kpoints` is provided, the workchain will directly generate input parameters
        for the structure and use the provided `KpointsData`, e.g. when one wants to Wannierise a conventional
        cell structure. If not provided, will use seekpath to generate a primitive cell, and generate input
        parameters for the PRIMITIVE cell. After submission of the workchian, a `seekpath_structure_analysis`
        calcfunction will be launched to store the provenance from non-primitive cell to primitive cell.
        In any case, the `get_builder_from_protocol` will NOT launch any calcfunction so the aiida database
        is kept unmodified. Defaults to None.
        :type kpoint_path: orm.KpointsData, optional
        :param bands_kpoints: Explicit kpoints to use for the Wannier bands interpolation. See `kpoint_path`.
        :type bands_kpoints: orm.KpointsData, optional
        :param bands_kpoints_distance: Minimum kpoints distance for the Wannier bands interpolation.
        Specify either this or `kpoint_path`. If not provided, will use the default of seekpath.
        Defaults to None
        :type bands_kpoints_distance: float, optional
        :param run_open_grid: if True use open_grid.x to accelerate calculations.
        :type run_open_grid: bool, defaults to False
        :param open_grid_only_scf: if True only one scf calculation will be performed in the OpenGridWorkChain.
        :type open_grid_only_scf: bool, defaults to True
        :return: a process builder instance with all inputs defined and ready for launch.
        :rtype: ProcessBuilder
        """
        from aiida.tools import get_explicit_kpoints_path

        from aiida_quantumespresso.common.types import SpinType

        summary = kwargs.pop("summary", {})
        print_summary = kwargs.pop("print_summary", True)

        if (
            sum(
                _ is not None
                for _ in (kpoint_path, bands_kpoints, bands_kpoints_distance)
            )
            > 1
        ):
            raise ValueError(
                "Can only specify one of the `kpoint_path`, `bands_kpoints` and `bands_kpoints_distance`"
            )

        inputs = cls.get_protocol_inputs(
            protocol=kwargs.get("protocol", None),
            overrides=kwargs.pop("overrides", None),
        )

        if run_open_grid and kwargs.get("spin_type", None) == SpinType.SPIN_ORBIT:
            raise ValueError("open_grid.x does not support spin orbit coupling")

        if run_open_grid:
            parent_class = Wannier90OpenGridWorkChain
            kwargs["open_grid_only_scf"] = open_grid_only_scf
        else:
            parent_class = Wannier90WorkChain

        if kpoint_path is None and bands_kpoints is None:
            # If no `kpoint_path` and `bands_kpoints` provided, the workchain will always run seekpath
            # even if the structure is a primitive cell.
            # However, if seekpath reduce the structure to primitive cell, then I need to populate the
            # builder with parameters for primitive cell, otherwise parameters depending on number of atoms
            # e.g. num_wann, num_bands are wrong!
            #
            # In principle, the cleanest way to run workflows is first run a bunch of
            # `seekpath_structure_analysis`, store the primitive structure and the corresponding kpath,
            # when launching `WannierBandsWorkChain` always use both structure and kpath for inputs.
            #
            # Note don't use `seekpath_structure_analysis`, since it's a calcfunction and will
            # modify aiida database!
            args = {"structure": structure}
            if bands_kpoints_distance:
                args["reference_distance"] = bands_kpoints_distance
            result = get_explicit_kpoints_path(**args)
            primitive_structure = result["primitive_structure"]
            # ase Atoms class can test if two structures are the same
            # if structure.get_ase() == primitive_structure.get_ase():
            if len(structure.sites) == len(primitive_structure.sites):
                parent_builder = parent_class.get_builder_from_protocol(
                    codes=codes,
                    structure=structure,
                    overrides=inputs,
                    **kwargs,
                    summary=summary,
                    print_summary=False,
                )
                # If set `kpoint_path`, the workchain won't run seekpath.
                # However, to be consistent, if no `kpoint_path` and `bands_kpoints` provided, I will
                # always run seekpath inside workchain.
                # parent_builder.kpoint_path = orm.Dict(
                #    dict={
                #        'path': result['parameters']['path'],
                #        'point_coords': result['parameters']['point_coords']
                #    }
                # )
                # parent_builder.kpoint_path = result['explicit_kpoints']
            else:
                notes = summary.get("notes", [])
                notes.append(
                    f"The input structure {structure.get_formula()}<{structure.pk}> is a supercell, "
                    "the auto generated parameters are for the primitive cell "
                    f"{primitive_structure.get_formula()} found by seekpath. "
                    "Although this is inconsistent, after submitting the workchain a seekpath run will "
                    "reduce the structure to primitive cell, so the Wannierisation is correct."
                )
                summary["notes"] = notes
                # I need to use primitive cell to generate all the input parameters, e.g. num_wann, num_bands, etc.
                parent_builder = parent_class.get_builder_from_protocol(
                    codes=codes,
                    structure=primitive_structure,
                    overrides=inputs,
                    **kwargs,
                    summary=summary,
                    print_summary=False,
                )
                # Don't set `kpoint_path` and `bands_kpoints_distance`, so the workchain will run seekpath.
                # However I still need to use the original cell, so the `seekpath_structure_analysis` will
                # store the provenance from original cell to primitive cell.
                parent_builder.structure = structure
        else:
            parent_builder = parent_class.get_builder_from_protocol(  # pylint: disable=too-many-function-args
                codes,
                structure,
                **kwargs,
                overrides=inputs,
                summary=summary,
                print_summary=False,
            )

        builder = cls.get_builder()
        builder._data = parent_builder._data  # pylint: disable=protected-access

        if kpoint_path:
            builder.kpoint_path = kpoint_path
        if bands_kpoints:
            builder.bands_kpoints = bands_kpoints

        if print_summary:
            cls.print_summary(summary)

        return builder

    def setup(self):
        """Define the current structure in the context to be the input structure."""
        from aiida_wannier90_workflows.utils.kpoints import get_path_from_kpoints

        super().setup()

        self.ctx.current_kpoint_path = None
        self.ctx.current_bands_kpoints = None

        if not self.should_run_seekpath():
            if "kpoint_path" in self.inputs:
                self.ctx.current_kpoint_path = get_path_from_kpoints(
                    self.inputs.kpoint_path
                )

            if "bands_kpoints" in self.inputs:
                self.ctx.current_bands_kpoints = self.inputs.bands_kpoints

    def should_run_seekpath(self):
        """Seekpath should only be run if the `kpoint_path` or `bands_kpoints` input is not specified."""
        return not any(_ in self.inputs for _ in ("kpoint_path", "bands_kpoints"))

    def run_seekpath(self):
        """Run the structure through SeeKpath to get the primitive and normalized structure."""
        from aiida_quantumespresso.calculations.functions.seekpath_structure_analysis import (
            seekpath_structure_analysis,
        )

        args = {
            "structure": self.inputs.structure,
            "metadata": {"call_link_label": "seekpath_structure_analysis"},
        }
        if "bands_kpoints_distance" in self.inputs:
            args["reference_distance"] = self.inputs["bands_kpoints_distance"]

        result = seekpath_structure_analysis(**args)

        self.ctx.current_structure = result["primitive_structure"]

        # Add `kpoint_path` for Wannier bands
        self.ctx.current_kpoint_path = orm.Dict(
            dict={
                "path": result["parameters"]["path"],
                "point_coords": result["parameters"]["point_coords"],
            }
        )

        structure_formula = self.inputs.structure.get_formula()
        primitive_structure_formula = result["primitive_structure"].get_formula()
        self.report(
            f"launching seekpath: {structure_formula} -> {primitive_structure_formula}"
        )

        self.out("primitive_structure", result["primitive_structure"])
        self.out("seekpath_parameters", result["parameters"])

    def prepare_wannier90_pp_inputs(self):
        """Override parent method."""
        base_inputs = super().prepare_wannier90_pp_inputs()
        inputs = base_inputs["wannier90"]

        parameters = inputs.parameters.get_dict()
        parameters["bands_plot"] = True
        inputs.parameters = orm.Dict(parameters)

        if self.ctx.current_kpoint_path:
            inputs.kpoint_path = self.ctx.current_kpoint_path
        if self.ctx.current_bands_kpoints:
            inputs.bands_kpoints = self.ctx.current_bands_kpoints

        base_inputs["wannier90"] = inputs
        return base_inputs

    def results(self):
        """Attach the relevant output nodes from the band calculation to the workchain outputs for convenience."""
        super().results()

        if "interpolated_bands" in self.outputs["wannier90"]:
            w90_bands = self.outputs["wannier90"]["interpolated_bands"]
            self.out("band_structure", w90_bands)
