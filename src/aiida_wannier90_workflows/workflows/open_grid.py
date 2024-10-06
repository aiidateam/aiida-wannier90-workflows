"""Wannierisation workflow using open_grid.x to bypass the nscf step."""

# pylint: disable=protected-access
import pathlib
import typing as ty

from aiida import orm
from aiida.common import AttributeDict
from aiida.engine.processes import ProcessBuilder, ToContext, if_

from aiida_quantumespresso.utils.mapping import prepare_process_inputs

from .base.open_grid import OpenGridBaseWorkChain
from .wannier90 import Wannier90WorkChain

__all__ = ("validate_inputs", "Wannier90OpenGridWorkChain")


def validate_inputs(  # pylint: disable=unused-argument,inconsistent-return-statements
    inputs, ctx=None
):
    """Validate the inputs of the entire input namespace of `Wannier90OpenGridWorkChain`."""
    from .wannier90 import validate_inputs as parent_validate_inputs

    # Call parent validator
    result = parent_validate_inputs(inputs)

    if result is not None:
        return result


class Wannier90OpenGridWorkChain(Wannier90WorkChain):
    """WorkChain using open_grid.x to bypass the nscf step.

    The open_grid.x unfolds the symmetrized kmesh to a full kmesh, thus
    the full-kmesh nscf step can be avoided.

    2 schemes:
      1. scf w/ symmetry, more nbnd -> open_grid -> pw2wannier90 -> wannier90
      2. scf w/ symmetry, default nbnd -> nscf w/ symm, more nbnd -> open_grid
         -> pw2wannier90 -> wannier90
    """

    @classmethod
    def define(cls, spec):
        """Define the process spec."""
        super().define(spec)

        spec.expose_inputs(
            OpenGridBaseWorkChain,
            namespace="open_grid",
            exclude=("clean_workdir", "open_grid.parent_folder"),
            namespace_options={
                "required": False,
                "populate_defaults": False,
                "help": "Inputs for the `OpenGridBaseWorkChain`, if not specified the open_grid step is skipped.",
            },
        )
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

        spec.expose_outputs(
            OpenGridBaseWorkChain,
            namespace="open_grid",
            namespace_options={"required": False},
        )

        spec.exit_code(
            490,
            "ERROR_SUB_PROCESS_FAILED_OPEN_GRID",
            message="the OpenGridBaseWorkChain sub process failed",
        )

    @classmethod
    def get_protocol_filepath(cls) -> pathlib.Path:
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files

        from . import protocols

        return files(protocols) / "open_grid.yaml"

    @classmethod
    def get_builder_from_protocol(  # pylint: disable=arguments-differ
        cls,
        codes: ty.Mapping[str, ty.Union[str, int, orm.Code]],
        structure: orm.StructureData,
        *,
        open_grid_only_scf: bool = True,
        **kwargs,
    ) -> ProcessBuilder:
        """Return a builder populated with predefined inputs that can be directly submitted.

        Optional keyword arguments are passed to the same function of `Wannier90WorkChain`.
        Overrides `Wannier90WorkChain` workchain.

        :param codes: [description]
        :type codes: ty.Mapping[str, ty.Union[str, int, orm.Code]]
        :param open_grid_only_scf: If True first do a scf with symmetry and increased number of bands,
        then launch open_grid.x to unfold kmesh; If False first do a scf with symmetry and default
        number of bands, then a nscf with symmetry and increased number of bands, followed by open_grid.x.
        :type open_grid_only_scf: bool
        """
        protocol = kwargs.pop("protocol", None)
        overrides = kwargs.pop("overrides", None)
        summary = kwargs.pop("summary", {})
        print_summary = kwargs.pop("print_summary", True)

        # Prepare workchain builder
        builder = cls.get_builder()

        inputs = cls.get_protocol_inputs(protocol, overrides)

        parent_builder = Wannier90WorkChain.get_builder_from_protocol(
            codes,
            structure,
            summary=summary,
            print_summary=False,
            overrides=inputs,
            **kwargs,
        )
        builder._data = parent_builder._data

        if open_grid_only_scf:
            # Remove `nscf` step and adapt `scf` parameters accordingly
            nscf_params = builder.pop("nscf")["pw"]["parameters"].get_dict()
            nbnd = nscf_params["SYSTEM"].get("nbnd", None)

            scf_params = builder.scf.pw.parameters.get_dict()

            if nbnd is not None:
                scf_params["SYSTEM"]["nbnd"] = nbnd

            scf_params["SYSTEM"].pop("nosym", None)
            scf_params["SYSTEM"].pop("noinv", None)
            scf_params["ELECTRONS"]["diago_full_acc"] = True
            builder.scf.pw.parameters = orm.Dict(scf_params)
        else:
            params = builder.nscf.pw.parameters.get_dict()

            params["SYSTEM"].pop("nosym", None)
            params["SYSTEM"].pop("noinv", None)

            builder.nscf.pw.parameters = orm.Dict(params)

            builder.nscf.pop("kpoints", None)
            builder.nscf.kpoints_distance = builder.scf.kpoints_distance
            builder.nscf.kpoints_force_parity = builder.scf.kpoints_force_parity

        # Prepare open_grid builder
        open_grid_overrides = inputs.get("open_grid", {})
        open_grid_builder = OpenGridBaseWorkChain.get_builder_from_protocol(
            code=codes["open_grid"],
            protocol=protocol,
            overrides=open_grid_overrides,
        )
        # Remove workchain excluded inputs
        open_grid_builder.pop("clean_workdir", None)
        builder.open_grid = open_grid_builder._inputs(prune=True)

        if print_summary:
            cls.print_summary(summary)

        return builder

    def should_run_open_grid(self):
        """If the 'open_grid' input namespace was specified, we run open_grid after scf or nscf calculation."""
        return "open_grid" in self.inputs

    def run_open_grid(self):
        """Use QE open_grid.x to unfold irreducible kmesh to a full kmesh."""
        inputs = AttributeDict(
            self.exposed_inputs(OpenGridBaseWorkChain, namespace="open_grid")
        )
        inputs.open_grid.parent_folder = self.ctx.current_folder
        inputs.metadata.call_link_label = "open_grid"

        inputs = prepare_process_inputs(OpenGridBaseWorkChain, inputs)
        running = self.submit(OpenGridBaseWorkChain, **inputs)
        self.report(f"launching {running.process_label}<{running.pk}>")

        return ToContext(workchain_open_grid=running)

    def inspect_open_grid(self):  # pylint: disable=inconsistent-return-statements
        """Verify that the `OpenGridBaseWorkChain` run successfully finished."""
        workchain = self.ctx.workchain_open_grid

        if not workchain.is_finished_ok:
            self.report(
                f"{workchain.process_label} failed with exit status {workchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_OPEN_GRID

        self.ctx.current_folder = workchain.outputs.remote_folder

    def prepare_wannier90_pp_inputs(self):
        """Override the parent method in `Wannier90WorkChain`.

        The wannier input kpoints are set as the parsed output from `OpenGridBaseWorkChain`.
        """
        base_inputs = super().prepare_wannier90_pp_inputs()
        inputs = base_inputs["wannier90"]

        if self.should_run_open_grid():
            open_grid_outputs = self.ctx.workchain_open_grid.outputs
            inputs.kpoints = open_grid_outputs.kpoints
            parameters = inputs.parameters.get_dict()
            parameters["mp_grid"] = open_grid_outputs.kpoints_mesh.get_kpoints_mesh()[0]
            inputs.parameters = orm.Dict(parameters)

        base_inputs["wannier90"] = inputs

        return base_inputs

    def results(self):
        """Override parent workchain."""
        if self.should_run_open_grid():
            self.out_many(
                self.exposed_outputs(
                    self.ctx.workchain_open_grid,
                    OpenGridBaseWorkChain,
                    namespace="open_grid",
                )
            )

        super().results()
