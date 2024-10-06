"""WorkChain to automatically calculate QE projected band structure."""

import pathlib
import typing as ty

from aiida import orm
from aiida.common import AttributeDict
from aiida.common.lang import type_check
from aiida.engine import ProcessBuilder, ToContext, if_

from aiida_quantumespresso.utils.mapping import prepare_process_inputs
from aiida_quantumespresso.workflows.pw.bands import PwBandsWorkChain

from .base.projwfc import ProjwfcBaseWorkChain

__all__ = ["validate_inputs", "ProjwfcBandsWorkChain"]


def validate_inputs(  # pylint: disable=unused-argument,inconsistent-return-statements
    inputs, ctx=None
):
    """Validate the inputs of the entire input namespace of `ProjwfcBandsWorkChain`."""
    from aiida_quantumespresso.workflows.pw.bands import (
        validate_inputs as parent_validate_inputs,
    )

    # Call parent validator
    result = parent_validate_inputs(inputs)
    if result is not None:
        return result


class ProjwfcBandsWorkChain(PwBandsWorkChain):
    """WorkChain to compute QE projected band structure for a given structure."""

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)

        spec.expose_inputs(
            ProjwfcBaseWorkChain,
            namespace="projwfc",
            exclude=("clean_workdir", "projwfc.parent_folder"),
            namespace_options={
                "help": "Inputs for the `ProjwfcBaseWorkChain` for the projwfc.x calculation."
            },
        )
        spec.inputs.validator = validate_inputs

        spec.outline(
            cls.setup,
            if_(cls.should_run_relax)(
                cls.run_relax,
                cls.inspect_relax,
            ),
            if_(cls.should_run_seekpath)(
                cls.run_seekpath,
            ),
            cls.run_scf,
            cls.inspect_scf,
            cls.run_bands,
            cls.inspect_bands,
            cls.run_projwfc,
            cls.inspect_projwfc,
            cls.results,
        )

        spec.expose_outputs(ProjwfcBaseWorkChain, namespace="projwfc")

        spec.exit_code(
            404,
            "ERROR_SUB_PROCESS_FAILED_PROJWFC",
            message="The ProjwfcBaseWorkChain sub process failed",
        )

    @classmethod
    def get_protocol_filepath(cls) -> pathlib.Path:
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files

        from . import protocols

        return files(protocols) / "projwfcbands.yaml"

    @classmethod
    def get_builder_from_protocol(  # pylint: disable=arguments-differ
        cls,
        pw_code: ty.Union[str, int, orm.Code],
        projwfc_code: ty.Union[str, int, orm.Code],
        structure: orm.StructureData,
        *,
        protocol: str = None,
        overrides: dict = None,
        **kwargs,
    ) -> ProcessBuilder:
        """Return a builder prepopulated with inputs selected according to the specified arguments.

        :param pw_code: code for pw.x.
        :type pw_code: ty.Union[str, int, orm.Code]
        :param projwfc_code: code for projwfc.x.
        :type projwfc_code: ty.Union[str, int, orm.Code]
        :param structure: the ``StructureData`` instance to use.
        :type structure: orm.StructureData
        :return: a process builder instance with all inputs defined and ready for launch.
        :rtype: ProcessBuilder
        """
        from aiida_wannier90_workflows.utils.workflows.builder.submit import (
            recursive_merge_builder,
        )

        type_check(pw_code, (str, int, orm.Code))
        type_check(projwfc_code, (str, int, orm.Code))
        type_check(structure, orm.StructureData)
        type_check(protocol, str, allow_none=True)
        type_check(overrides, dict, allow_none=True)

        # Prepare workchain builder
        builder = cls.get_builder()

        protocol_inputs = cls.get_protocol_inputs(
            protocol=protocol, overrides=overrides
        )

        projwfc_overrides = protocol_inputs.pop("projwfc", None)

        pwbands_builder = PwBandsWorkChain.get_builder_from_protocol(
            code=pw_code,
            structure=structure,
            protocol=protocol,
            overrides=protocol_inputs,
            **kwargs,
        )

        # By default do not run relax
        pwbands_builder.pop("relax", None)

        projwfc_builder = ProjwfcBaseWorkChain.get_builder_from_protocol(
            projwfc_code, protocol=protocol, overrides=projwfc_overrides
        )
        projwfc_builder.pop("clean_workdir", None)

        builder.projwfc = projwfc_builder
        builder = recursive_merge_builder(builder, pwbands_builder)

        return builder

    def run_projwfc(self):
        """Run projwfc.x."""
        inputs = AttributeDict(
            self.exposed_inputs(ProjwfcBaseWorkChain, namespace="projwfc")
        )
        inputs.metadata.call_link_label = "projwfc"
        inputs.clean_workdir = orm.Bool(False)

        inputs.projwfc.parent_folder = self.ctx.workchain_bands.outputs.remote_folder

        inputs = prepare_process_inputs(ProjwfcBaseWorkChain, inputs)
        running = self.submit(ProjwfcBaseWorkChain, **inputs)

        self.report(f"launching {running.process_label}<{running.pk}>")

        return ToContext(workchain_projwfc=running)

    def inspect_projwfc(self):  # pylint: disable=inconsistent-return-statements
        """Verify that the ProjwfcBaseWorkChain for the projwfc.x run finished successfully."""
        workchain = self.ctx.workchain_projwfc

        if not workchain.is_finished_ok:
            self.report(
                f"ProjwfcBaseWorkChain failed with exit status {workchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_PROJWFC

    def results(self):
        """Attach the relevant output nodes."""
        super().results()

        self.out_many(
            self.exposed_outputs(
                self.ctx.workchain_projwfc, ProjwfcBaseWorkChain, namespace="projwfc"
            )
        )
