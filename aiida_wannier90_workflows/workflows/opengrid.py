# -*- coding: utf-8 -*-
"""Wannierisation workflow using open_grid.x to bypass the nscf step."""
import typing as ty
import pathlib

from aiida import orm
from aiida.common import AttributeDict
from aiida.engine.processes import ToContext, if_, ProcessBuilder

from aiida_quantumespresso.utils.mapping import prepare_process_inputs

from .base.opengrid import OpengridBaseWorkChain
from .wannier90 import Wannier90WorkChain

__all__ = ('validate_inputs', 'Wannier90OpengridWorkChain')


def validate_inputs(inputs, ctx=None):  # pylint: disable=unused-argument
    """Validate the inputs of the entire input namespace of `Wannier90OpengridWorkChain`."""
    from .wannier90 import validate_inputs as parent_validate_inputs

    # Call parent validator
    result = parent_validate_inputs(inputs)

    if result is not None:
        return result


class Wannier90OpengridWorkChain(Wannier90WorkChain):
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
            OpengridBaseWorkChain,
            namespace='opengrid',
            exclude=('clean_workdir', 'opengrid.parent_folder'),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the `OpengridBaseWorkChain`, if not specified the opengrid step is skipped.'
            }
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
            if_(cls.should_run_opengrid)(
                cls.run_opengrid,
                cls.inspect_opengrid,
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

        spec.expose_outputs(OpengridBaseWorkChain, namespace='opengrid', namespace_options={'required': False})

        spec.exit_code(490, 'ERROR_SUB_PROCESS_FAILED_OPENGRID', message='the OpengridBaseWorkChain sub process failed')

    @classmethod
    def get_protocol_filepath(cls) -> pathlib.Path:
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files
        from . import protocols
        return files(protocols) / 'opengrid.yaml'

    @classmethod
    def get_builder_from_protocol(  # pylint: disable=arguments-differ
        cls,
        *,
        codes: ty.Mapping[str, ty.Union[str, int, orm.Code]],
        opengrid_only_scf: bool = True,
        **kwargs
    ) -> ProcessBuilder:
        """Return a builder populated with predefined inputs that can be directly submitted.

        Optional keyword arguments are passed to the same function of `Wannier90WorkChain`.
        Overrides `Wannier90WorkChain` workchain.

        :param codes: [description]
        :type codes: ty.Mapping[str, ty.Union[str, int, orm.Code]]
        :param opengrid_only_scf: If True first do a scf with symmetry and increased number of bands,
        then launch open_grid.x to unfold kmesh; If False first do a scf with symmetry and default
        number of bands, then a nscf with symmetry and increased number of bands, followed by open_grid.x.
        :type opengrid_only_scf: bool
        """
        from aiida_wannier90_workflows.utils.workflows.builder import recursive_merge_builder

        summary = kwargs.pop('summary', {})
        print_summary = kwargs.pop('print_summary', True)

        # Prepare workchain builder
        builder = cls.get_builder()

        inputs = cls.get_protocol_inputs(protocol=kwargs.get('protocol', None), overrides=kwargs.get('overrides', None))
        builder = recursive_merge_builder(builder, inputs)

        parent_builder = super().get_builder_from_protocol(codes=codes, summary=summary, print_summary=False, **kwargs)
        inputs = parent_builder._inputs(prune=True)  # pylint: disable=protected-access
        builder = recursive_merge_builder(builder, inputs)

        # Adapt pw.x parameters
        if opengrid_only_scf:
            nbnd = builder.nscf['pw']['parameters'].get_dict()['SYSTEM'].get('nbnd', None)
            params = builder.scf['pw']['parameters'].get_dict()
            if nbnd is not None:
                params['SYSTEM']['nbnd'] = nbnd
            params['SYSTEM'].pop('nosym', None)
            params['SYSTEM'].pop('noinv', None)
            params['ELECTRONS']['diago_full_acc'] = True
            builder.scf['pw']['parameters'] = orm.Dict(dict=params)
            builder.nscf.clear()
        else:
            params = builder.nscf['pw']['parameters'].get_dict()
            params['SYSTEM'].pop('nosym', None)
            params['SYSTEM'].pop('noinv', None)
            builder.nscf['pw']['parameters'] = orm.Dict(dict=params)
            builder.nscf.pop('kpoints', None)
            builder.nscf['kpoints_distance'] = builder.scf['kpoints_distance']
            builder.nscf['kpoints_force_parity'] = builder.scf['kpoints_force_parity']

        # Prepare opengrid
        opengrid_overrides = kwargs.get('overrides', {}).get('opengrid', {})
        opengrid_builder = OpengridBaseWorkChain.get_builder_from_protocol(
            code=codes['opengrid'], protocol=kwargs.get('protocol', None), overrides=opengrid_overrides
        )
        # Remove workchain excluded inputs
        opengrid_builder.pop('clean_workdir', None)
        builder.opengrid = opengrid_builder._inputs(prune=True)  # pylint: disable=protected-access

        if print_summary:
            cls.print_summary(summary)

        return builder

    def should_run_opengrid(self):
        """If the 'opengrid' input namespace was specified, we run opengrid after scf or nscf calculation."""
        return 'opengrid' in self.inputs

    def run_opengrid(self):
        """Use QE open_grid.x to unfold irriducible kmesh to a full kmesh."""
        inputs = AttributeDict(self.exposed_inputs(OpengridBaseWorkChain, namespace='opengrid'))
        inputs.opengrid.parent_folder = self.ctx.current_folder
        inputs.metadata.call_link_label = 'opengrid'

        inputs = prepare_process_inputs(OpengridBaseWorkChain, inputs)
        running = self.submit(OpengridBaseWorkChain, **inputs)
        self.report(f'launching {running.process_label}<{running.pk}>')

        return ToContext(workchain_opengrid=running)

    def inspect_opengrid(self):
        """Verify that the `OpengridBaseWorkChain` run successfully finished."""
        workchain = self.ctx.workchain_opengrid

        if not workchain.is_finished_ok:
            self.report(f'{workchain.process_label} failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_OPENGRID

        self.ctx.current_folder = workchain.outputs.remote_folder

    def prepare_wannier90_inputs(self):
        """Override the parent method in `Wannier90WorkChain`.

        The wannier input kpoints are set as the parsed output from `OpengridBaseWorkChain`.
        """
        base_inputs = super().prepare_wannier90_inputs()
        inputs = base_inputs['wannier90']

        if self.should_run_opengrid():
            opengrid_outputs = self.ctx.workchain_opengrid.outputs
            inputs.kpoints = opengrid_outputs.kpoints
            parameters = inputs.parameters.get_dict()
            parameters['mp_grid'] = opengrid_outputs.kpoints_mesh.get_kpoints_mesh()[0]
            inputs.parameters = orm.Dict(dict=parameters)

        base_inputs['wannier90'] = inputs

        return base_inputs

    def results(self):
        """Override parent workchain."""
        if self.should_run_opengrid():
            self.out_many(
                self.exposed_outputs(self.ctx.workchain_opengrid, OpengridBaseWorkChain, namespace='opengrid')
            )

        super().results()
