from aiida import orm
from aiida.common import AttributeDict
from aiida.engine.processes import ToContext, if_, ProcessBuilder

from aiida_quantumespresso.utils.mapping import prepare_process_inputs
from aiida_quantumespresso.calculations.opengrid import OpengridCalculation

from .wannier import Wannier90WorkChain

__all__ = ('Wannier90OpengridWorkChain', )

def validate_inputs(inputs, ctx=None):  # pylint: disable=unused-argument
    """Validate the inputs of the entire input namespace."""
    # pylint: disable=no-member
    from aiida_wannier90_workflows.workflows.wannier import validate_inputs as parent_validator
    result = parent_validator(inputs)
    if result is not None:
        return result


class Wannier90OpengridWorkChain(Wannier90WorkChain):
    """This WorkChain uses open_grid.x to unfold the 
    symmetrized kmesh to a full kmesh in the Wannier90WorkChain.
    The full-kmesh nscf can be avoided.

    2 schemes:
    1. scf w/ symmetry, more nbnd -> open_grid 
       -> pw2wannier90 -> wannier90
    2. scf w/ symmetry, default nbnd -> nscf w/ symm, more nbnd 
       -> open_grid -> pw2wannier90 -> wannier90
    """
    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.expose_inputs(
            OpengridCalculation,
            namespace='opengrid',
            exclude=('parent_folder', 'structure'),
            namespace_options={
                'required': False,
                'populate_defaults':
                False,
                'help': 'Inputs for the `OpengridCalculation`, if not specified the opengrid step is skipped.'
            }
        )
        spec.inputs.validator = validate_inputs

        spec.outline(
            cls.setup,
            if_(cls.should_run_relax)(
                cls.run_relax,
                cls.inspect_relax,
            ),
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
                cls.inspect_projwfc
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
            OpengridCalculation,
            namespace='opengrid',
            namespace_options={
                'required': False,
            }
        )

        spec.exit_code(
            480,
            'ERROR_SUB_PROCESS_FAILED_OPENGRID',
            message='the OpengridCalculation sub process failed'
        )

    def should_run_opengrid(self):
        """If the 'opengrid' input namespace was specified, we run opengrid after scf or nscf calculation."""
        return 'opengrid' in self.inputs

    def run_opengrid(self):
        """Use QE open_grid.x to unfold irriducible kmesh to a full kmesh"""
        inputs = AttributeDict(
            self.exposed_inputs(OpengridCalculation, namespace='opengrid')
        )
        inputs.metadata.call_link_label = 'opengrid'

        inputs.parent_folder = self.ctx.current_folder
        # ProjwfcCalculation requires the previous Calculation having
        # an input StructureData to parse atomic orbitals.
        inputs.structure = self.ctx.current_structure

        inputs = prepare_process_inputs(OpengridCalculation, inputs)
        running = self.submit(OpengridCalculation, **inputs)
        self.report(f'launching {running.process_label}<{running.pk}>')

        return ToContext(calc_opengrid=running)

    def inspect_opengrid(self):
        """Verify that the OpengridCalculation run successfully finished."""
        workchain = self.ctx.calc_opengrid

        if not workchain.is_finished_ok:
            self.report(
                f'{workchain.process_label} failed with exit status {workchain.exit_status}'
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_OPENGRID

        self.ctx.current_folder = workchain.outputs.remote_folder

    def prepare_wannier90_inputs(self):
        """Overrides the parent method in Wannier90WorkChain.
        The wannier input kpoints are set as the parsed output from OpengridCalculation."""
        inputs = super().prepare_wannier90_inputs()

        if self.should_run_opengrid():
            opengrid_outputs = self.ctx.calc_opengrid.outputs
            inputs.kpoints = opengrid_outputs.kpoints
            parameters = inputs.parameters.get_dict()
            parameters['mp_grid'] = opengrid_outputs.kpoints_mesh.get_kpoints_mesh()[0]
            inputs.parameters = orm.Dict(dict=parameters)

        return inputs

    def results(self):
        """Overrides parent workchain"""
        if self.should_run_opengrid():
            self.out_many(
                self.exposed_outputs(
                    self.ctx.calc_opengrid,
                    OpengridCalculation,
                    namespace='opengrid'
                )
            )

        super().results()

    @classmethod
    def get_builder_from_protocol(
        cls,
        codes: dict,
        structure: orm.StructureData,
        *,
        opengrid_only_scf: bool = True,
        **kwargs
    ) -> ProcessBuilder:
        """Reture a builder populated with predefined inputs that can be directly submitted.
        Optional keyword arguments are passed to the same function of Wannier90WorkChain.
        
        Overrides Wannier90WorkChain workchain.

        :param codes: [description]
        :type codes: [type]
        :param structure: [description]
        :type structure: [type]
        :param opengrid_only_scf: If True first do a scf with symmetry and increased number of bands, then open_grid to unfold kmesh; If False first do a scf with symmetry and default number of bands, then a nscf with symmetry and increased number of bands, followed by open_grid.
        :type opengrid_only_scf: [type]
        """
        
        summary = kwargs.pop('summary', {})
        print_summary = kwargs.pop('print_summary', True)

        builder = super().get_builder_from_protocol(
            codes, structure, **kwargs,
            summary=summary, print_summary=False
        )

        if opengrid_only_scf:
            nbnd = builder.nscf['pw']['parameters'].get_dict()['SYSTEM'].get(
                'nbnd', None
            )
            params = builder.scf['pw']['parameters'].get_dict()
            has_modified = False
            if nbnd is not None:
                nbnd_scf = params['SYSTEM'].get('nbnd', None)
                if nbnd_scf != nbnd:
                    params['SYSTEM']['nbnd'] = nbnd
                    has_modified = True
            nosym = params['SYSTEM'].pop('nosym', None)
            noinv = params['SYSTEM'].pop('noinv', None)
            if nosym is not None or noinv is not None:
                has_modified = True
            if has_modified:
                builder.scf['pw']['parameters'] = orm.Dict(dict=params)
            builder.nscf.clear()
        else:
            params = builder.nscf['pw']['parameters'].get_dict()
            has_modified = False
            nosym = params['SYSTEM'].pop('nosym', None)
            noinv = params['SYSTEM'].pop('noinv', None)
            if nosym is not None or noinv is not None:
                has_modified = True
            if has_modified:
                builder.nscf['pw']['parameters'] = orm.Dict(dict=params)
            builder.nscf.pop('kpoints', None)
            builder.nscf['kpoints_distance'] = builder.scf['kpoints_distance']
            builder.nscf['kpoints_force_parity'] = builder.scf['kpoints_force_parity']

        builder.opengrid = {
            'code': codes['opengrid'],
            'parameters': orm.Dict(dict={
                'inputpp': {
                    # open_grid.x should not append '_open' to the prefix of QE
                    'overwrite_prefix': True
                }
            }),
            'metadata': {
                'options': {
                    'resources': {
                        'num_machines': 1
                    }
                },
            }
        }

        notes = summary.get('notes', [])
        notes.append('The open_grid.x unfolded kmesh will be used as wannier90 input kpoints.')
        summary['notes'] = notes

        if print_summary:
            cls.print_summary(summary)

        return builder
