# -*- coding: utf-8 -*-
"""WorkChain to automatically calculate wannier band structure."""
from aiida import orm
from aiida.engine import if_, ProcessBuilder
from aiida_quantumespresso.calculations.functions.seekpath_structure_analysis import seekpath_structure_analysis
from aiida_quantumespresso.common.types import SpinType
from aiida_wannier90_workflows.workflows.opengrid import Wannier90OpengridWorkChain
from aiida_wannier90_workflows.utils.kmesh import get_path_from_kpoints

__all__ = ['Wannier90BandsWorkChain', 'get_builder_for_pwbands']


class Wannier90BandsWorkChain(Wannier90OpengridWorkChain):
    """WorkChain to automatically compute a Wannier band structure for a given structure."""

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)
        spec.input(
            'kpoint_path',
            valid_type=orm.KpointsData,
            required=False,
            help=(
                'High symmetry kpoints to use for the wannier90 bands interpolation. '
                'If specified, the high symmetry kpoint labels will be used and wannier90 will use the '
                '`bands_num_points` mechanism to auto generate a list of kpoints along the kpath. '
                'If not specified, the workchain will run seekpath to generate '
                'a primitive cell and a bands_kpoints. Specify either this or `bands_kpoints` '
                'or `kpoint_path_distance`.'
            )
        )
        spec.input(
            'bands_kpoints',
            valid_type=orm.KpointsData,
            required=False,
            help=(
                'Explicit kpoints to use for the wannier90 bands interpolation. '
                'If specified, wannier90 will use this list of kpoints and will not use the '
                '`bands_num_points` mechanism to auto generate a list of kpoints along the kpath. '
                'If not specified, the workchain will run seekpath to generate '
                'a primitive cell and a bands_kpoints. Specify either this or `bands_kpoints` '
                'or `kpoint_path_distance`. '
                'This ensures the wannier interpolated bands has the exact same number of kpoints '
                'as PW bands, to calculate bands distance.'
            )
        )
        spec.input(
            'kpoint_path_distance',  # TODO rename as bands_kpoints_distance  pylint: disable=fixme
            valid_type=orm.Float,
            required=False,
            help='Minimum kpoints distance for seekpath to generate a list of kpoints along the path. '
            'Specify either this or `bands_kpoints` or `kpoint_path`.'
        )

        # We expose the in/output of Wannier90OpengridWorkChain since Wannier90WorkChain in/output
        # is a subset of Wannier90OpengridWorkChain,
        # this allow us to launch either Wannier90WorkChain or Wannier90OpengridWorkChain.
        spec.expose_inputs(
            Wannier90OpengridWorkChain,
            exclude=('wannier90.wannier90.kpoint_path', 'wannier90.wannier90.bands_kpoints'),
            namespace_options={'required': True}
        )
        spec.inputs.validator = cls.validate_inputs

        spec.outline(
            cls.setup,
            if_(cls.should_run_seekpath)(cls.run_seekpath,),
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
            'primitive_structure',
            valid_type=orm.StructureData,
            required=False,
            help='The normalized and primitivized structure for which the calculations are computed.'
        )
        spec.output(
            'seekpath_parameters',
            valid_type=orm.Dict,
            required=False,
            help='The parameters used in the SeeKpath call to normalize the input or relaxed structure.'
        )
        spec.expose_outputs(Wannier90OpengridWorkChain, namespace_options={'required': True})
        spec.output('band_structure', valid_type=orm.BandsData, help='The Wannier interpolated band structure.')

    @staticmethod
    def validate_inputs(inputs, ctx=None):  # pylint: disable=unused-argument
        """Validate the inputs of the entire input namespace."""
        # Call parent validator
        result = Wannier90OpengridWorkChain.validate_inputs(inputs)
        if result is not None:
            return result

        # Cannot specify both `kpoint_path` and `kpoint_path_distance`
        if sum(key in inputs for key in ['kpoint_path', 'bands_kpoints', 'kpoint_path_distance']) > 1:
            return 'Can only specify one of the `kpoint_path`, `bands_kpoints` and `kpoint_path_distance`.'

        # `kpoint_path` must contain `labels`
        if 'kpoint_path' in inputs:
            if inputs['kpoint_path'].labels is None:
                return '`kpoint_path` must contain `labels`'
        if 'bands_kpoints' in inputs:
            if inputs['bands_kpoints'].labels is None:
                return '`bands_kpoints` must contain `labels`'

    def setup(self):
        """Define the current structure in the context to be the input structure."""
        super().setup()

        if not self.should_run_seekpath():
            self.ctx.current_kpoint_path = None
            if 'kpoint_path' in self.inputs:
                self.ctx.current_kpoint_path = get_path_from_kpoints(self.inputs.kpoint_path)

            self.ctx.current_bands_kpoints = None
            if 'bands_kpoints' in self.inputs:
                self.ctx.current_bands_kpoints = self.inputs.bands_kpoints

    def should_run_seekpath(self):
        """Seekpath should only be run if the `kpoint_path` or `bands_kpoints` input is not specified."""
        return not any(_ in self.inputs for _ in ('kpoint_path', 'bands_kpoints'))

    def run_seekpath(self):
        """Run the structure through SeeKpath to get the primitive and normalized structure."""
        structure_formula = self.inputs.structure.get_formula()
        self.report(f'launching seekpath for: {structure_formula}')

        args = {'structure': self.inputs.structure, 'metadata': {'call_link_label': 'seekpath_structure_analysis'}}
        if 'kpoint_path_distance' in self.inputs:
            args['reference_distance'] = self.inputs['kpoint_path_distance']

        result = seekpath_structure_analysis(**args)

        self.ctx.current_structure = result['primitive_structure']

        # add kpoint_path for Wannier bands
        self.ctx.current_kpoint_path = orm.Dict(
            dict={
                'path': result['parameters']['path'],
                'point_coords': result['parameters']['point_coords']
            }
        )

        self.out('primitive_structure', result['primitive_structure'])
        self.out('seekpath_parameters', result['parameters'])

    def prepare_wannier90_inputs(self):
        """Override parent method.

        :return: the inputs port
        :rtype: InputPort
        """
        base_inputs = super().prepare_wannier90_inputs()
        inputs = base_inputs['wannier90']

        parameters = inputs.parameters.get_dict()
        parameters['bands_plot'] = True
        inputs.parameters = orm.Dict(dict=parameters)
        if self.ctx.current_kpoint_path:
            inputs.kpoint_path = self.ctx.current_kpoint_path
        if self.ctx.current_bands_kpoints:
            inputs.bands_kpoints = self.ctx.current_bands_kpoints

        base_inputs['wannier90'] = inputs

        return base_inputs

    def results(self):
        """Attach the relevant output nodes from the band calculation to the workchain outputs for convenience."""
        super().results()

        if 'interpolated_bands' in self.outputs['wannier90']:
            w90_bands = self.outputs['wannier90']['interpolated_bands']
            self.out('band_structure', w90_bands)

    @classmethod
    def get_builder_from_protocol(  # pylint: disable=arguments-differ
        cls,
        codes: dict,
        structure: orm.StructureData,
        *,
        kpoint_path: orm.Dict = None,
        bands_kpoints: orm.KpointsData = None,
        kpoint_path_distance: float = None,
        run_opengrid: bool = False,
        opengrid_only_scf: bool = True,
        **kwargs
    ) -> ProcessBuilder:
        """Return a builder prepopulated with inputs selected according to the specified arguments.

        :param codes: a dictionary of ``Code`` instance for pw.x, pw2wannier90.x, wannier90.x, (optionally) projwfc.x.
        :type codes: dict
        :param structure: the ``StructureData`` instance to use.
        :type structure: orm.StructureData
        :param kpoint_path: Explicit kpoints to use for the Wannier bands interpolation.
        If provided, the workchain will directly generate input parameters for the structure
        and use the provided `kpoint_path`,
        e.g. when one wants to wannierise a conventional cell structure.
        If not provided, and if the structure is a primitive cell, will use seekpath to generate a kpath using
        `kpoint_path_distance` as the kpoints density. If the structure is not a primitive cell, will use seekpath
        to generate a primitive cell and a kpath, and generate input parameters for the PRIMITIVE cell.
        After submission of the workchian, a `seekpath_structure_analysis` calcfunction will be launched to
        store the provenance from non-primitive cell to primitive cell.
        In any case, the `get_builder_from_protocol` will
        NOT launch any calcfunction so the aiida database is kept unmodified.
        Defaults to None.
        :type kpoint_path: orm.KpointsData, optional
        :param kpoint_path_distance: Minimum kpoints distance for the Wannier bands interpolation.
        Specify either this or `kpoint_path`. If not provided, will use the default of seekpath.
        Defaults to None
        :type kpoint_path_distance: float, optional
        :param run_opengrid: if True use open_grid.x to accelerate calculations, defaults to False
        :type run_opengrid: bool, optional
        :param opengrid_only_scf: if True only one scf calculation will be performed in the OpengridWorkChain,
        defaults to True
        :type opengrid_only_scf: bool, optional
        :return: a process builder instance with all inputs defined and ready for launch.
        :rtype: ProcessBuilder
        """
        from aiida.tools import get_explicit_kpoints_path

        if sum(_ is not None for _ in (kpoint_path, bands_kpoints, kpoint_path_distance)) > 1:
            raise ValueError('Can only specify one of the `kpoint_path`, `bands_kpoints` and `kpoint_path_distance`')

        if run_opengrid and kwargs.get('electronic_type', None) == SpinType.SPIN_ORBIT:
            raise ValueError('open_grid.x does not support spin orbit coupling')

        # I will call different parent_class.get_builder_from_protocl()
        if run_opengrid:
            # i.e. Wannier90OpengridWorkChain
            parent_class = super(Wannier90BandsWorkChain, cls)
            kwargs['opengrid_only_scf'] = opengrid_only_scf
        else:
            # i.e. Wannier90WorkChain
            parent_class = super(Wannier90OpengridWorkChain, cls)

        summary = kwargs.pop('summary', {})
        print_summary = kwargs.pop('print_summary', True)

        if kpoint_path is None and bands_kpoints is None:
            # The following code will test if the input structure is a primitive structure,
            # if it is a primitive structure, a kpath will be generated so the workchain will not
            # run a seekpath analysis internally. However, this causes the workchain's behaviour
            # unpredictable since seekpath is not idempotent, i.e. even if the input structure is a
            # primitive cell generated from seekpath, a second seekpath run might still rotate the
            # cell, causing confusions. So now if no `kpoint_path` is provided, the workchain will
            # always run seekpath even if the structure is a primitive cell.
            #
            # In principle, one should first run a bunch of `seekpath_structure_analysis` and store
            # the primitive structure and the corresponding kpath, and always use both structure and
            # kpath for the WannierBandsWorkChain.
            #
            # Don't use `seekpath_structure_analysis`, since it's a calcfunction and will modify aiida database
            args = {'structure': structure}
            if kpoint_path_distance is not None:
                args['reference_distance'] = kpoint_path_distance
            result = get_explicit_kpoints_path(**args)
            primitive_structure = result['primitive_structure']
            # ase Atoms class can test if two structures are the same
            if structure.get_ase() == primitive_structure.get_ase():
                builder = parent_class.get_builder_from_protocol(
                    codes, structure, **kwargs, summary=summary, print_summary=False
                )
                # set kpoint_path, so the workchain won't run seekpath
                #builder.kpoint_path = orm.Dict(
                #    dict={
                #        'path': result['parameters']['path'],
                #        'point_coords': result['parameters']['point_coords']
                #    }
                #)
                builder.kpoint_path = result['explicit_kpoints']
            else:
                msg = f'The input structure {structure.get_formula()}<{structure.pk}> is NOT a primitive cell, '
                msg += 'the generated parameters are for the primitive cell '
                msg += f'{primitive_structure.get_formula()} found by seekpath.'
                notes = summary.get('notes', [])
                notes.append(msg)
                summary['notes'] = notes
                # I need to use primitive cell to generate all the input parameters, e.g. num_wann, num_bands, ...
                builder = parent_class.get_builder_from_protocol(
                    codes, primitive_structure, **kwargs, summary=summary, print_summary=False
                )
                # don't set `kpoint_path` and `kpoint_path_distance`, so the workchain will run seekpath
                # however I still need to use the original cell, so the `seekpath_structure_analysis` will
                # store the provenance from original cell to primitive cell.
                builder.structure = structure
        else:
            builder = parent_class.get_builder_from_protocol(
                codes, structure, **kwargs, summary=summary, print_summary=False
            )
            if kpoint_path:
                builder.kpoint_path = kpoint_path
            if bands_kpoints:
                builder.bands_kpoints = bands_kpoints

        if print_summary:
            cls.print_summary(summary)

        return builder


def get_builder_for_pwbands(wannier_workchain: Wannier90BandsWorkChain) -> ProcessBuilder:
    """Get a PwBaseWorkChain builder for calculating bands strcutre from a finished Wannier90BandsWorkChain.

    Useful for compareing QE and Wannier90 interpolated bands structures.

    :param wannier_workchain: [description]
    :type wannier_workchain: Wannier90BandsWorkChain
    """
    from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain

    if not wannier_workchain.is_finished_ok:
        msg = f'The {wannier_workchain.process_label}<{wannier_workchain.pk}> has not finished, '
        msg += f'current status: {wannier_workchain.process_state}, '
        msg += 'please retry after workchain has successfully finished.'
        print(msg)
        return

    scf_inputs = wannier_workchain.inputs['scf']
    scf_outputs = wannier_workchain.outputs['scf']
    builder = PwBaseWorkChain.get_builder()

    # wannier_workchain.outputs['scf']['pw'] has no `structure`, I will fill it in later
    excluded_inputs = ['pw']
    for k in scf_inputs:
        if k in excluded_inputs:
            continue
        builder[k] = scf_inputs[k]

    structure = wannier_workchain.inputs['structure']
    if 'primitive_structure' in wannier_workchain.outputs:
        structure = wannier_workchain.outputs['primitive_structure']

    pw_inputs = scf_inputs['pw']
    pw_inputs['structure'] = structure
    builder['pw'] = pw_inputs

    # should use wannier90 kpath, otherwise number of kpoints
    # of DFT and w90 are not consistent
    wannier_outputs = wannier_workchain.outputs['wannier90']
    wannier_bands = wannier_outputs['interpolated_bands']

    wannier_kpoints = orm.KpointsData()
    wannier_kpoints.set_kpoints(wannier_bands.get_kpoints())
    wannier_kpoints.set_attribute_many({
        'cell': wannier_bands.attributes['cell'],
        'pbc1': wannier_bands.attributes['pbc1'],
        'pbc2': wannier_bands.attributes['pbc2'],
        'pbc3': wannier_bands.attributes['pbc3'],
        'labels': wannier_bands.attributes['labels'],
        # 'array|kpoints': ,
        'label_numbers': wannier_bands.attributes['label_numbers']
    })
    builder.kpoints = wannier_kpoints

    builder['pw']['parent_folder'] = scf_outputs['remote_folder']

    parameters = builder['pw']['parameters'].get_dict()
    parameters.setdefault('CONTROL', {})
    parameters.setdefault('SYSTEM', {})
    parameters.setdefault('ELECTRONS', {})
    parameters['CONTROL']['calculation'] = 'bands'
    # parameters['ELECTRONS'].setdefault('diagonalization', 'cg')
    parameters['ELECTRONS'].setdefault('diago_full_acc', True)

    if 'nscf' in wannier_workchain.inputs:
        nscf_inputs = wannier_workchain.inputs['nscf']
        nbnd = nscf_inputs['pw']['parameters']['SYSTEM']['nbnd']
        parameters['SYSTEM']['nbnd'] = nbnd

    builder['pw']['parameters'] = orm.Dict(dict=parameters)

    return builder
