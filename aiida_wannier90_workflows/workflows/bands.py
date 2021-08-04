import typing

from aiida import orm
from aiida.engine import if_, ProcessBuilder
from aiida.engine.launch import run

from aiida_quantumespresso.calculations.functions.seekpath_structure_analysis import seekpath_structure_analysis
from aiida_quantumespresso.common.types import SpinType

from .wannier import Wannier90WorkChain
from .opengrid import Wannier90OpengridWorkChain

__all__ = ['Wannier90BandsWorkChain', 'get_builder_for_pwbands']


def validate_inputs(inputs, ctx=None):  # pylint: disable=unused-argument
    """Validate the inputs of the entire input namespace."""
    # pylint: disable=no-member

    # Cannot specify both `kpoint_path` and `kpoint_path_distance`
    if all([key in inputs for key in ['kpoint_path', 'kpoint_path_distance']]):
        return Wannier90BandsWorkChain.exit_codes.ERROR_INVALID_INPUT_KPOINTS.message


class Wannier90BandsWorkChain(Wannier90OpengridWorkChain):
    """
    A high level workchain which can automatically compute a Wannier band structure for a given structure. Can also output Wannier Hamiltonian.
    """
    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)
        spec.input(
            'kpoint_path',
            valid_type=orm.KpointsData,
            required=False,
            help=
            'Explicit kpoints to use for the BANDS calculation. If not specified, the workchain will run seekpath to generate a primitive cell and a kpoint_path. Specify either this or `kpoint_path_distance`.'
        )
        spec.input(
            'kpoint_path_distance',
            valid_type=orm.Float,
            required=False,
            help=
            'Minimum kpoints distance for the BANDS calculation. Specify either this or `kpoint_path`.'
        )

        # We expose the in/output of Wannier90OpengridWorkChain since Wannier90WorkChain in/output
        # is a subset of Wannier90OpengridWorkChain,
        # this allow us to launch either Wannier90WorkChain or Wannier90OpengridWorkChain.
        spec.expose_inputs(
            Wannier90OpengridWorkChain,
            exclude=('wannier90.kpoint_path', ),
            namespace_options={'required': True}
        )
        spec.inputs.validator = validate_inputs

        spec.outline(
            cls.setup,
            cls.validate_parameters,
            if_(cls.should_run_seekpath)(
                cls.run_seekpath,
            ),
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
            help=
            'The normalized and primitivized structure for which the calculations are computed.'
        )
        spec.output(
            'seekpath_parameters',
            valid_type=orm.Dict,
            required=False,
            help=
            'The parameters used in the SeeKpath call to normalize the input or relaxed structure.'
        )
        spec.expose_outputs(
            Wannier90OpengridWorkChain,
            namespace_options={'required': True}
        )
        spec.output(
            'band_structure',
            valid_type=orm.BandsData,
            help='The Wannier interpolated band structure.'
        )

        spec.exit_code(
            401,
            'ERROR_INVALID_INPUT_UNRECOGNIZED_KIND',
            message='Input `StructureData` contains an unsupported kind.'
        )
        spec.exit_code(
            402,
            'ERROR_INVALID_INPUT_KPOINTS',
            message=
            'Cannot specify both `kpoint_path` and `kpoint_path_distance`.'
        )
        spec.exit_code(
            403,
            'ERROR_INVALID_INPUT_OPENGRID',
            message='No open_grid.x Code provided.'
        )
        spec.exit_code(
            404,
            'ERROR_INVALID_INPUT_PSEUDOPOTENTIAL',
            message='Invalid pseudopotentials.'
        )

    def setup(self):
        """Define the current structure in the context to be the input structure."""
        super().setup()

        if not self.should_run_seekpath():
            self.ctx.current_kpoint_path = self.inputs.kpoint_path

    def should_run_seekpath(self):
        """Seekpath should only be run if the `kpoint_path` input is not specified."""
        return 'kpoint_path' not in self.inputs

    def run_seekpath(self):
        """Run the structure through SeeKpath to get the primitive and normalized structure."""
        structure_formula = self.inputs.structure.get_formula()
        self.report(f'launching seekpath for: {structure_formula}')

        args = {
            'structure': self.inputs.structure,
            'metadata': {
                'call_link_label': 'seekpath_structure_analysis'
            }
        }
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
        """overrides parent method

        :return: the inputs port
        :rtype: InputPort
        """
        inputs = super().prepare_wannier90_inputs()

        parameters = inputs.parameters.get_dict()
        parameters['bands_plot'] = True
        inputs.parameters = orm.Dict(dict=parameters)
        inputs.kpoint_path = self.ctx.current_kpoint_path

        return inputs

    def results(self):
        """Attach the relevant output nodes from the band calculation to the workchain outputs for convenience."""
        super().results()

        if 'interpolated_bands' in self.outputs['wannier90']:
            w90_bands = self.outputs['wannier90']['interpolated_bands']
            self.out('band_structure', w90_bands)

    @classmethod
    def get_builder_from_protocol(
        cls,
        codes: dict,
        structure: orm.StructureData,
        *,
        kpoint_path: orm.Dict = None,
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
        If provided, the workchain will directly generate input parameters for the structure and use the provided `kpoint_path`,
        e.g. when one wants to wannierise a conventional cell structure.
        If not provided, and if the structure is a primitive cell, will use seekpath to generate a kpath using 
        `kpoint_path_distance` as the kpoints density. If the structure is not a primitive cell, will use seekpath
        to generate a primitive cell and a kpath, and generate input parameters for the PRIMITIVE cell. 
        After submission of the workchian, a `seekpath_structure_analysis` calcfunction will be launched to 
        store the provenance from non-primitive cell to primitive cell. In any case, the `get_builder_from_protocol` will
        NOT launch any calcfunction so the aiida database is kept unmodified.
        Defaults to None.
        :type kpoint_path: orm.KpointsData, optional
        :param kpoint_path_distance: Minimum kpoints distance for the Wannier bands interpolation. Specify either this or `kpoint_path`. If not provided, will use the default of seekpath. Defaults to None
        :type kpoint_path_distance: float, optional
        :param run_opengrid: if True use open_grid.x to accelerate calculations, defaults to False
        :type run_opengrid: bool, optional
        :param opengrid_only_scf: if True only one scf calculation will be performed in the OpengridWorkChain, defaults to True
        :type opengrid_only_scf: bool, optional
        :return: a process builder instance with all inputs defined and ready for launch.
        :rtype: ProcessBuilder
        """
        from aiida.tools import get_explicit_kpoints_path

        if kpoint_path is not None and kpoint_path_distance is not None:
            raise ValueError(
                "Cannot specify both `kpoint_path` and `kpoint_path_distance`"
            )

        if run_opengrid and kwargs.get(
            'electronic_type', None
        ) == SpinType.SPIN_ORBIT:
            raise ValueError(
                'open_grid.x does not support spin orbit coupling'
            )
        
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

        if kpoint_path is None:
            # don't use `seekpath_structure_analysis`, since it's a calcfunction and will modify aiida database
            args = {'structure': structure}
            if kpoint_path_distance is not None:
                args['reference_distance'] = kpoint_path_distance
            result = get_explicit_kpoints_path(**args)
            primitive_structure = result['primitive_structure']
            # ase Atoms class can test if two structures are the same
            if structure.get_ase() == primitive_structure.get_ase():
                builder = parent_class.get_builder_from_protocol(
                    codes, structure, **kwargs,
                    summary=summary, print_summary=False
                )
                # set kpoint_path, so the workchain won't run seekpath
                builder.kpoint_path = orm.Dict(
                    dict={
                        'path': result['path'],
                        'point_coords': result['point_coords']
                    }
                )
            else:
                msg = f'The input structure {structure.get_formula()}<{structure.pk}> is NOT a primitive cell, '
                msg += f'the generated parameters are for the primitive cell {primitive_structure.get_formula()} found by seekpath.'
                notes = summary.get('notes', [])
                notes.append(msg)
                summary['notes'] = notes
                # I need to use primitive cell to generate all the input parameters, e.g. num_wann, num_bands, ...
                builder = parent_class.get_builder_from_protocol(
                    codes, primitive_structure, **kwargs,
                    summary=summary, print_summary=False
                )
                # don't set `kpoint_path` and `kpoint_path_distance`, so the workchain will run seekpath
                # however I still need to use the original cell, so the `seekpath_structure_analysis` will
                # store the provenance from original cell to primitive cell.
                builder.structure = structure
        else:
            builder = parent_class.get_builder_from_protocol(
                codes, structure, **kwargs,
                summary=summary, print_summary=False
            )
            builder.kpoint_path = kpoint_path

        if print_summary:
            cls.print_summary(summary)

        return builder


def get_builder_for_pwbands(
    wannier_workchain: Wannier90BandsWorkChain
) -> ProcessBuilder:
    """Get a PwBaseWorkChain builder for calculating bands strcutre 
    from a finished Wannier90BandsWorkChain.
    Useful for compareing QE and Wannier90 interpolated bands structures.

    :param wannier_workchain: [description]
    :type wannier_workchain: Wannier90BandsWorkChain
    """
    from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain

    if not wannier_workchain.is_finished_ok:
        msg = f'The {wannier_workchain.process_label}<{wannier_workchain.pk}> has not finished, '
        msg += f'current status: {wannier_workchain.process_state}, '
        msg += f'please retry after workchain has successfully finished.'
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
        'cell':
        wannier_bands.attributes['cell'],
        'pbc1':
        wannier_bands.attributes['pbc1'],
        'pbc2':
        wannier_bands.attributes['pbc2'],
        'pbc3':
        wannier_bands.attributes['pbc3'],
        'labels':
        wannier_bands.attributes['labels'],
        # 'array|kpoints': ,
        'label_numbers':
        wannier_bands.attributes['label_numbers']
    })
    builder.kpoints = wannier_kpoints

    builder.pw['parent_folder'] = scf_outputs['remote_folder']

    parameters = builder['pw']['parameters'].get_dict()
    parameters.setdefault('CONTROL', {})
    parameters.setdefault('SYSTEM', {})
    parameters.setdefault('ELECTRONS', {})
    parameters['CONTROL']['calculation'] = 'bands'
    parameters['ELECTRONS'].setdefault('diagonalization', 'cg')
    parameters['ELECTRONS'].setdefault('diago_full_acc', True)

    if 'nscf' in wannier_workchain.inputs:
        nscf_inputs = wannier_workchain.inputs['nscf']
        nbnd = nscf_inputs['pw']['parameters']['SYSTEM']['nbnd']
        parameters['SYSTEM']['nbnd'] = nbnd

    builder.pw['parameters'] = orm.Dict(dict=parameters)

    return builder


def get_default_options(structure, with_mpi=False):
    """Increase wallclock to 5 hour, use mpi, set number of machines according to 
    number of atoms.
    
    :param with_mpi: [description], defaults to False
    :type with_mpi: bool, optional
    :return: [description]
    :rtype: [type]
    """
    def estimate_num_machines(structure):
        """
        1 <= num_atoms <= 10 -> 1 machine
        11 <= num_atoms <= 20 -> 2 machine
        ...
        
        :param structure: crystal structure
        :type structure: aiida.orm.StructureData
        :return: estimated number of machines based on number of atoms
        :rtype: int
        """
        from math import ceil
        num_atoms = len(structure.sites)
        return ceil(num_atoms / 10)

    from aiida_quantumespresso.utils.resources import get_default_options as get_opt
    num_machines = estimate_num_machines(structure)
    opt = get_opt(
        max_num_machines=num_machines,
        max_wallclock_seconds=3600 * 5,  # 5 hours
        with_mpi=with_mpi
    )
    return opt


def get_manual_options():
    # QE projwfc.x complains
    #         Calling projwave ....
    #     linear algebra parallelized on  16 procs

    # Problem Sizes
    # natomwfc =           14
    # nx       =            4
    # nbnd     =           24
    # nkstot   =          216
    # npwx     =           83
    # nkb      =           20

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #     Error in routine  blk2cyc_zredist (1):

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #     Error in routine  blk2cyc_zredist (1):
    #     nb less than the number of proc

    # https://aiida-core.readthedocs.io/en/v1.1.1/scheduler/index.html#nodenumberjobresource
    return {
        'resources': {
            'num_machines': 1,
            'num_mpiprocs_per_machine': 8,
            # 'tot_num_mpiprocs': ,
            # 'num_cores_per_machine': ,
            # 'num_cores_per_mpiproc': ,
        },
        'max_wallclock_seconds': 3600 * 5,
        'withmpi': True,
    }
