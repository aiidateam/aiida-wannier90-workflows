# -*- coding: utf-8 -*-
"""Workchain to automatically optimize dis_proj_min/max for projectability disentanglement."""
from copy import deepcopy
import numpy as np
from aiida import orm
from aiida.orm.nodes.data.base import to_aiida_type
from aiida.common import AttributeDict
from aiida.engine import while_, if_, ToContext, append_
from aiida_quantumespresso.utils.mapping import prepare_process_inputs
from aiida_wannier90_workflows.utils.node import get_last_calcjob
from aiida_wannier90_workflows.utils.bandsdist import bands_distance
from aiida_wannier90_workflows.workflows.base.wannier import Wannier90BaseWorkChain
from aiida_wannier90_workflows.workflows.bands import Wannier90BandsWorkChain


class Wannier90OptimizeWorkChain(Wannier90BandsWorkChain):
    """Workchain to optimize dis_proj_min/max for projectability disentanglement."""

    # The following keys are for wannier90.x plotting, i.e. they can be restarted from
    # chk file by setting `restart = plot` in wannier90.win.
    _WANNIER90_PLOT_INPUTS = (
        'wannier_plot', 'bands_plot', 'write_tb', 'write_hr', 'write_hhmn', 'write_hkmn', 'write_hvmn', 'write_hdmn'
    )

    @classmethod
    def define(cls, spec):
        """Define the process spec."""
        super().define(spec)

        spec.input(
            'separate_plotting',
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            serializer=to_aiida_type,
            help=(
                'If True separate the maximal localisation and the plotting of bands/Wannier function in two steps. '
                'This allows reusing the chk file to restart plotting if it were crashed due to memory issue.'
            )
        )
        spec.input(
            'optimize_disproj',
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            serializer=to_aiida_type,
            help=('If True iterate dis_proj_min/max to find the best MLWFs for projectability disentanglement.')
        )
        spec.input(
            'optimize_disprojmax_range',
            valid_type=orm.List,
            default=lambda: orm.List(list=list(np.linspace(0.99, 0.85, 15))),
            serializer=to_aiida_type,
            help=('The range to iterate dis_proj_min. `None` means disabling projectability disentanglement.')
        )
        spec.input(
            'optimize_disprojmin_range',
            valid_type=orm.List,
            default=lambda: orm.List(list=list(np.linspace(0.01, 0.02, 2))),
            serializer=to_aiida_type,
            help=('The range to iterate dis_proj_max. `None` means disabling projectability disentanglement.')
        )
        spec.input(
            'optimize_reference_bands',
            valid_type=orm.BandsData,
            required=False,
            help=(
                'If provided, during the iteration of dis_proj_min/max, the BandsData will be the reference '
                'for calculating bands distance, the final optimal MLWFs will be selected based on both spreads '
                'and bands distance. If not provided, spreads will be the criterion for selecting optimal MLWFs. '
                'The bands distance is calculated for bands below Fermi energy + 2eV.'
            )
        )
        spec.input(
            'optimize_bands_distance_threshold',
            valid_type=orm.Float,
            required=False,
            serializer=to_aiida_type,
            help=(
                'If provided, during the iteration of dis_proj_min/max, if the bands distance is smaller '
                'than this threshold, the optimization will stop. Unit is eV.'
            )
        )
        spec.input(
            'optimize_spreads_imbalence_threshold',
            valid_type=orm.Float,
            required=False,
            serializer=to_aiida_type,
            help=(
                'If provided, during the iteration of dis_proj_min/max, if the spreads imbalence is smaller '
                'than this threshold, the optimization will stop.'
            )
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
            Wannier90BaseWorkChain, namespace='wannier90_optimal', namespace_options={'required': False}
        )
        spec.expose_outputs(Wannier90BaseWorkChain, namespace='wannier90_plot', namespace_options={'required': False})

        spec.output(
            'bands_distance',
            valid_type=orm.List,
            required=False,
            help='Bands distances between reference bands and Wannier interpolated bands for Ef to Ef+5eV.'
        )

        spec.exit_code(
            490,
            'ERROR_SUB_PROCESS_FAILED_WANNIER90_OPTIMIZE',
            message='All the trials on dis_proj_min/max have failed, cannot compare bands distance'
        )
        spec.exit_code(
            491,
            'ERROR_SUB_PROCESS_FAILED_WANNIER90_OPTIMIZE',
            message='All the trials on dis_proj_min/max have failed, cannot compare spreads'
        )
        spec.exit_code(
            490,
            'ERROR_SUB_PROCESS_FAILED_WANNIER90_PLOT',
            message='the Wannier90Calculation plotting sub process failed'
        )

    @staticmethod
    def validate_inputs(inputs, ctx=None):  # pylint: disable=unused-argument
        """Validate the inputs of the entire input namespace."""
        # Call parent validator
        result = Wannier90BandsWorkChain.validate_inputs(inputs)
        if result is not None:
            return result

        parameters = inputs['wannier90']['wannier90']['parameters'].get_dict()

        if inputs['optimize_disproj']:
            if all(_ not in parameters for _ in ('dis_proj_min', 'dis_proj_max')):
                return 'Trying to optimize dis_proj_min/max but no dis_proj_min/max in wannier90 parameters?'

        if 'optimize_reference_bands' in inputs and not inputs['optimize_disproj']:
            return '`optimize_reference_bands` is provided but `optimize_disproj = False`?'

        if 'optimize_bands_distance_threshold' in inputs and 'optimize_reference_bands' not in inputs:
            return 'No `optimize_reference_bands` but `optimize_bands_distance_threshold` is set?'

        if inputs['separate_plotting']:
            plot_inputs = [parameters.get(_, False) for _ in Wannier90OptimizeWorkChain._WANNIER90_PLOT_INPUTS]
            if not any(plot_inputs):
                return (
                    'Trying to separate plotting routines but no '
                    f"{'/'.join(Wannier90OptimizeWorkChain._WANNIER90_PLOT_INPUTS)} in wannier90 parameters?"
                )

        if inputs['optimize_disproj'] and not inputs['separate_plotting']:
            return (
                '`optimize_disproj = True` but `separate_plotting = False`. For optimizing projectability '
                'disentanglement, it is highly recommended to run the plotting mode in a separate step.'
            )

    def setup(self):
        """Define the current structure in the context to be the input structure."""
        super().setup()

        dis_proj_min = self.inputs['optimize_disprojmin_range'].get_list()
        dis_proj_max = self.inputs['optimize_disprojmax_range'].get_list()
        # dis_proj_max changes the fastest
        self.ctx.optimize_minmax_new = [(i, j) for i in dis_proj_min for j in dis_proj_max]

        # Arrays to save calculated results
        self.ctx.optimize_minmax = []
        self.ctx.optimize_bandsdist = []
        self.ctx.optimize_spreads_imbalence = []
        # The index of the optimal wannier90 workchain
        self.ctx.optimize_best = None

        # For separate_plotting, restore these inputs when running plotting calc.
        self.ctx.saved_parameters = {}
        if self.inputs['separate_plotting']:
            parameters = self.inputs.wannier90.wannier90['parameters'].get_dict()
            # I convert the tuple to list so it can be changed
            excluded_inputs = list(Wannier90OptimizeWorkChain._WANNIER90_PLOT_INPUTS)
            # I need to calculate bands for comparing bands distance
            if 'optimize_reference_bands' in self.inputs:
                excluded_inputs.remove('bands_plot')
            for key in excluded_inputs:
                plot_input = parameters.get(key, False)
                if plot_input:
                    self.ctx.saved_parameters[key] = plot_input

    def should_run_wannier90_optimize(self):
        """Whether should optimize dis_proj_min/max."""
        if not self.inputs['optimize_disproj']:
            return False

        if 'optimize_bands_distance_threshold' in self.inputs:
            threshold = self.inputs['optimize_bands_distance_threshold']
            if self.ctx.workchain_wannier90_bandsdist <= threshold:
                # Stop if the initial bands distance is already good enough
                self.ctx.optimize_minmax_new = []
            elif len(self.ctx.optimize_bandsdist) > 0 and np.min(self.ctx.optimize_bandsdist) <= threshold:
                self.ctx.optimize_minmax_new = []
        elif 'optimize_spreads_imbalence_threshold' in self.inputs:
            threshold = self.inputs['optimize_spreads_imbalence_threshold']
            if self.ctx.workchain_wannier90_spreads_imbalence <= threshold:
                # Stop if the initial spreads are already good enough
                self.ctx.optimize_minmax_new = []
            elif len(self.ctx.optimize_spreads_imbalence
                     ) > 0 and np.min(self.ctx.optimize_spreads_imbalence) <= threshold:
                self.ctx.optimize_minmax_new = []

        if len(self.ctx.optimize_minmax_new) == 0:
            return False

        return True

    def has_run_wannier90_optimize(self):
        """Whether the optimization loop has been invoked."""
        return 'workchain_wannier90_optimize' in self.ctx

    def should_run_wannier90_plot(self):
        """Whether to run wannier90 maximal localisation and plotting in two steps or in one step."""
        return self.inputs['separate_plotting']

    def prepare_wannier90_inputs(self):
        """Override parent method.

        :return: the inputs port
        :rtype: InputPort
        """
        base_inputs = super().prepare_wannier90_inputs()
        inputs = base_inputs['wannier90']

        parameters = inputs.parameters.get_dict()

        # Do not run plotting subroutines in the Wannier90BandsWorkChain, they will be run in a separate step
        if self.should_run_wannier90_plot():
            for key in self.ctx.saved_parameters:
                parameters.pop(key, None)
            inputs.parameters = orm.Dict(dict=parameters)

            if 'optimize_reference_bands' not in self.inputs:
                inputs.pop('kpoint_path', None)
                inputs.pop('bands_kpoints', None)

            base_inputs['wannier90'] = inputs

        return base_inputs

    def run_wannier90(self):
        """Overide parent, pop stash settings."""
        base_inputs = AttributeDict(self.exposed_inputs(Wannier90BaseWorkChain, namespace='wannier90'))
        inputs = base_inputs['wannier90']

        # Use the Wannier90BaseWorkChain-corrected parameters
        last_calc = get_last_calcjob(self.ctx.workchain_wannier90_pp)
        # copy postproc inputs, especially the `kmesh_tol` might have been corrected
        for key in last_calc.inputs:
            inputs[key] = last_calc.inputs[key]

        inputs['remote_input_folder'] = self.ctx.current_folder

        if 'settings' in inputs:
            settings = inputs.settings.get_dict()
        else:
            settings = {}
        settings['postproc_setup'] = False

        inputs.settings = settings

        # I should not stash files if there is an additional plotting step,
        # otherwise there is a RemoteStashFolderData in outputs
        if self.should_run_wannier90_plot():
            inputs['metadata']['options'].pop('stash', None)

        base_inputs['wannier90'] = inputs
        base_inputs['metadata'] = {'call_link_label': 'wannier90'}
        base_inputs['clean_workdir'] = orm.Bool(False)
        inputs = prepare_process_inputs(Wannier90BaseWorkChain, base_inputs)

        running = self.submit(Wannier90BaseWorkChain, **inputs)
        self.report(f'launching {running.process_label}<{running.pk}>')

        return ToContext(workchain_wannier90=running)

    def inspect_wannier90(self):
        """Overide parent."""
        super().inspect_wannier90()

        workchain = self.ctx.workchain_wannier90
        if 'optimize_reference_bands' in self.inputs:
            bandsdist = self._get_bands_distance(workchain)
            self.ctx.workchain_wannier90_bandsdist = bandsdist
            self.report(f'current workchain<{workchain.pk}> bands distance={bandsdist:.2e}eV')

        self.ctx.workchain_wannier90_spreads_imbalence = get_spreads_imbalence(
            workchain.outputs.output_parameters['wannier_functions_output']
        )

    def run_wannier90_optimize(self):
        """Optimize dis_proj_min/max."""

        base_inputs = AttributeDict(self.exposed_inputs(Wannier90BaseWorkChain, namespace='wannier90'))
        inputs = base_inputs['wannier90']

        # I need to save `inputs.wannier90.metadata.options.resources`, somehow it is missing if I
        # copy all the inputs from `self.ctx.workchain_wannier90`.
        # resources = base_inputs['wannier90']['wannier90']['metadata']['options']['resources']

        # Use the Wannier90BaseWorkChain-corrected parameters, especially `num_mpiprocs_per_machine`
        last_calc = get_last_calcjob(self.ctx.workchain_wannier90)
        for key in last_calc.inputs:
            inputs[key] = last_calc.inputs[key]

        parameters = inputs.parameters.get_dict()

        dis_proj_min, dis_proj_max = self.ctx.optimize_minmax_new[0]
        parameters['dis_proj_min'] = dis_proj_min
        parameters['dis_proj_max'] = dis_proj_max

        if 'optimize_reference_bands' in self.inputs:
            parameters['bands_plot'] = True
            if self.ctx.current_kpoint_path:
                inputs.kpoint_path = self.ctx.current_kpoint_path
            if self.ctx.current_bands_kpoints:
                inputs.bands_kpoints = self.ctx.current_bands_kpoints

        inputs.parameters = orm.Dict(dict=parameters)

        # I should not stash files if there is an additional plotting step,
        # otherwise there is a RemoteStashFolderData in outputs
        if self.should_run_wannier90_plot():
            inputs['metadata']['options'].pop('stash', None)

        base_inputs['wannier90'] = inputs
        iteration = len(self.ctx.optimize_minmax) + 1  # Start from 1
        base_inputs['metadata'] = {'call_link_label': f'wannier90_optimize_iteration{iteration}'}
        base_inputs['clean_workdir'] = orm.Bool(False)

        # Disable the error handler which might modify dis_proj_min
        handler_overrides = {'handle_disentanglement_not_enough_states': False}
        base_inputs['handler_overrides'] = orm.Dict(dict=handler_overrides)

        inputs = prepare_process_inputs(Wannier90BaseWorkChain, base_inputs)

        running = self.submit(Wannier90BaseWorkChain, **inputs)
        self.report(
            f'launching {running.process_label}<{running.pk}> with dis_proj_min={dis_proj_min} '
            f'dis_proj_max={dis_proj_max}'
        )

        return ToContext(workchain_wannier90_optimize=append_(running))

    def inspect_wannier90_optimize(self):
        """Verify that the `Wannier90BaseWorkChain` for the wannier90 optimization run successfully finished."""
        workchain = self.ctx.workchain_wannier90_optimize[-1]

        if workchain.is_finished_ok:
            spreads = get_spreads_imbalence(workchain.outputs.output_parameters['wannier_functions_output'])
            if 'optimize_reference_bands' in self.inputs and 'interpolated_bands' in workchain.outputs:
                bandsdist = self._get_bands_distance(workchain)
                self.report(f'current workchain<{workchain.pk}> bands distance={bandsdist:.2e}eV')
            else:
                bandsdist = None
        else:
            self.report(
                f'{workchain.process_label} failed with exit status {workchain.exit_status}, '
                'but I will keep launching next iteration'
            )
            spreads = None
            bandsdist = None

        minmax = self.ctx.optimize_minmax_new.pop(0)
        self.ctx.optimize_minmax.append(minmax)
        self.ctx.optimize_bandsdist.append(bandsdist)
        self.ctx.optimize_spreads_imbalence.append(spreads)

    def inspect_wannier90_optimize_final(self):
        """Select the optimal choice for dis_proj_min/max."""
        if not self.has_run_wannier90_optimize():
            return

        workchains = self.ctx.workchain_wannier90_optimize

        # The index of the optimal wannier90 workchain
        self.ctx.optimize_best = None

        if 'optimize_reference_bands' in self.inputs:
            # Usually good bands distance means MLWFs have good spreads
            bandsdist = np.array([_ if _ else 1e5 for _ in self.ctx.optimize_bandsdist])
            idx = np.argmin(bandsdist)
            self.ctx.optimize_best = idx
            minmax = self.ctx.optimize_minmax[idx]
            self.report(
                f'Optimal bands distance={bandsdist[idx]:.2e}, '
                f'dis_proj_min={minmax[0]} dis_proj_max={minmax[1]}'
            )
        else:
            # I only check the spreads are balenced
            spreads = np.array([_ if _ else 1e5 for _ in self.ctx.optimize_spreads_imbalence])
            idx = np.argmin(spreads)
            self.ctx.optimize_best = idx
            minmax = self.ctx.optimize_minmax[idx]
            self.report(f'Optimal spreads={spreads[idx]}, ' f'dis_proj_min={minmax[0]} dis_proj_max={minmax[1]}')

        self.ctx.current_folder = workchains[self.ctx.optimize_best].outputs.remote_folder

    def run_wannier90_plot(self):
        """Wannier90 plot step, also stash files."""
        base_inputs = AttributeDict(self.exposed_inputs(Wannier90BaseWorkChain, namespace='wannier90'))
        inputs = base_inputs['wannier90']

        # I should stash files, which was removed from metadata in the postproc step
        stash = None
        if 'stash' in inputs['metadata']['options']:
            # I deepcopy it to avoid it being overwritten
            stash = deepcopy(inputs['metadata']['options']['stash'])

        # Use the corrected parameters
        if self.has_run_wannier90_optimize():
            # Use the optimal parameters
            optimal_workchain = self.ctx.workchain_wannier90_optimize[self.ctx.optimize_best]
        else:
            # Just use the base workchain
            optimal_workchain = self.ctx.workchain_wannier90
        # Copy inputs, especially the `dis_proj_min/max` might have been corrected
        last_calc = get_last_calcjob(optimal_workchain)
        for key in last_calc.inputs:
            inputs[key] = last_calc.inputs[key]

        inputs['remote_input_folder'] = self.ctx.current_folder

        # Restore stash files
        if stash:
            inputs['metadata']['options']['stash'] = stash

        # Restore plotting related tags
        parameters = inputs.parameters.get_dict()
        for key in self.ctx.saved_parameters:
            parameters[key] = True

        parameters['restart'] = 'plot'
        inputs.parameters = orm.Dict(dict=parameters)

        if parameters.get('bands_plot', False):
            if self.ctx.current_kpoint_path:
                inputs.kpoint_path = self.ctx.current_kpoint_path
            if self.ctx.current_bands_kpoints:
                inputs.bands_kpoints = self.ctx.current_bands_kpoints

        base_inputs['wannier90'] = inputs
        base_inputs['metadata'] = {'call_link_label': 'wannier90_plot'}
        base_inputs['clean_workdir'] = orm.Bool(False)
        inputs = prepare_process_inputs(Wannier90BaseWorkChain, base_inputs)

        running = self.submit(Wannier90BaseWorkChain, **inputs)
        self.report(f'launching {running.process_label}<{running.pk}> in plotting mode')

        return ToContext(workchain_wannier90_plot=running)

    def inspect_wannier90_plot(self):
        """Verify that the `Wannier90BaseWorkChain` for the wannier90 plotting run successfully finished."""
        workchain = self.ctx.workchain_wannier90_plot

        if not workchain.is_finished_ok:
            self.report(f'{workchain.process_label} failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_WANNIER90_PLOT

        self.ctx.current_folder = workchain.outputs.remote_folder

    def results(self):
        """Attach the relevant output nodes from the band calculation to the workchain outputs for convenience."""
        super().results()

        if self.inputs['optimize_disproj']:
            if self.has_run_wannier90_optimize():
                optimal_workchain = self.ctx.workchain_wannier90_optimize[self.ctx.optimize_best]
            else:
                optimal_workchain = self.ctx.workchain_wannier90
            self.out_many(
                self.exposed_outputs(optimal_workchain, Wannier90BaseWorkChain, namespace='wannier90_optimal')
            )

        if self.should_run_wannier90_plot():
            self.out_many(
                self.exposed_outputs(
                    self.ctx.workchain_wannier90_plot, Wannier90BaseWorkChain, namespace='wannier90_plot'
                )
            )
            if 'interpolated_bands' in self.outputs['wannier90_plot']:
                w90_bands = self.outputs['wannier90_plot']['interpolated_bands']
                self.out('band_structure', w90_bands)

        if 'optimize_reference_bands' in self.inputs:
            if self.has_run_wannier90_optimize():
                optimal_workchain = self.ctx.workchain_wannier90_optimize[self.ctx.optimize_best]
            else:
                # Even if I haven't run optimization, I still output bands distance if reference bands is present
                optimal_workchain = self.ctx.workchain_wannier90
            bandsdist = self._get_bands_distance_raw(optimal_workchain).tolist()
            bandsdist = orm.List(list=bandsdist)
            bandsdist.store()
            self.out('bands_distance', bandsdist)

    def _get_bands_distance_raw(self, wannier_workchain) -> np.array:
        """Get bands distance for Fermi energy to Fermi energy + 5eV."""
        pw_bands = self.inputs['optimize_reference_bands']
        wan_bands = wannier_workchain.outputs['interpolated_bands']

        wan_parameters = wannier_workchain.inputs['wannier90']['parameters'].get_dict()
        fermi_energy = wan_parameters.get('fermi_energy')
        exclude_list_dft = wan_parameters.get('exclude_bands', None)

        # bands distance from Ef to Ef+5
        bandsdist = bands_distance(pw_bands, wan_bands, fermi_energy, exclude_list_dft)

        # Only return average distance, not max distance
        bandsdist = bandsdist[:, 1]

        return bandsdist

    def _get_bands_distance(self, wannier_workchain) -> float:
        """Get bands distance for Fermi energy + 2eV."""
        bandsdist = self._get_bands_distance_raw(wannier_workchain)
        bandsdist = bandsdist[2]  # TODO check 2 <-> Ef+2?  pylint: disable=fixme

        return bandsdist


def get_spreads_imbalence(wannier_functions_output: dict) -> float:
    """Calculate the variance of spreads.

    There could be other ways to calculate the spreads imbalence, for now I just use variance.
    :param wannier_functions_output: [description]
    :type wannier_functions_output: dict
    :return: [description]
    :rtype: float
    """
    spreads = [_['wf_spreads'] for _ in wannier_functions_output]
    var = np.var(spreads)

    # TODO try K-Means clustering?  pylint: disable=fixme
    return var
