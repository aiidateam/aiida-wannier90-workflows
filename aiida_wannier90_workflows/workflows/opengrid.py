from aiida import orm
from aiida.common import AttributeDict
from aiida.engine.processes import calcfunction
from aiida.engine.processes import WorkChain, ToContext, if_
from aiida_quantumespresso.utils.mapping import prepare_process_inputs
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from .wannier import Wannier90WorkChain, update_nscf_num_bands, get_num_projections_from_pseudos
from aiida_quantumespresso.calculations.opengrid import OpengridCalculation

class Wannier90OpengridWorkChain(Wannier90WorkChain):
    """This WorkChain uses open_grid.x to unfold the 
    symmetrized kmesh to a full kmesh in the Wannier90WorkChain.
    The full-kmesh nscf can be avoided.

    2 schemes:
    1. scf w/ symmetry, more nbnd -> open_grid 
       -> pw2wannier90 -> wannier90
    2. scf w/ symmetry, default nbnd -> nscf w/ symm, more nbnd 
       -> open_grid -> pw2wannier90 -> wannier90

    :param Wannier90WorkChain: [description]
    :type Wannier90WorkChain: [type]
    """
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input(
            'opengrid.run_nscf',
            valid_type=orm.Bool,
            default=lambda: orm.Bool(True),
            help='If True first do a scf with symmetry and default number of bands, then a nscf with symmetry and increased number of bands, followed by open_grid; If False first do a scf with symmetry and increased number of bands, then open_grid to unfold kmesh.'
        )
        spec.input(
            'opengrid.code',
            valid_type=orm.Code,
            required=True,
            help='Code to run open_grid.x'
        )
        spec.outline(
            cls.setup,
            if_(cls.should_do_relax)(
                cls.run_relax,
                cls.inspect_relax,
            ),
            if_(cls.should_do_nscf)(
                cls.run_scf,
                cls.inspect_scf,
                cls.run_nscf_symm,
                cls.inspect_nscf,
            ).else_(
                cls.run_scf_nbnd,
                cls.inspect_scf,
            ),
            cls.run_opengrid,
            cls.inspect_opengrid,
            if_(cls.should_do_projwfc)(
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
        spec.exit_code(410, 
            'ERROR_SUB_PROCESS_FAILED_PW',
            message='the scf/nscf WorkChain did not output a remote_folder node'
        )
        spec.exit_code(411,
            'ERROR_SUB_PROCESS_FAILED_OPENGRID',
            message='the OpengridCalculation sub process failed'
        )

    def should_do_nscf(self):
        """
        """
        opengrid = AttributeDict(self.inputs.opengrid)
        do_nscf = opengrid.run_nscf.value
        return do_nscf

    def run_nscf_symm(self):
        """
        Run the PwBaseWorkChain in nscf mode
        """
        inputs = self.prepare_nscf_inputs()

        # inputs.pw.pseudos is an AttributeDict, but calcfunction only accepts
        # orm.Data, so we unpack it to pass in orm.UpfData
        inputs.pw.parameters = update_nscf_num_bands(
            orm.Dict(dict=inputs.pw.parameters),
            self.ctx.workchain_scf.outputs.output_parameters,
            self.ctx.current_structure, self.inputs.only_valence,
            **inputs.pw.pseudos
        )
        self.report(
            'nscf number of bands set as ' +
            str(inputs.pw.parameters['SYSTEM']['nbnd'])
        )

        kmesh = self.prepare_kmesh(inputs)
        self.ctx.nscf_kmesh = kmesh  # store it since it will be used by w90
        inputs.kpoints = kmesh

        inputs = prepare_process_inputs(PwBaseWorkChain, inputs)
        running = self.submit(PwBaseWorkChain, **inputs)

        self.report(
            'nscf step - launching PwBaseWorkChain<{}> in {} mode'.format(
                running.pk, 'nscf'
            )
        )

        return ToContext(workchain_nscf=running)

    def run_scf_nbnd(self):
        """
        Run the PwBaseWorkChain in scf mode on the primitive cell of (optionally relaxed) input structure.
        """
        inputs = self.prepare_scf_inputs()

        # inputs.pw.pseudos is an AttributeDict, but calcfunction only accepts
        # orm.Data, so we unpack it to pass in orm.UpfData
        inputs.pw.parameters = update_scf_num_bands(
            orm.Dict(dict=inputs.pw.parameters),
            self.ctx.current_structure, self.inputs.only_valence,
            **inputs.pw.pseudos
        )
        self.report(
            'scf number of bands set as ' +
            str(inputs.pw.parameters['SYSTEM']['nbnd'])
        )

        kmesh = self.prepare_kmesh(inputs)
        self.ctx.nscf_kmesh = kmesh  # store it since it will be used by w90
        inputs.kpoints = kmesh

        inputs = prepare_process_inputs(PwBaseWorkChain, inputs)
        running = self.submit(PwBaseWorkChain, **inputs)

        self.report(
            'scf step - launching PwBaseWorkChain<{}> in {} mode'.format(
                running.pk, 'scf'
            )
        )

        return ToContext(workchain_scf=running)

    def run_opengrid(self):
        """open_grid.x to unfold kmesh
        """
        inputs = AttributeDict()
        inputs.code = AttributeDict(self.inputs.opengrid).code

        # ProjwfcCalculation requires the previous Calculation having
        # a input StructureData to parse atomic orbitals.
        try:
            if self.should_do_nscf():
                remote_folder = self.ctx.workchain_nscf.outputs.remote_folder
                # for inputs need pw__structure to access pw.structure
                structure = self.ctx.workchain_nscf.inputs.pw__structure
            else:
                remote_folder = self.ctx.workchain_scf.outputs.remote_folder
                # for inputs need pw__structure to access pw.structure
                structure = self.ctx.workchain_scf.inputs.pw__structure
        except AttributeError:
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_PW

        inputs.parent_folder = remote_folder
        inputs.structure = structure

        inputs = prepare_process_inputs(OpengridCalculation, inputs)
        running = self.submit(OpengridCalculation, **inputs)

        self.report(
            'open_grid step - launching OpengridCalculation<{}>'.format(
                running.pk
            )
        )
        return ToContext(calc_opengrid=running)

    def inspect_opengrid(self):
        """Verify that the OpengridCalculation run successfully finished."""
        workchain = self.ctx.calc_opengrid

        if not workchain.is_finished_ok:
            self.report(
                'OpengridCalculation failed with exit status {}'.format(
                    workchain.exit_status
                )
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_OPENGRID

        self.ctx.current_folder = workchain.outputs.remote_folder
        self.report("OpengridCalculation successfully finished")

def get_num_electrons_from_pseudos(structure, **pseudos):
    def get_nelecs_from_upf(upf):
        upf_name = upf.list_object_names()[0]
        upf_content = upf.get_object_content(upf_name)
        upf_content = upf_content.split('\n')
        # get PP_HEADER block
        ppheader_block = ''
        found_begin = False
        found_end = False
        for line in upf_content:
            if '<PP_HEADER' in line:
                ppheader_block += line + '\n'
                if not found_begin:
                    found_begin = True
                    continue
            if found_begin and ('/>' in line or '</PP_HEADER>' in line):
                ppheader_block += line + '\n'
                if not found_end:
                    found_end = True
                    break
            if found_begin:
                ppheader_block += line + '\n'
        print(ppheader_block)

        num_electrons = 0
        # parse XML
        import xml.etree.ElementTree as ET
        PP_HEADER = ET.XML(ppheader_block)
        if len(PP_HEADER.getchildren()) == 0:
            # old upf format, at the 6th line, e.g.
            # <PP_HEADER>
            #    0                   Version Number
            #   Be                   Element
            #    US                  Ultrasoft pseudopotential
            #     F                  Nonlinear Core Correction
            #  SLA  PW   PBX  PBC    PBE  Exchange-Correlation functional
            #     4.00000000000      Z valence
            #   -27.97245939710      Total energy
            #     0.00000    0.00000 Suggested cutoff for wfc and rho
            #     2                  Max angular momentum component
            #   769                  Number of points in mesh
            #     3    6             Number of Wavefunctions, Number of Projectors
            #  Wavefunctions         nl  l   occ
            #                        1S  0  2.00
            #                        2S  0  2.00
            #                        2P  1  0.00
            # </PP_HEADER>
            lines = ppheader_block.split('\n')[6]
            # some may have z_valence="1.300000000000000E+001", str -> float
            num_electrons = float(lines.strip().split()[0])
        else:
            # upf format 2.0.1, e.g.
            # <PP_HEADER
            #    generated="Generated using ONCVPSP code by D. R. Hamann"
            #    author="anonymous"
            #    date="180627"
            #    comment=""
            #    element="Ag"
            #    pseudo_type="NC"
            #    relativistic="scalar"
            #    is_ultrasoft="F"
            #    is_paw="F"
            #    is_coulomb="F"
            #    has_so="F"
            #    has_wfc="F"
            #    has_gipaw="F"
            #    core_correction="F"
            #    functional="PBE"
            #    z_valence="   19.00"
            #    total_psenergy="  -2.86827035760E+02"
            #    rho_cutoff="   1.39700000000E+01"
            #    l_max="2"
            #    l_local="-1"
            #    mesh_size="  1398"
            #    number_of_wfc="4"
            #    number_of_proj="6"/>
            num_electrons = float(PP_HEADER.get('z_valence'))
        return num_electrons

    tot_nelecs = 0
    composition = structure.get_composition()
    for kind in pseudos:
        upf = pseudos[kind]
        nelecs = get_nelecs_from_upf(upf)
        tot_nelecs += nelecs * composition[kind]
    return tot_nelecs

@calcfunction
def update_scf_num_bands(
    scf_input_parameters, structure, only_valence,
    **pseudos
):
    '''
    this calcfunction does 2 works:
        1. calculate nbnd based on scf output_parameters
        2. calculate number of projections based on pseudos
    The resulting nbnd is the max of the two.
    '''
    scf_in_dict = scf_input_parameters.get_dict()
    nspin = scf_in_dict['SYSTEM'].get('nspin', 1)
    nelectron = get_num_electrons_from_pseudos(structure, **pseudos)
    if only_valence:
        nbands = int(nelectron / 2)
    else:
        nbands = int(0.5 * nelectron * nspin + 4 * nspin)
        # nbands must > num_projections = num_wann
        nprojs = get_num_projections_from_pseudos(structure, **pseudos)
        nbands = max(nbands, nprojs + 10)
    scf_in_dict['SYSTEM']['nbnd'] = nbands
    return orm.Dict(dict=scf_in_dict)