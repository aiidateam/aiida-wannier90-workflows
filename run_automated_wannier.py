#!/usr/bin/env runaiida
import argparse
from aiida import orm
from aiida.engine import submit
from aiida.common.exceptions import NotExistent
# from ase.io import read as aseread
from pymatgen.core import Structure
from aiida_wannier90_workflows.workflows import Wannier90BandsWorkChain
from aiida_quantumespresso.common.types import SpinType
import numpy as np

# Please modify these according to your machine
str_pw = 'qe-pw-7.1@magpu'
str_pw2wan = 'qe-pw2wannier90@magpu'
str_projwfc = 'qe-projwfc@magpu'
str_wan = 'wannier90@magpu'

group_name = 'scdm_workflow'


def check_codes():
    # will raise NotExistent error
    try:
        codes = dict(
            pw=orm.load_code(str_pw),
            pw2wannier90=orm.load_code(str_pw2wan),
            projwfc=orm.load_code(str_projwfc),
            wannier90=orm.load_code(str_wan),
        )
    except NotExistent as e:
        print(e)
        print(
            'Please modify the code labels in this script according to your machine'
        )
        exit(1)
    return codes


def parse_arugments():
    parser = argparse.ArgumentParser(
        description=
        "A script to run the AiiDA workflows to automatically compute the MLWF using the SCDM method and the automated protocol described in the Vitale et al. paper"
    )
    parser.add_argument(
        "cif", metavar="cif_fileNAME", help="path to an input structure file (CIF, MCIF, XSF,...)"
    )
    parser.add_argument(
        '-p',
        "--protocol",
        help="available protocols are 'moderate', 'precise', and 'fast'",
        default="fast"
    )
    parser.add_argument(
        '-m',
        "--do-mlwf",
        help="do maximal localization of Wannier functions",
        action="store_false"
    )
    parser.add_argument(
        '-d',
        "--do-disentanglement",
        help=
        "do disentanglement in Wanner90 step (This should be False, otherwise band structure is not optimal!)",
        action="store_true"
    )
    parser.add_argument(
        '-v',
        "--only-valence",
        help=
        "Compute only for valence bands (you must be careful to apply this only for insulators!)",
        action="store_true"
    )
    parser.add_argument(
        '-r',
        "--retrieve-hamiltonian",
        help="Retrieve Wannier Hamiltonian after the workflow finished",
        action="store_true"
    )
    parser.add_argument(
        "--soi",
        help="Consider Spin-Orbit Interaction",
        action="store_true"
    )
    args = parser.parse_args()
    return args

def get_initial_moment(magmoms, threshold=1e-6):
    """
    Return a dictionary with the magnetic moments in the QuantumEspresso format
    """
    # Check collinearity
    collinear = True
    for mom in magmoms:
        if mom[0] > threshold or mom[1] > threshold:
            collinear = False
            break
    init_mom={}
    for i, mom in enumerate(magmoms):
        num=str(i+1)
        m = np.linalg.norm(mom[:3],ord=2)
        if m > threshold:
            mtheta=np.arccos(mom[2]/m)
            mphi=np.arctan2(mom[1],mom[0])
        else:
            mtheta=0
            mphi=0
        
        init_mom[f'starting_magnetization({num})'] = m
        if not collinear:
            init_mom[f'angle1({num})'] = mtheta
            init_mom[f'angle2({num})'] = mphi
    return init_mom, collinear

def read_structure(cif_file):
    pmg=Structure.from_file(cif_file)
    structure = orm.StructureData(pymatgen=pmg)
    if pmg.site_properties.get('magmom',None) is not None:
        init_mom, collinear = get_initial_moment(pmg.site_properties['magmom'])
        structure.base.extras.set('magmom',init_mom)
        structure.base.extras.set('collinear',collinear)
    structure.store()
    print(
        'Structure {} read and stored with pk {}.'.format(
            structure.get_formula(), structure.pk
        )
    )
    return structure


def update_group_name(
    group_name, only_valence, do_disen, do_mlwf, exclude_bands=None
):
    if only_valence:
        group_name += "_onlyvalence"
    else:
        group_name += "_withconduction"
    if do_disen:
        group_name += '_disentangle'
    if do_mlwf:
        group_name += '_mlwf'
    if exclude_bands is not None:
        group_name += '_excluded{}'.format(len(exclude_bands))
    return group_name


def add_to_group(node, group_name):
    if group_name is not None:
        try:
            g = orm.Group.get(label=group_name)
            group_statistics = "that already contains {} nodes".format(
                len(g.nodes)
            )
        except NotExistent:
            g = orm.Group(label=group_name)
            group_statistics = "that does not exist yet"
            g.store()
        g.add_nodes(node)
        print(
            "Wannier90BandsWorkChain<{}> will be added to the group {} {}".
            format(node.pk, group_name, group_statistics)
        )


def print_help(workchain, structure):
    print(
        'launched Wannier90BandsWorkChain pk {} for structure {}'.format(
            workchain.pk, structure.get_formula()
        )
    )
    print('')
    print('# To get a detailed state of the workflow, run:')
    print('verdi process report {}'.format(workchain.pk))


def submit_workchain(
    cif_file, protocol, only_valence, do_disentanglement, do_mlwf,
    retrieve_hamiltonian, group_name, soi
):
    codes = check_codes()

    group_name = update_group_name(
        group_name, only_valence, do_disentanglement, do_mlwf
    )

    if isinstance(cif_file, orm.StructureData):
        structure = cif_file
    else:
        structure = read_structure(cif_file)

    controls = {
        'retrieve_hamiltonian': orm.Bool(retrieve_hamiltonian),
        'only_valence': orm.Bool(only_valence),
        'do_disentanglement': orm.Bool(do_disentanglement),
        'do_mlwf': orm.Bool(do_mlwf)
    }

    if only_valence:
        print(
            "Running only_valence/insulating for {}".format(
                structure.get_formula()
            )
        )
    else:
        print(
            "Running with conduction bands for {}".format(
                structure.get_formula()
            )
        )

    spintype = SpinType.NONE
    if soi:
        spintype = SpinType.SPIN_ORBIT

    # wannier90_workchain_parameters = {
    #     "code": {
    #         'pw': codes['pw_code'],
    #         'pw2wannier90': codes['pw2wannier90_code'],
    #         'projwfc': codes['projwfc_code'],
    #         'wannier90': codes['wannier90_code']
    #     },
    #     "protocol": orm.Dict(dict={'name': protocol}),
    #     "structure": structure,
    #     "controls": controls,
    #     "spin_type": spintype
    # }
    # workchain = submit(
    #     Wannier90BandsWorkChain, **wannier90_workchain_parameters
    # )
    builder = Wannier90BandsWorkChain.get_builder_from_protocol(
        codes,
        structure,
        protocol=protocol,
        retrieve_hamiltonian=retrieve_hamiltonian,
        # controls=controls,
        spin_type=spintype
    )
    workchain = submit(builder)

    add_to_group(workchain, group_name)
    print_help(workchain, structure)
    return workchain.pk


if __name__ == "__main__":
    args = parse_arugments()

    submit_workchain(
        args.cif, args.protocol, args.only_valence, args.do_disentanglement,
        args.do_mlwf, args.retrieve_hamiltonian, group_name, args.soi
    )