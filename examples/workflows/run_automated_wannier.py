#!/usr/bin/env runaiida
import argparse
from aiida import orm
from aiida.engine import submit
from aiida.common.exceptions import NotExistent
from ase.io import read as aseread
from aiida_wannier90_workflows.workflows import Wannier90BandsWorkChain

# Please modify these according to your machine
str_pw = 'qe-git-pw@localhost'
str_pw2wan = 'qe-git-pw2wannier90@localhost'
str_projwfc = 'qe-git-projwfc@localhost'
str_wan = 'wannier90-git-wannier90@localhost'
str_opengrid = 'qe-git-opengrid@localhost'

group_name = 'scdm_workflow'


def check_codes():
    # will raise NotExistent error
    try:
        codes = dict(
            pw=orm.Code.get_from_string(str_pw),
            pw2wannier90=orm.Code.get_from_string(str_pw2wan),
            projwfc=orm.Code.get_from_string(str_projwfc),
            wannier90=orm.Code.get_from_string(str_wan),
        )
    except NotExistent as e:
        print(e)
        print(
            'Please modify the code labels in this script according to your machine'
        )
        exit(1)
    # optional code
    try:
        codes['opengrid'] = orm.Code.get_from_string(str_opengrid)
    except NotExistent:
        pass
    return codes


def parse_arugments():
    parser = argparse.ArgumentParser(
        description=
        "A script to run the AiiDA workflows to automatically compute the MLWF using the SCDM method and the automated protocol described in the Vitale et al. paper"
    )
    parser.add_argument(
        "xsf", metavar="XSF_FILENAME", help="path to an input XSF file"
    )
    parser.add_argument(
        '-p',
        "--protocol",
        help="available protocols are 'theos-ht-1.0' and 'testing'",
        # default="theos-ht-1.0"
        default="testing"
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
    args = parser.parse_args()
    return args


def read_structure(xsf_file):
    structure = orm.StructureData(ase=aseread(xsf_file))
    structure.store()
    print(
        'Structure {} read and stored with pk {}'.format(
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
    print(
        '\n'
        '# To get a detailed state of the workflow, run:\n'
        f'verdi process report {workchain.pk}\n'
        '\n'
        'Several tools for visualization, after workchain finished, launch as:\n'
        f'    ../../aiida_wannier90_workflows/tools/plot_projectabilities.py {workchain.pk}\n'
        f'    ../../aiida_wannier90_workflows/tools/compare_dft_wannier_bands.py {workchain.pk}\n'
    )


def submit_workchain(
    xsf_file, protocol, only_valence, do_disentanglement, do_mlwf,
    retrieve_hamiltonian, group_name
):
    codes = check_codes()

    group_name = update_group_name(
        group_name, only_valence, do_disentanglement, do_mlwf
    )

    if isinstance(xsf_file, orm.StructureData):
        structure = xsf_file
    else:
        structure = read_structure(xsf_file)

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

    wannier90_workchain_parameters = {
        "codes": codes,
        "structure": structure,
        "protocol": orm.Dict(dict={'name': protocol}),
        'only_valence': orm.Bool(only_valence),
        #'disentanglement': orm.Bool(do_disentanglement),
        'disentanglement': orm.Bool(True),
        'maximal_localisation': orm.Bool(do_mlwf),
        'retrieve_hamiltonian': orm.Bool(retrieve_hamiltonian),
        'scdm_projections': orm.Bool(False),
        #'spdf_projections': orm.Bool(True),
        'pswfc_projections': orm.Bool(True),
        'auto_froz_max': orm.Bool(False),
        # optional
        'use_opengrid': orm.Bool(False),
        'compare_dft_bands': orm.Bool(True),
        'spin_orbit_coupling': orm.Bool(False)
    }

    workchain = submit(
        Wannier90BandsWorkChain, **wannier90_workchain_parameters
    )

    add_to_group(workchain, group_name)
    print_help(workchain, structure)
    return workchain.pk


if __name__ == "__main__":
    args = parse_arugments()

    #args.xsf = orm.load_node(2562)
    submit_workchain(
        args.xsf, args.protocol, args.only_valence, args.do_disentanglement,
        args.do_mlwf, args.retrieve_hamiltonian, group_name
    )
