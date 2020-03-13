#!/usr/bin/env python
import argparse
import aiida
aiida.load_profile()
from aiida import orm
from aiida.engine import submit
from aiida.common.exceptions import NotExistent
from ase.io import read as aseread
from aiida_wannier90_theosworkflows.workflows import Wannier90BandsWorkChain

# Please modify these according to your machine
str_pw = 'qe-6.5-pw@localhost'
str_pw2wan = 'qe-6.5-pw2wannier90@localhost'
str_projwfc = 'qe-6.5-projwfc@localhost'
str_wan = 'wannier90-3.1.0-wannier@localhost'

group_name = 'scdm_workflow'

def check_codes():
    # will raise NotExistent error
    try:
        codes = dict(
            pw_code = orm.Code.get_from_string(str_pw),
            pw2wannier90_code = orm.Code.get_from_string(str_pw2wan),
            projwfc_code = orm.Code.get_from_string(str_projwfc),
            wannier90_code = orm.Code.get_from_string(str_wan),
        )
    except NotExistent as e:
        print(e)
        print('Please modify the code labels in this script according to your machine')
        exit(1)
    return codes

def parse_arugments():
    parser = argparse.ArgumentParser(
        description=
        "A script to run the AiiDA workflows to automatically compute the MLWF using the SCDM method and the automated protocol described in the Vitale et al. paper"
    )
    parser.add_argument(
        "xsf",
        metavar="XSF_FILENAME",
        help="path to an input XSF file"
    )
    parser.add_argument(
        '-p',
        "--protocol",
        help=
        "available protocols are 'theos-ht-1.0' and 'testing'",
        default="testing")
    parser.add_argument(
        '-m',
        "--do-mlwf",
        help=
        "do maximal localization of Wannier functions",
        action="store_false")
    parser.add_argument(
        '-d',
        "--do-disentanglement",
        help=
        "do disentanglement in Wanner90 step (This should be False, otherwise band structure is not optimal!)",
        action="store_true")
    parser.add_argument(
        '-v',
        "--only-valence",
        help=
        "Compute only for valence bands (you must be careful to apply this only for insulators!)",
        action="store_true")
    parser.add_argument(
        '-r',
        "--retrieve-hamiltonian",
        help=
        "Retrieve Wannier Hamiltonian after the workflow finished",
        action="store_true")
    args = parser.parse_args()
    return args

def read_structure(xsf_file):
    structure = orm.StructureData(ase=aseread(xsf_file))
    structure.store()
    print('Structure {} read and stored with pk {}.'.format(
        structure.get_formula(), structure.pk))
    return structure

def update_group_name(group_name, only_valence, do_disen, do_mlwf, exclude_bands=None):
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
            group_statistics = "that already contains {} nodes".format(len(g.nodes))
        except NotExistent:
            g = orm.Group(label=group_name)
            group_statistics = "that does not exist yet"
            g.store()
        g.add_nodes(node)
        print("Wannier90BandsWorkChain<{}> will be added to the group {} {}".format(
            node.pk, group_name, group_statistics))

def print_help(workchain, structure):
    print('launched {} pk {} for structure {}'.format(
            workchain.process_type, workchain.pk, structure.get_formula()))
    print('')

    # print('#' * 72)
    # print('# INSTRUCTIONS ON HOW TO CONTINUE FROM HERE')
    # print('# AiiDA is now running the workflow.')
    # print(
    #     '# First of all, you can enter the AiiDA virtual environment typing:'
    # )
    # print('#    workon aiida')
    # print(
    #     '# You can monitor the state of the workflow with the following commands:'
    # )
    # print('# * Monitor the state of the workflow')
    # print('#     verdi process list -p1 -a')
    # print(
    #     '#   and check the status of the workflow with PK={}.'.format(
    #         wc_run.pk))
    print('# * Get a detailed state of the running workfow:')
    print('verdi process report {}'.format(workchain.pk))
        # print('#')
        # print(
        #     '# Once the workflow is finished, you can find the ID of the band'
        # )
        # print('# structure using the command:')
        # print('#     verdi node show {}'.format(wc_run.pk))
        # print(
        #     '# and then getting the value from the column PK for the row')
        # print(
        #     '# corresponding to the link label "MLWF_interpolated_bands".')
        # print('# ')
        # print('# With this PK, you can show the bands with xmgrace using')
        # print('# (replace <PK> with the PK you found in the step before):')
        # print('#   verdi data bands show -f xmgrace <PK>')
        # print('# or export to file using:')
        # print('#   verdi data bands export -f agr <PK>')
        # print('# ')
        # print('# If you want to compare also with the DFT band structure,')
        # print('# you can run the script')
        # print('#    ~/run_DFT_bands.py {}'.format(wc_run.pk))
        # print(
        #     '# (where {} is the PK of this Wannier90 workchain, passed in order'
        #     .format(wc_run.pk))
        # print('# to reuse its input structure and kpoints path.')
        # print('# Then, follow the instructions printed by that script.')
        # print('#' * 72)
        # print(
        #     '# Thanks for using the automated Wannier function workflows with AiiDA!'
        # )
        # print(
        #     '# Read instructions above to know how to check the results.')
        # print('#' * 72)
    # else:
    #     print('would launch {}'.format(structure.get_formula()))
    #print('Output will be added to group: {}'.format(g_name))

def submit_workchain(xsf_file, protocol, only_valence, do_disentanglement, do_mlwf, retrieve_hamiltonian, group_name):
    codes = check_codes()

    group_name = update_group_name(
       group_name, only_valence, do_disentanglement, do_mlwf)

    if isinstance(xsf_file, orm.StructureData):
        structure = xsf_file
    else:
        structure = read_structure(xsf_file)

    controls = {
        'retrieve_hamiltonian': orm.Bool(retrieve_hamiltonian),
        'only_valence': orm.Bool(only_valence),
        'do_disentanglement': orm.Bool(do_disentanglement),
        'do_mlwf': orm.Bool(do_mlwf)
    }

    if only_valence:
        print("Running only_valence/insulating, do_mlwf={} for {}".format(
            do_mlwf, structure.get_formula()))
    else:
        print("Running with conduction bands, do_disentanglement={}, do_mlwf={} for {}".
                format(do_disentanglement, do_mlwf, structure.get_formula()))

    wannier90_workchain_parameters = {
        "code": {
            'pw': codes['pw_code'],
            'pw2wannier90': codes['pw2wannier90_code'],
            'projwfc': codes['projwfc_code'],
            'wannier90': codes['wannier90_code']
        },
        "protocol": orm.Dict(dict={'name': protocol}),
        "structure": structure,
        "controls": controls
    }
    # print(wannier90_workchain_parameters)

    workchain = submit(Wannier90BandsWorkChain,
        **wannier90_workchain_parameters
    )
    # workchain = orm.load_node(13996)

    add_to_group(workchain, group_name)
    print_help(workchain, structure)
    return workchain.pk
   
if __name__ == "__main__":
    args = parse_arugments()
    # print(args)

    submit_workchain(
        args.xsf,
        args.protocol,
        args.only_valence,
        args.do_disentanglement,
        args.do_mlwf,
        args.retrieve_hamiltonian,
        group_name
    )
