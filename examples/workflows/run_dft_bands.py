#!/usr/bin/env runaiida
import argparse
from aiida import orm
from aiida.engine import submit, run_get_node
from ase.io import read as aseread
from aiida_wannier90_workflows.workflows import Wannier90BandsWorkChain, PwBandStructureWorkChain

# Please modify these according to your machine
code_str = 'qe-git-pw@localhost'
code = orm.Code.get_from_string(code_str)

def read_structure(xsf_file):
    structure = orm.StructureData(ase=aseread(xsf_file))
    structure.store()
    print(
        'Structure {} read and stored with pk {}.'.format(
            structure.get_formula(), structure.pk
        )
    )
    return structure

def parse_argugments():
    parser = argparse.ArgumentParser(
        "A script to run the DFT band structure (without structural relax) using Quantum ESPRESSO starting from an automated workflow (to reuse structure), or from XSF file."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-x", "--xsf", help="path to an input XSF file")
    group.add_argument("-w", "--workchain", help=
            "The PK of the Wannier90BandsWorkChain - if you didn't run it, run it first using the ./run_automated_wannier.py script"
        )
    args = parser.parse_args()
    if args.xsf is not None:
        structure = read_structure(args.xsf)
    else:
        wannier90_workchain_pk = int(args.workchain)
        try:
            wannier90_workchain = orm.load_node(wannier90_workchain_pk)
        except Exception:
            print(
                "I could not load an AiiDA node with PK={}, did you use the correct PK?"
                .format(wannier90_workchain))
            exit()
        if wannier90_workchain.process_class != Wannier90BandsWorkChain:
            print(
                "The node with PK={} is not a Wannier90BandsWorkChain, it is a {}"
                .format(wannier90_workchain_pk, type(wannier90_workchain)))
            print(
                "Please pass a node that was the output of the Wannier90 workflow executed using"
            )
            print("the ./run_automated_wannier.py script.")
            exit()
        structure = wannier90_workchain.inputs.structure
    return structure

def submit_workchain(structure):
    print("running dft band structure calculation for {}".format(
        structure.get_formula()))

    from aiida_quantumespresso.utils.resources import get_default_options

    # Submit the DFT bands workchain
    dft_workchain = submit(
        PwBandStructureWorkChain,
        code=code,
        structure=structure,
        #protocol=orm.Dict(dict={'name': 'theos-ht-1.0'}),
        protocol=orm.Dict(dict={'name': 'testing'}),
        options=orm.Dict(dict=get_default_options(
            max_wallclock_seconds=3600*5, 
            with_mpi=True))
    )
    return dft_workchain

def print_help(workchain, structure):
    print(
        'launched Wannier90BandsWorkChain pk {} for structure {}'.format(
            workchain.pk, structure.get_formula()
        )
    )
    print('')
    print('# To get a detailed state of the workflow, run:')
    print('verdi process report {}'.format(workchain.pk))

if __name__ == "__main__":
    structure = parse_argugments()
    workchain = submit_workchain(structure)
    print_help(workchain, structure)