#!/usr/bin/env runaiida
"""compare DFT and Wannier band structures
"""
from aiida import orm
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_wannier90_workflows.workflows.bands import Wannier90BandsWorkChain
from aiida_wannier90.calculations import Wannier90Calculation


def required_length(nmin, nmax):
    class RequiredLength(argparse.Action):
        def __call__(self, parser, args, values, option_string=None):
            if not nmin <= len(values) <= nmax:
                msg = 'argument "{f}" requires between {nmin} and {nmax} arguments'.format(
                    f=self.dest, nmin=nmin, nmax=nmax
                )
                raise argparse.ArgumentTypeError(msg)
            setattr(args, self.dest, values)

    return RequiredLength


def get_mpl_code_for_bands(
    dft_bands,
    wan_bands,
    fermi_energy=None,
    title=None,
    save=False,
    filename=None
):
    # dft_bands.show_mpl()
    dft_mpl_code = dft_bands._exportcontent(
        fileformat='mpl_singlefile',
        legend=f'{dft_bands.pk}',
        main_file_name=''
    )[0]
    wan_mpl_code = wan_bands._exportcontent(
        fileformat='mpl_singlefile',
        legend=f'{wan_bands.pk}',
        main_file_name='',
        bands_color='r',
        bands_linestyle='dashed'
    )[0]

    dft_mpl_code = dft_mpl_code.replace(b'pl.show()', b'')
    wan_mpl_code = wan_mpl_code.replace(b'fig = pl.figure()', b'')
    wan_mpl_code = wan_mpl_code.replace(b'p = fig.add_subplot(1,1,1)', b'')
    mpl_code = dft_mpl_code + wan_mpl_code

    if title is None:
        title = f'1st bands pk {dft_bands.pk}, 2nd bands pk {wan_bands.pk}'
    replacement = f'p.set_title("{title}")\npl.show()'
    mpl_code = mpl_code.replace(b'pl.show()', replacement.encode())

    if fermi_energy is not None:
        replacement = f"\nfermi_energy =  {fermi_energy}\n"
        replacement += f"p.axhline(y=fermi_energy, color='blue', linestyle='--', label='Fermi', zorder=-1)\n"
        replacement += 'pl.legend()\npl.show()\n'
        mpl_code = mpl_code.replace(b'pl.show()', replacement.encode())

    if save:
        if filename is None:
            filename = f'bandsdiff_{dft_bands.pk}_{wan_bands.pk}.py'
        with open(filename, 'w') as f:
            f.write(mpl_code.decode())

    return mpl_code


def get_mpl_code_for_workchains(
    workchain0, workchain1, title=None, save=False, filename=None
):
    def get_output_bands(workchain):
        if workchain.process_class == PwBaseWorkChain:
            return workchain0.outputs.output_band
        elif workchain.process_class == Wannier90BandsWorkChain:
            return workchain.outputs.band_structure
        elif workchain.process_class == Wannier90Calculation:
            return workchain.outputs.interpolated_bands
        else:
            raise ValueError(f"Unrecognized workchain type: {workchain}")

    # assume workchain0 is pw, workchain1 is wannier
    dft_bands = get_output_bands(workchain0)
    wan_bands = get_output_bands(workchain1)

    formula = workchain1.inputs.structure.get_formula()
    if title is None:
        title = f'{formula}, {workchain0.process_label}<{workchain0.pk}> bands<{dft_bands.pk}>, '
        title += f'{workchain1.process_label}<{workchain1.pk}> bands<{wan_bands.pk}>'

    if save and (filename is None):
        filename = f'bandsdiff_{formula}_{workchain0.pk}_{workchain1.pk}.py'

    fermi_energy = workchain1.outputs['scf']['output_parameters'
                                             ]['fermi_energy']

    mpl_code = get_mpl_code_for_bands(
        dft_bands, wan_bands, fermi_energy, title, save, filename
    )

    return mpl_code


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description=f'Plot DFT and Wannier bands for comparison.'
    )
    parser.add_argument(
        'pk',
        metavar='PK',
        type=int,
        nargs='+',
        action=required_length(2, 2),
        help=
        'The PKs of a PwBaseWorkChain and a Wannier90BandsWorkChain, or PKs of 2 BandsData to be compared.'
    )
    parser.add_argument(
        '-s',
        '--save',
        action='store_true',
        help=
        "save as a python plotting script instead of showing matplotlib window"
    )
    args = parser.parse_args()

    pk0 = orm.load_node(args.pk[0])
    pk1 = orm.load_node(args.pk[1])
    input_is_workchain = isinstance(pk0, orm.WorkChainNode
                                    ) and isinstance(pk1, orm.WorkChainNode)
    if input_is_workchain:
        mpl_code = get_mpl_code_for_workchains(pk0, pk1, save=args.save)
    else:
        mpl_code = get_mpl_code_for_bands(pk0, pk1, save=args.save)

    # print(mpl_code.decode())

    if not args.save:
        exec(mpl_code)
