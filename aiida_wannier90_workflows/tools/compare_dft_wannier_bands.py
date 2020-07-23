#!/usr/bin/env runaiida
"""compare DFT and Wannier band structures
"""
import argparse
from aiida import orm

def required_length(nmin, nmax):
    class RequiredLength(argparse.Action):
        def __call__(self, parser, args, values, option_string=None):
            if not nmin<=len(values)<=nmax:
                msg='argument "{f}" requires between {nmin} and {nmax} arguments'.format(
                    f=self.dest,nmin=nmin,nmax=nmax)
                raise argparse.ArgumentTypeError(msg)
            setattr(args, self.dest, values)
    return RequiredLength

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=
    f'Plot DFT and Wannier bands for comparison.')
    parser.add_argument('pk', metavar='PK', type=int, nargs='+', action=required_length(1,2), help='The PK of a Wannier90BandsWorkChain, the `compare_dft_bands` inputs of the Wannier90BandsWorkChain should be True; or PKs of 2 BandsData to be compared.')
    parser.add_argument('-s', '--save', action='store_true', help="save as a python plotting script instead of showing matplotlib window")
    args = parser.parse_args()

    input_is_workchain = len(args.pk) == 1
    if input_is_workchain:
        workchain = orm.load_node(args.pk[0])
        dft_bands = workchain.outputs.dft_bands
        wan_bands = workchain.outputs.wannier90_interpolated_bands
    else:
        dft_bands = orm.load_node(args.pk[0])
        wan_bands = orm.load_node(args.pk[1])

    # dft_bands.show_mpl()
    dft_mpl_code = dft_bands._exportcontent(fileformat='mpl_singlefile', legend=f'{dft_bands.pk}', main_file_name='')[0]
    wan_mpl_code = wan_bands._exportcontent(fileformat='mpl_singlefile', legend=f'{wan_bands.pk}', main_file_name='',
    bands_color='r', bands_linestyle='dashed')[0]

    dft_mpl_code = dft_mpl_code.replace(b'pl.show()', b'')
    wan_mpl_code = wan_mpl_code.replace(b'fig = pl.figure()', b'')
    wan_mpl_code = wan_mpl_code.replace(b'p = fig.add_subplot(1,1,1)', b'')
    mpl_code = dft_mpl_code + wan_mpl_code

    formula = workchain.inputs.structure.get_formula()
    # add title
    if input_is_workchain:
        replacement = f'workchain pk {workchain.pk}, {formula}, dft_bands pk {dft_bands.pk}, wan_bands pk {wan_bands.pk}'
    else:
        replacement = f'1st bands pk {dft_bands.pk}, 2nd bands pk {wan_bands.pk}'
    replacement = f'p.set_title("{replacement}")\npl.show()'
    mpl_code = mpl_code.replace(b'pl.show()', replacement.encode())

    if input_is_workchain:
        fname = f'bandsdiff_{formula}_{workchain.pk}.py'
    else:
        fname = f'bandsdiff_{dft_bands.pk}_{wan_bands.pk}.png'

    if args.save:
        with open(fname, 'w') as f:
            f.write(mpl_code.decode())
    else:
        exec(mpl_code)