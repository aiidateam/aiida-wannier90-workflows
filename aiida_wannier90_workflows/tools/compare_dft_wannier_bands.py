#!/usr/bin/env runaiida
"""compare DFT and Wannier band structures
"""
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

def get_mpl_code_for_bands(dft_bands, wan_bands, fermi_energy=None, title=None, save=False, filename=None):
    # dft_bands.show_mpl()
    dft_mpl_code = dft_bands._exportcontent(fileformat='mpl_singlefile', legend=f'{dft_bands.pk}', main_file_name='')[0]
    wan_mpl_code = wan_bands._exportcontent(fileformat='mpl_singlefile', legend=f'{wan_bands.pk}', main_file_name='',
    bands_color='r', bands_linestyle='dashed')[0]

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

def get_mpl_code_for_workchain(workchain, title=None, save=False, filename=None):
    dft_bands = workchain.outputs.dft_bands
    wan_bands = workchain.outputs.wannier90_interpolated_bands

    formula = workchain.inputs.structure.get_formula()
    if title is None:
        title = f'workchain pk {workchain.pk}, {formula}, dft_bands pk {dft_bands.pk}, wan_bands pk {wan_bands.pk}'

    if save and (filename is None):
        filename = f'bandsdiff_{formula}_{workchain.pk}.py'

    fermi_energy = workchain.outputs.scf_parameters['fermi_energy']

    mpl_code = get_mpl_code_for_bands(dft_bands, wan_bands, fermi_energy, title, save, filename)

    return mpl_code

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=
    f'Plot DFT and Wannier bands for comparison.')
    parser.add_argument('pk', metavar='PK', type=int, nargs='+', action=required_length(1,2), help='The PK of a Wannier90BandsWorkChain, the `compare_dft_bands` inputs of the Wannier90BandsWorkChain should be True; or PKs of 2 BandsData to be compared.')
    parser.add_argument('-s', '--save', action='store_true', help="save as a python plotting script instead of showing matplotlib window")
    args = parser.parse_args()

    input_is_workchain = len(args.pk) == 1
    if input_is_workchain:
        workchain = orm.load_node(args.pk[0])
        mpl_code = get_mpl_code_for_workchain(workchain, save=args.save)
    else:
        dft_bands = orm.load_node(args.pk[0])
        wan_bands = orm.load_node(args.pk[1])
        mpl_code = get_mpl_code_for_bands(dft_bands, wan_bands, save=args.save)

    # print(mpl_code.decode())

    if not args.save:
        exec(mpl_code)
