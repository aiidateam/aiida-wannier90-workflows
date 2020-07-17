#!/usr/bin/env runaiida
"""compare DFT and Wannier band structures
"""
import argparse
from aiida import orm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=
    f'Plot DFT and Wannier bands for comparison. The compare_dft_bands of the Wannier90BandsWorkChain should be True.')
    parser.add_argument('pk', metavar='WORKCHAIN_PK', type=int, help='The PK of a Wannier90BandsWorkChain')
    args = parser.parse_args()

    workchain = orm.load_node(args.pk)
    
    dft_bands = workchain.outputs.dft_bands
    wan_bands = workchain.outputs.wannier90_interpolated_bands

    # dft_bands.show_mpl()
    dft_mpl_code = dft_bands._exportcontent(fileformat='mpl_singlefile', main_file_name='')[0]
    wan_mpl_code = wan_bands._exportcontent(fileformat='mpl_singlefile', main_file_name='',
    bands_color='r', bands_linestyle='dashed')[0]

    dft_mpl_code = dft_mpl_code.replace(b'pl.show()', b'')
    wan_mpl_code = wan_mpl_code.replace(b'fig = pl.figure()', b'')
    wan_mpl_code = wan_mpl_code.replace(b'p = fig.add_subplot(1,1,1)', b'')
    exec(dft_mpl_code + wan_mpl_code)