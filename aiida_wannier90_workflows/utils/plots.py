#!/usr/bin/python
# -*- coding: utf-8 -*-
from aiida import orm
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_quantumespresso.workflows.pw.bands import PwBandsWorkChain
from aiida_wannier90.calculations import Wannier90Calculation
from aiida_wannier90_workflows.workflows import Wannier90BandsWorkChain, Wannier90WorkChain
from aiida_wannier90_workflows.utils.scdm import erfc_scdm, fit_scdm_mu_sigma_aiida
import matplotlib.pyplot as plt

def plot_scdm_fit(workchain: int, save: bool = False):
    """A function to plot the projectabilities distribution"""
    
    if workchain.process_class not in [
        Wannier90BandsWorkChain,
        Wannier90WorkChain
    ]:
        raise ValueError(f"Input workchain type should be {Wannier90BandsWorkChain}")

    formula = workchain.inputs.structure.get_formula()

    w90calc = workchain.get_outgoing(link_label_filter='wannier90').one().node
    p2wcalc = workchain.get_outgoing(link_label_filter='pw2wannier90').one().node
    projcalc = workchain.get_outgoing(link_label_filter='projwfc').one().node

    fermi_energy = w90calc.inputs.parameters['fermi_energy']
    sigma = p2wcalc.inputs.parameters['inputpp']['scdm_sigma']
    mu = p2wcalc.inputs.parameters['inputpp']['scdm_mu']
    projections = projcalc.outputs.projections
    bands = projcalc.outputs.bands

    mu_fit, sigma_fit, data = fit_scdm_mu_sigma_aiida(
        bands, projections, sigma_factor=orm.Float(0), return_data=True
    )

    print(f"{formula:6s}:")
    print(f"        fermi_energy = {fermi_energy}, mu = {mu}, sigma = {sigma}")

    # check the fitting are consistent
    eps = 1e-6
    assert abs(sigma - sigma_fit) < eps
    sigma_factor = workchain.inputs.scdm_sigma_factor.value
    assert abs(mu - (mu_fit - sigma_fit * sigma_factor)) < eps
    sorted_bands = data[0, :]
    sorted_projwfc = data[1, :]

    plt.figure()
    plt.plot(sorted_bands, sorted_projwfc, 'o')
    plt.plot(sorted_bands, erfc_scdm(sorted_bands, mu_fit, sigma_fit))
    plt.axvline([mu_fit], color='red', label=r"$\mu$")
    plt.axvline([mu_fit - sigma_factor * sigma_fit],
                color='orange',
                label=r"$\mu-" + str(sigma_factor) + r"\sigma$")
    plt.axvline([fermi_energy], color='green', label=r"$E_f$")
    plt.title(f"{workchain.process_label}<{workchain.pk}>: {formula}")
    plt.xlabel('Energy [eV]')
    plt.ylabel('Projectability')
    plt.legend(loc='best')

    if save:
        plt.savefig(f'scdmfit_{formula}_{workchain.pk}.png')
    else:
        plt.show()

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
        if workchain.process_class == PwBandsWorkChain:
            return workchain0.outputs.band_structure
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
