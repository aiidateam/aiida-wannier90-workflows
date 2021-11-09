#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Plot band structures."""
import typing as ty
import matplotlib.pyplot as plt
from aiida import orm
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_quantumespresso.workflows.pw.bands import PwBandsWorkChain
from aiida_wannier90.calculations import Wannier90Calculation
from aiida_wannier90_workflows.workflows import Wannier90BandsWorkChain, Wannier90WorkChain
from aiida_wannier90_workflows.utils.scdm import erfc_scdm, fit_scdm_mu_sigma_aiida


def plot_scdm_fit(workchain: int, save: bool = False):
    """Plot the projectabilities distribution of SCDM fitting."""

    if workchain.process_class not in [Wannier90BandsWorkChain, Wannier90WorkChain]:
        raise ValueError(f'Input workchain type should be {Wannier90BandsWorkChain}')

    formula = workchain.inputs.structure.get_formula()

    w90calc = workchain.get_outgoing(link_label_filter='wannier90').one().node
    p2wcalc = workchain.get_outgoing(link_label_filter='pw2wannier90').one().node
    projcalc = workchain.get_outgoing(link_label_filter='projwfc').one().node

    fermi_energy = w90calc.inputs.parameters['fermi_energy']
    sigma = p2wcalc.inputs.parameters['inputpp']['scdm_sigma']
    mu = p2wcalc.inputs.parameters['inputpp']['scdm_mu']
    projections = projcalc.outputs.projections
    bands = projcalc.outputs.bands

    mu_fit, sigma_fit, data = fit_scdm_mu_sigma_aiida(bands, projections, sigma_factor=orm.Float(0), return_data=True)

    print(f'{formula:6s}:')
    print(f'        fermi_energy = {fermi_energy}, mu = {mu}, sigma = {sigma}')

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
    plt.axvline([mu_fit], color='red', label=r'$\mu$')
    plt.axvline([mu_fit - sigma_factor * sigma_fit], color='orange', label=r'$\mu-' + str(sigma_factor) + r'\sigma$')
    plt.axvline([fermi_energy], color='green', label=r'$E_f$')
    plt.title(f'{workchain.process_label}<{workchain.pk}>: {formula}')
    plt.xlabel('Energy [eV]')
    plt.ylabel('Projectability')
    plt.legend(loc='best')

    if save:
        plt.savefig(f'scdmfit_{formula}_{workchain.pk}.png')
    else:
        plt.show()


def get_mpl_code_for_bands(dft_bands, wan_bands, fermi_energy=None, title=None, save=False, filename=None):
    """Return matplotlib code for comparing band structures."""

    # dft_bands.show_mpl()
    dft_mpl_code = dft_bands._exportcontent(fileformat='mpl_singlefile', legend=f'{dft_bands.pk}', main_file_name='')[0]  # pylint: disable=protected-access
    wan_mpl_code = wan_bands._exportcontent(  # pylint: disable=protected-access
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
        replacement = f'\nfermi_energy =  {fermi_energy}\n'
        replacement += "p.axhline(y=fermi_energy, color='blue', linestyle='--', label='Fermi', zorder=-1)\n"
        replacement += 'pl.legend()\npl.show()\n'
        mpl_code = mpl_code.replace(b'pl.show()', replacement.encode())

    if save:
        if filename is None:
            filename = f'bandsdiff_{dft_bands.pk}_{wan_bands.pk}.py'
        with open(filename, 'w') as handle:
            handle.write(mpl_code.decode())

    return mpl_code


def get_mpl_code_for_workchains(workchain0, workchain1, title=None, save=False, filename=None):
    """Return matplotlib code for comparing band structures of two workchains."""

    def get_output_bands(workchain):
        if workchain.process_class == PwBaseWorkChain:
            return workchain.outputs.output_band
        if workchain.process_class == PwBandsWorkChain:
            return workchain.outputs.band_structure
        if workchain.process_class == Wannier90BandsWorkChain:
            return workchain.outputs.band_structure
        if workchain.process_class == Wannier90Calculation:
            return workchain.outputs.interpolated_bands
        raise ValueError(f'Unrecognized workchain type: {workchain}')

    # assume workchain0 is pw, workchain1 is wannier
    dft_bands = get_output_bands(workchain0)
    wan_bands = get_output_bands(workchain1)

    formula = workchain1.inputs.structure.get_formula()
    if title is None:
        title = f'{formula}, {workchain0.process_label}<{workchain0.pk}> bands<{dft_bands.pk}>, '
        title += f'{workchain1.process_label}<{workchain1.pk}> bands<{wan_bands.pk}>'

    if save and (filename is None):
        filename = f'bandsdiff_{formula}_{workchain0.pk}_{workchain1.pk}.py'

    if workchain1.process_class == Wannier90BandsWorkChain:
        fermi_energy = get_wannier_workchain_fermi_energy(workchain1)
    else:
        if workchain0.process_class == PwBandsWorkChain:
            fermi_energy = workchain0.outputs['scf_parameters']['fermi_energy']
        else:
            raise ValueError('Cannot find fermi energy')

    mpl_code = get_mpl_code_for_bands(dft_bands, wan_bands, fermi_energy, title, save, filename)

    return mpl_code


def get_wannier_workchain_fermi_energy(workchain: Wannier90BandsWorkChain) -> float:
    """Get Fermi energy of Wannier90BandsWorkChain.

    :param workchain: [description]
    :type workchain: Wannier90BandsWorkChain
    :return: [description]
    :rtype: float
    """
    from aiida_wannier90_workflows.utils.node import get_last_calcjob

    if 'scf' in workchain.outputs:
        fermi_energy = workchain.outputs['scf']['output_parameters']['fermi_energy']
    else:
        w90calc = get_last_calcjob(workchain.get_outgoing(link_label_filter='wannier90').one().node)
        if 'fermi_energy' in w90calc.inputs.parameters.get_dict():
            fermi_energy = w90calc.inputs.parameters.get_dict()['fermi_energy']
        else:
            raise ValueError('Cannot find fermi energy')

    return fermi_energy


def get_mapping_for_group(
    wan_group: ty.Union[str, orm.Group],
    dft_group: ty.Union[str, orm.Group],
    match_by_formula: bool = False
) -> ty.Dict[orm.Node, orm.Node]:
    """Find the corresponding DFT workchain for each Wannier workchain.

    :param wan_group: group label of ``Wannier90BandsWorkChain``
    :type wan_group: str
    :param dft_group: group label of ``PwBandsWorkChain``
    :type dft_group: str
    :param match_by_formula: match by structure formula or structure node itself, defaults to False
    :type match_by_formula: bool, optional
    :return: A dict with ``Wannier90BandsWorkChain`` as key and the corresponding ``PwBandsWorkChain`` as
    value. If not found the value is ``None``.
    :rtype: dict
    """
    if isinstance(wan_group, str):
        wan_group = orm.load_group(wan_group)
    if isinstance(dft_group, str):
        dft_group = orm.load_group(dft_group)

    if len(dft_group.nodes) == 0:
        print(f'DFT group<{dft_group.pk}> is empty')
        return None

    if len(wan_group.nodes) == 0:
        print(f'Wannier group<{wan_group.pk}> is empty')
        return None

    if 'structure' in dft_group.nodes[0].inputs:
        dft_structures = {_.inputs.structure: _ for _ in dft_group.nodes}
    elif 'structure' in dft_group.nodes[0].inputs['pw']:
        dft_structures = {_.inputs.pw.structure: _ for _ in dft_group.nodes}
    if match_by_formula:
        dft_structures = {k.get_formula(): v for k, v in dft_structures.items()}
    # print(f'Found DFT calculations: {dft_structures}')

    mapping = {}
    for wan_wc in wan_group.nodes:
        structure = wan_wc.inputs.structure
        formula = structure.get_formula()

        try:
            if match_by_formula:
                dft_wc = dft_structures[formula]
            else:
                dft_wc = dft_structures[structure]
            mapping[wan_wc] = dft_wc
        except KeyError:
            mapping[wan_wc] = None

    return mapping


def export_bands_for_group(
    wan_group: ty.Union[str, orm.Group],
    dft_group: ty.Union[str, orm.Group],
    save_dir: str,
    match_by_formula: bool = False
):
    """Export matplotlib code for comparing DFT and Wannier bands in two groups.

    :param wan_group: [description]
    :type wan_group: ty.Union[str, orm.Group]
    :param dft_group: [description]
    :type dft_group: ty.Union[str, orm.Group]
    :param save_dir: [description]
    :type save_dir: str
    :param match_by_formula: [description], defaults to False
    :type match_by_formula: bool, optional
    """
    import os.path

    if isinstance(wan_group, str):
        wan_group = orm.load_group(wan_group)
    if isinstance(dft_group, str):
        dft_group = orm.load_group(dft_group)

    mapping = get_mapping_for_group(wan_group, dft_group, match_by_formula)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    os.chdir(save_dir)
    print(f'files are saved in {save_dir}')

    for wan_wc in wan_group.nodes:
        if not wan_wc.is_finished_ok:
            print(f'! Skip unfinished {wan_wc.process_label}<{wan_wc.pk}> of {formula}')
            continue

        dft_wc = mapping[wan_wc]
        if dft_wc is None:
            msg = f'! Cannot find DFT bands for {wan_wc.process_label}<{wan_wc.pk}> of {formula}'
            print(msg)
            continue

        if not dft_wc.is_finished_ok:
            print(f'! Skip unfinished DFT {dft_wc.process_label}<{dft_wc.pk}> of {formula}')
            continue
        dft_bands = dft_wc.outputs.output_band

        formula = wan_wc.inputs.structure.get_formula()
        filename = f'bandsdiff_{formula}_{wan_wc.pk}.py'

        if wan_wc.process_class == Wannier90Calculation:
            fermi_energy = wan_wc.inputs.parameters['fermi_energy']
            w90_bands = wan_wc.outputs.band_structure
            get_mpl_code_for_bands(dft_bands, w90_bands, fermi_energy=fermi_energy, save=True, filename=filename)
        else:
            get_mpl_code_for_workchains(dft_wc, wan_wc, save=True, filename=filename)


def bands_py_to_png(py_dir: str, png_dir: str):
    """Convert ``bandsdiff_*.py`` files generated by ``export_bands_for_group`` to png files.

    :param py_dir: directory of ``*.py`` files
    :type py_dir: str
    :param png_dir: directory to save png files
    :type png_dir: str
    """
    import glob

    py_pattern = 'bandsdiff_*.py'
    print(f'Searching {py_pattern} in {py_dir}, save png in {png_dir}')

    globbed = glob.glob(f'{py_dir}/{py_pattern}')
    for filename in globbed:
        with open(filename) as handle:
            mplcode = ''.join(handle.readlines())
            mplcode = mplcode.replace('fig = pl.figure()', 'fig = pl.figure(figsize=(16,10))')
            png_filename = filename.removesuffix('.py') + '.png'
            print(f'{py_dir}/{filename} -> {png_dir}/{png_filename}')
            mplcode = mplcode.replace('pl.show()', f"pl.savefig('{png_dir}/{png_filename}', bbox_inches='tight')")
            exec(mplcode)  # pylint: disable=exec-used
