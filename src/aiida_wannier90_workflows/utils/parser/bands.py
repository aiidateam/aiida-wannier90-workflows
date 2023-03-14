#!/usr/bin/env python
"""Parse wannier90 bands dat files."""
from aiida import orm


def parse_w90_bands(band_dat, band_kpt, band_labelinfo, win):
    """Parse wannier90 seedname[_band.dat, _band.kpt, _band.labelinfo.dat, .win] files.

    :param band_dat: [description]
    :type band_dat: [type]
    :param band_kpt: [description]
    :type band_kpt: [type]
    :param band_labelinfo: [description]
    :type band_labelinfo: [type]
    :param win: [description]
    :type win: [type]
    :return: [description]
    :rtype: [type]
    """
    import ase
    import w90utils  # pylint: disable=import-error

    from aiida_wannier90.parsers.wannier90 import band_parser

    lat = w90utils.io.win.read_dlv(win, units="angstrom")
    atoms = w90utils.io.win.read_atoms(win, units="crystal")
    symbols, pos = zip(*atoms)
    ase_struct = ase.Atoms(symbols=symbols, scaled_positions=pos, cell=lat, pbc=True)
    struct = orm.StructureData(ase=ase_struct)

    with open(band_dat, encoding="utf-8") as handle:
        band_dat_lines = handle.readlines()
    with open(band_kpt, encoding="utf-8") as handle:
        band_kpt_lines = handle.readlines()
    with open(band_labelinfo, encoding="utf-8") as handle:
        band_labelinfo_lines = handle.readlines()

    bands, warnings = band_parser(
        band_dat_lines, band_kpt_lines, band_labelinfo_lines, struct
    )
    if len(warnings) != 0:
        print(warnings)
    return bands


def parse_w90_bands_by_seedname(seedname):
    """Parse wannier90 bands data files with a given seedname."""
    return parse_w90_bands(
        f"{seedname}_band.dat",
        f"{seedname}_band.kpt",
        f"{seedname}_band.labelinfo.dat",
        f"{seedname}.win",
    )
