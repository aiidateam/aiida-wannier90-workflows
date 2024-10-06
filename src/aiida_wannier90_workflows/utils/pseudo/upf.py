"""Utility functions for parsing pseudo potential file."""

import xml.etree.ElementTree as ET

from aiida import orm

__all__ = (  # for the content of UPF, i.e. these functions accept str as parameter
    "parse_zvalence",
    "parse_pswfc_nosoc",
    "parse_pswfc_soc",
    "parse_number_of_pswfc",
    # for orm.UpfData, i.e. these functions accept orm.UpfData as parameter
    "get_number_of_electrons_from_upf",
    "get_projections_from_upf",
    "get_number_of_projections_from_upf",
    # for orm.StructreData, i.e. these functions accept orm.StructreData as parameter
    # 'get_number_of_electrons',
    # 'get_projections',
    # 'get_number_of_projections',
    # 'get_wannier_number_of_bands',
    # helper functions
    "is_soc_pseudo",
    # 'load_pseudo_metadata'
)


def get_ppheader(upf_content: str) -> str:
    """Get PP_HEADER."""
    upf_content = upf_content.split("\n")
    # get PP_HEADER block
    ppheader_block = ""
    found_begin = False
    found_end = False
    for line in upf_content:
        if "<PP_HEADER" in line:
            ppheader_block += line + "\n"
            if not found_begin:
                found_begin = True
                if "/>" in line or "</PP_HEADER>" in line:
                    # in the same line
                    break
                continue
        if found_begin and ("/>" in line or "</PP_HEADER>" in line):
            ppheader_block += line + "\n"
            if not found_end:
                found_end = True
                break
        if found_begin:
            ppheader_block += line + "\n"
    # print(ppheader_block)
    return ppheader_block


def is_soc_pseudo(upf_content: str) -> bool:
    """Check if it is a SOC pseudo.

    :param upf_content: the content of the UPF file
    :type upf_content: str
    :return: [description]
    :rtype: bool
    """
    ppheader_block = get_ppheader(upf_content)
    # parse XML
    PP_HEADER = ET.XML(ppheader_block)  # pylint: disable=invalid-name
    if len(PP_HEADER.attrib) == 0:
        # old upf format, TODO check how to retrieve has_so of old upf format
        has_so = False
    else:
        # upf format 2.0.1
        has_so = PP_HEADER.get("has_so")[0].lower() == "t"
    return has_so


def parse_zvalence(upf_content: str) -> float:
    """Get z_valcence from a UPF file.

    No AiiDA dependencies.
    Works for both UPF v1 & v2 format, non-relativistic & relativistic.
    Tested on all the SSSP pseudos.

    :param upf_content: the content of the UPF file
    :type upf_content: str
    :return: z_valence of the UPF file
    :rtype: float
    """
    ppheader_block = get_ppheader(upf_content)
    num_electrons = 0
    # parse XML
    PP_HEADER = ET.XML(ppheader_block)  # pylint: disable=invalid-name
    if len(PP_HEADER.attrib) == 0:
        # old upf format, at the 6th line, e.g.
        # <PP_HEADER>
        #    0                   Version Number
        #   Be                   Element
        #    US                  Ultrasoft pseudopotential
        #     F                  Nonlinear Core Correction
        #  SLA  PW   PBX  PBC    PBE  Exchange-Correlation functional
        #     4.00000000000      Z valence
        #   -27.97245939710      Total energy
        #     0.00000    0.00000 Suggested cutoff for wfc and rho
        #     2                  Max angular momentum component
        #   769                  Number of points in mesh
        #     3    6             Number of Wavefunctions, Number of Projectors
        #  Wavefunctions         nl  l   occ
        #                        1S  0  2.00
        #                        2S  0  2.00
        #                        2P  1  0.00
        # </PP_HEADER>
        lines = ppheader_block.split("\n")[6]
        # some may have z_valence="1.300000000000000E+001", str -> float
        num_electrons = float(lines.strip().split()[0])
    else:
        # upf format 2.0.1, e.g.
        # <PP_HEADER
        #    generated="Generated using ONCVPSP code by D. R. Hamann"
        #    author="anonymous"
        #    date="180627"
        #    comment=""
        #    element="Ag"
        #    pseudo_type="NC"
        #    relativistic="scalar"
        #    is_ultrasoft="F"
        #    is_paw="F"
        #    is_coulomb="F"
        #    has_so="F"
        #    has_wfc="F"
        #    has_gipaw="F"
        #    core_correction="F"
        #    functional="PBE"
        #    z_valence="   19.00"
        #    total_psenergy="  -2.86827035760E+02"
        #    rho_cutoff="   1.39700000000E+01"
        #    l_max="2"
        #    l_local="-1"
        #    mesh_size="  1398"
        #    number_of_wfc="4"
        #    number_of_proj="6"/>
        num_electrons = float(PP_HEADER.get("z_valence"))
    return num_electrons


def get_upf_content(upf: orm.UpfData) -> str:
    """Retreive the content of the UpfData.

    :param upf: [description]
    :type upf: orm.UpfData
    :return: [description]
    :rtype: str
    """
    import aiida_pseudo.data.pseudo.upf

    if not isinstance(upf, (orm.UpfData, aiida_pseudo.data.pseudo.upf.UpfData)):
        raise ValueError(
            f"The type of upf is {type(upf)}, only aiida.orm.UpfData is accepted"
        )
    upf_name = upf.base.repository.list_object_names()[0]
    upf_content = upf.base.repository.get_object_content(upf_name)
    return upf_content


def get_number_of_electrons_from_upf(upf: orm.UpfData) -> float:
    """Get number of electrons.

    Wrapper for `parse_zvalence`.

    :param upf: pseudo
    :type upf: aiida.orm.UpfData
    :return: number of electrons
    :rtype: float
    """
    upf_content = get_upf_content(upf)
    return parse_zvalence(upf_content)


def parse_pswfc_soc(upf_content: str) -> list:
    """Parse the PP_SPIN_ORB block in SOC UPF.

    This is also the orbitals used for projections in projwfc.x.
    No AiiDA dependencies.
    Works for both UPF v1 & v2 format.

    :param upf_content: [description]
    :type upf_content: str
    :raises ValueError: [description]
    :return: list of dict, each dict contains 3 keys for quantum number n, l, j
    :rtype: list
    """
    if not is_soc_pseudo(upf_content):
        raise ValueError("Only accept SOC pseudo")
    upf_content = upf_content.split("\n")
    # get PP_SPIN_ORB block
    pswfc_block = ""
    found_begin = False
    found_end = False
    for line in upf_content:
        if "PP_SPIN_ORB" in line:
            pswfc_block += line + "\n"
            if not found_begin:
                found_begin = True
                continue
            if not found_end:
                found_end = True
                break
        if found_begin:
            pswfc_block += line + "\n"

    # contains element: {'n', 'l', 'j'} for 3 quantum numbers
    projections = []
    # parse XML
    PP_PSWFC = ET.XML(pswfc_block)  # pylint: disable=invalid-name
    if len(list(PP_PSWFC)) == 0:
        # old upf format, TODO check
        raise ValueError

    # upf format 2.0.1, see q-e/Modules/uspp.f90:n_atom_wfc
    for child in PP_PSWFC:
        if not "PP_RELWFC" in child.tag:
            continue
        nn = int(child.get("nn"))  # pylint: disable=invalid-name
        # use int, otherwise the returned num_projections is float
        lchi = int(child.get("lchi"))
        jchi = float(child.get("jchi"))
        oc = child.get("oc")  # pylint: disable=invalid-name
        # pslibrary PP has 'oc' attribute, but pseudodojo does not have 'oc'
        # e.g. in pslibrary Ag.rel-pbe-n-kjpaw_psl.1.0.0.UPF
        # <PP_RELWFC.1 index="1" els="5S" nn="1" lchi="0" jchi="5.000000000000e-1" oc="1.500000000000e0"/>
        # in pseudo-dojo Ag.upf
        # <PP_RELWFC.1  index="1"  lchi="0" jchi="0.5" nn="1"/>
        if oc is not None:
            oc = float(oc)  # pylint: disable=invalid-name
            if oc < 0:
                continue
        projections.append({"n": nn, "l": lchi, "j": jchi})

    return projections


def parse_pswfc_nosoc(upf_content: str) -> list:
    """Parse pswfc for non-SOC pseudo.

    :param upf_content: [description]
    :type upf_content: str
    :return: list of dict, each dict contains 1 key for quantum number l
    :rtype: list
    """
    if is_soc_pseudo(upf_content):
        raise ValueError("Only accept non-SOC pseudo")
    upf_content = upf_content.split("\n")
    # get PP_PSWFC block
    pswfc_block = ""
    found_begin = False
    found_end = False
    for line in upf_content:
        if "PP_PSWFC" in line:
            pswfc_block += line + "\n"
            if not found_begin:
                found_begin = True
                continue
            if not found_end:
                found_end = True
                break
        if found_begin:
            pswfc_block += line + "\n"

    projections = []
    # parse XML
    PP_PSWFC = ET.XML(pswfc_block)  # pylint: disable=invalid-name
    if len(list(PP_PSWFC)) == 0:
        # old upf format
        import re

        r = re.compile(r"[\d]([SPDF])")  # pylint: disable=invalid-name
        spdf = r.findall(PP_PSWFC.text)
        for orbit in spdf:
            orbit = orbit.lower()
            if orbit == "s":
                l = 0
            elif orbit == "p":
                l = 1
            elif orbit == "d":
                l = 2
            elif orbit == "f":
                l = 3
            projections.append({"l": l})
    else:
        # upf format 2.0.1
        for child in PP_PSWFC:
            l = int(child.get("l"))
            projections.append({"l": l})
    return projections


def parse_pswfc_energy_nosoc(upf_content: str) -> list:
    """Parse pswfc pseudo_energy for non-SOC pseudo.

    :param upf_content: [description]
    :type upf_content: str
    :return: list of dict, each dict contains 1 key for quantum number l
    :rtype: list
    """
    if is_soc_pseudo(upf_content):
        raise ValueError("Only accept non-SOC pseudo")
    upf_content = upf_content.split("\n")
    # get PP_PSWFC block
    pswfc_block = ""
    found_begin = False
    found_end = False
    for line in upf_content:
        if "PP_PSWFC" in line:
            pswfc_block += line + "\n"
            if not found_begin:
                found_begin = True
                continue
            if not found_end:
                found_end = True
                break
        if found_begin:
            pswfc_block += line + "\n"

    projections = []
    # parse XML
    PP_PSWFC = ET.XML(pswfc_block)  # pylint: disable=invalid-name
    if len(list(PP_PSWFC)) == 0:
        raise NotImplementedError
        #  pylint: disable=unreachable
        # old upf format
        import re

        r = re.compile(r"[\d]([SPDF])")  # pylint: disable=invalid-name
        spdf = r.findall(PP_PSWFC.text)
        for orbit in spdf:
            orbit = orbit.lower()
            if orbit == "s":
                l = 0
            elif orbit == "p":
                l = 1
            elif orbit == "d":
                l = 2
            elif orbit == "f":
                l = 3
            projections.append({"l": l})

    # upf format 2.0.1
    for child in PP_PSWFC:
        pseudo_energy = float(child.get("pseudo_energy"))
        label = str(child.get("label"))
        projections.append({"pseudo_energy": pseudo_energy, "label": label})

    return projections


def get_projections_from_upf(upf: orm.UpfData):
    """Return a list of strings for Wannier90 projection block.

    :param upf: the pseduo to be parsed
    :type upf: orm.UpfData
    :return: list of projections
    :rtype: list
    """

    class Orbit:
        """A simple class to help sorting/removing the orbitals in a list."""

        def __init__(self, orbit_dict):
            self.n = orbit_dict["n"]
            self.l = orbit_dict["l"]
            self.j = orbit_dict["j"]

        def __eq__(self, orbit):
            return (
                self.n == orbit.n and self.l == orbit.l and abs(self.j - orbit.j) < 1e-6
            )

        def __lt__(self, orbit):
            if self.n < orbit.n:
                return True
            if self.n == orbit.n:
                if self.l < orbit.l:
                    return True
                if self.l == orbit.l:
                    return self.j < orbit.j
                return False
            return False

    orbit_map = {0: "s", 1: "p", 2: "d", 3: "f"}
    upf_content = get_upf_content(upf)
    wannier_projections = []
    has_so = is_soc_pseudo(upf_content)
    if not has_so:
        pswfc = parse_pswfc_nosoc(upf_content)
        for wfc in pswfc:
            wannier_projections.append(f'{upf.element}: {orbit_map[wfc["l"]]}')
    else:
        pswfc = []
        for wfc in parse_pswfc_soc(upf_content):
            pswfc.append(Orbit(wfc))
        # First sort by n, then l, then j, in ascending order
        sorted_pswfc = sorted(pswfc)  # will use __lt__
        # Check that for a given l (>0), there are two j: j = l - 1/2 and j = l + 1/2,
        # then we can combine these two j and form a projection orbital for wannier90.
        # e.g. {n = 1, l = 1, j = 0.5} and {n = 1, l = 1, j = 1.5} together correspond
        # to a p orbital in wannier projection block.
        # pylint: disable=unnecessary-lambda-assignment
        is_equal = lambda x, y: abs(x - y) < 1e-6
        i = 0
        while i < len(sorted_pswfc):
            wfc = sorted_pswfc[i]
            n = wfc.n
            l = wfc.l
            j = wfc.j
            if l == 0:
                assert is_equal(j, 0.5)
            else:
                pair_orbit = Orbit({"n": n, "l": l, "j": j + 1})
                assert i + 1 < len(sorted_pswfc)
                assert sorted_pswfc[i + 1] == pair_orbit  # will use __eq__
                pswfc.remove(
                    pair_orbit
                )  # remove uses __eq__, and remove the 1st matched element
                assert (
                    pair_orbit not in pswfc
                )  # in uses __eq__, pswfc should contain one and only one pair_orbit
                i += 1  # skip next one
            i += 1
        # Now all the j = l + 1/2 orbitals have been removed
        for wfc in pswfc:
            wannier_projections.append(f"{upf.element}: {orbit_map[wfc.l]}")
    return wannier_projections


def parse_number_of_pswfc(upf_content: str) -> int:
    """Get the number of orbitals in the UPF file.

    This is also the number of orbitals used for projections in projwfc.x.
    No AiiDA dependencies.
    Works for both UPF v1 & v2 format, non-relativistic & relativistic.
    Tested on all the SSSP pseudos.

    :param upf_content: the content of the UPF file
    :type upf_content: str
    :return: number of PSWFC
    :rtype: int
    """
    num_projections = 0
    has_so = is_soc_pseudo(upf_content)
    if not has_so:
        pswfc = parse_pswfc_nosoc(upf_content)
        for wfc in pswfc:
            l = wfc["l"]
            num_projections += 2 * l + 1
    else:
        pswfc = parse_pswfc_soc(upf_content)
        # For a given quantum number l, there are 2 cases:
        # 1. j = l - 1/2 then there are 2*j + 1 = 2l states
        # 2. j = l + 1/2 then there are 2*j + 1 = 2l + 2 states so we have to add another 2
        # This follows the logic in q-e/Modules/uspp.f90:n_atom_wfc
        for wfc in pswfc:
            l = wfc["l"]
            j = wfc["j"]
            num_projections += 2 * l
            if abs(j - l - 0.5) < 1e-6:
                num_projections += 2
    return num_projections


def get_number_of_projections_from_upf(upf: orm.UpfData) -> int:
    """Aiida wrapper for `parse_number_of_pswfc`.

    :param upf: the UPF file
    :type upf: aiida.orm.UpfData
    :return: number of projections in the UPF file
    :rtype: int
    """
    upf_content = get_upf_content(upf)
    return parse_number_of_pswfc(upf_content)
