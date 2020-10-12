import typing
import os
import json
from aiida import orm
import xml.etree.ElementTree as ET
import hashlib

__all__ = (# for the content of UPF, i.e. these functions accept str as parameter
           'parse_zvalence',
           'parse_pswfc_nosoc', 'parse_pswfc_soc',
           'parse_number_of_pswfc',
           # for orm.UpfData, i.e. these functions accept orm.UpfData as parameter
           'get_number_of_electrons_from_upf',
           'get_projections_from_upf',
           'get_number_of_projections_from_upf', 
           # for orm.StructreData, i.e. these functions accept orm.StructreData as parameter
           'get_number_of_electrons',
           'get_projections',
           'get_number_of_projections',
           'get_wannier_number_of_bands',
           # helper functions
           'is_soc_pseudo',
           '_load_pseudo_metadata')

Dict_of_Upf = typing.Dict[str, orm.UpfData]

def get_ppheader(upf_content: str) -> str:
    upf_content = upf_content.split('\n')
    # get PP_HEADER block
    ppheader_block = ''
    found_begin = False
    found_end = False
    for line in upf_content:
        if '<PP_HEADER' in line:
            ppheader_block += line + '\n'
            if not found_begin:
                found_begin = True
                if '/>' in line or '</PP_HEADER>' in line:
                    # in the same line
                    break
                else:
                    continue
        if found_begin and ('/>' in line or '</PP_HEADER>' in line):
            ppheader_block += line + '\n'
            if not found_end:
                found_end = True
                break
        if found_begin:
            ppheader_block += line + '\n'
    # print(ppheader_block)
    return ppheader_block

def is_soc_pseudo(upf_content: str) -> bool:
    """check if it is a SOC pseudo

    :param upf_content: the content of the UPF file
    :type upf_content: str
    :return: [description]
    :rtype: bool
    """
    ppheader_block = get_ppheader(upf_content)
    # parse XML
    PP_HEADER = ET.XML(ppheader_block)
    if len(PP_HEADER.attrib) == 0:
        # old upf format, TODO check how to retrieve has_so of old upf format
        has_so = False
    else:
        # upf format 2.0.1
        has_so = PP_HEADER.get('has_so')[0].lower() == 't'
    return has_so

def parse_zvalence(upf_content: str) -> float:
    """get z_valcence from a UPF file. No AiiDA dependencies.
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
    PP_HEADER = ET.XML(ppheader_block)
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
        lines = ppheader_block.split('\n')[6]
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
        num_electrons = float(PP_HEADER.get('z_valence'))
    return num_electrons

def get_upf_content(upf: orm.UpfData) -> str:
    """Retreive the content of the UpfData

    :param upf: [description]
    :type upf: orm.UpfData
    :return: [description]
    :rtype: str
    """
    if not isinstance(upf, orm.UpfData):
        raise ValueError(f'The type of upf is {type(upf)}, only aiida.orm.UpfData is accepted')
    upf_name = upf.list_object_names()[0]
    upf_content = upf.get_object_content(upf_name)
    return upf_content

def get_number_of_electrons_from_upf(upf: orm.UpfData) -> float:
    """AiiDA wrapper for `parse_zvalence'

    :param upf: pseudo
    :type upf: aiida.orm.UpfData
    :return: number of electrons
    :rtype: float
    """
    upf_content = get_upf_content(upf)
    return parse_zvalence(upf_content)

def get_number_of_electrons(structure: orm.StructureData, pseudos: Dict_of_Upf) -> float:
    """get number of electrons for the structure based on pseudopotentials

    Usage:
        nprojs = get_number_of_electrons(struct_MgO, {'Mg':UpfData_Mg, 'O':UpfData_O})

    :param structure: crystal structure
    :type structure: aiida.orm.StructureData
    :param pseudos: a dictionary contains orm.UpfData of the structure
    :type pseudos: dict
    :return: number of electrons
    :rtype: float
    """
    if not isinstance(structure, orm.StructureData):
        raise ValueError(f'The type of structure is {type(structure)}, only aiida.orm.StructureData is accepted')
    if not isinstance(pseudos, dict):
        raise ValueError(f'The type of pseudos is {type(pseudos)}, only dict is accepted')
    for k, v in pseudos.items():
        if not isinstance(k, str) or not isinstance(v, orm.UpfData):
            raise ValueError(f'The type of <{k}, {v}> in pseudos is <{type(k)}, {type(v)}>, only <str, aiida.orm.UpfData> type is accepted')

    tot_nelecs = 0
    # e.g. composition = {'Ga': 1, 'As': 1}
    composition = structure.get_composition()
    for kind in composition:
        upf = pseudos[kind]
        nelecs = get_number_of_electrons_from_upf(upf)
        tot_nelecs += nelecs * composition[kind]
    return tot_nelecs

def parse_pswfc_soc(upf_content: str) -> list:
    """parse the PP_SPIN_ORB block in SOC UPF.
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
        raise ValueError('Only accept SOC pseudo')
    upf_content = upf_content.split('\n')
    # get PP_SPIN_ORB block
    pswfc_block = ''
    found_begin = False
    found_end = False
    for line in upf_content:
        if 'PP_SPIN_ORB' in line:
            pswfc_block += line + '\n'
            if not found_begin:
                found_begin = True
                continue
            else:
                if not found_end:
                    found_end = True
                    break
        if found_begin:
            pswfc_block += line + '\n'

    # contains element: {'n', 'l', 'j'} for 3 quantum numbers
    projections = []
    # parse XML
    PP_PSWFC = ET.XML(pswfc_block)
    if len(PP_PSWFC.getchildren()) == 0:
        # old upf format, TODO check
        raise ValueError
    else:
        # upf format 2.0.1, see q-e/Modules/uspp.f90:n_atom_wfc
        for child in PP_PSWFC:
            if not 'PP_RELWFC' in child.tag:
                continue
            nn = int(child.get('nn'))
            # use int, otherwise the returned num_projections is float
            lchi = int(child.get('lchi'))
            jchi = float(child.get('jchi'))
            oc = child.get('oc')
            # pslibrary PP has 'oc' attribute, but pseudodojo does not have 'oc'
            # e.g. in pslibrary Ag.rel-pbe-n-kjpaw_psl.1.0.0.UPF
            # <PP_RELWFC.1 index="1" els="5S" nn="1" lchi="0" jchi="5.000000000000e-1" oc="1.500000000000e0"/>
            # in pseudo-dojo Ag.upf
            # <PP_RELWFC.1  index="1"  lchi="0" jchi="0.5" nn="1"/>
            if oc is not None:
                oc = float(oc)
                if oc < 0:
                    continue
            projections.append({'n': nn, 'l': lchi, 'j': jchi})
    return projections

def parse_pswfc_nosoc(upf_content: str) -> list:
    """for non-relativistic pseudo

    :param upf_content: [description]
    :type upf_content: str
    :return: list of dict, each dict contains 1 key for quantum number l
    :rtype: list
    """
    if is_soc_pseudo(upf_content):
        raise ValueError('Only accept non-SOC pseudo')
    upf_content = upf_content.split('\n')
    # get PP_PSWFC block
    pswfc_block = ''
    found_begin = False
    found_end = False
    for line in upf_content:
        if 'PP_PSWFC' in line:
            pswfc_block += line + '\n'
            if not found_begin:
                found_begin = True
                continue
            else:
                if not found_end:
                    found_end = True
                    break
        if found_begin:
            pswfc_block += line + '\n'

    projections = []
    # parse XML
    PP_PSWFC = ET.XML(pswfc_block)
    if len(PP_PSWFC.getchildren()) == 0:
        # old upf format
        import re
        r = re.compile(r'[\d]([SPDF])')
        spdf = r.findall(PP_PSWFC.text)
        for orbit in spdf:
            orbit = orbit.lower()
            if orbit == 's':
                l = 0
            elif orbit == 'p':
                l = 1
            elif orbit == 'd':
                l = 2
            elif orbit == 'f':
                l = 3
            projections.append({'l': l})
    else:
        # upf format 2.0.1
        for child in PP_PSWFC:
            l = int(child.get('l'))
            projections.append({'l': l})
    return projections

def get_projections_from_upf(upf: orm.UpfData):
    """Return a list of strings for Wannier90 projection block

    :param upf: the pseduo to be parsed
    :type upf: orm.UpfData
    :return: list of projections
    :rtype: list
    """
    class Orbit:
        """A simple class to help sorting/removing the orbitals in a list
        """
        def __init__(self, orbit_dict):
            self.n = orbit_dict['n']
            self.l = orbit_dict['l']
            self.j = orbit_dict['j']

        def __eq__(self, orbit):
            return self.n == orbit.n and self.l == orbit.l and abs(self.j - orbit.j) < 1e-6

        def __lt__(self, orbit):
            if self.n < orbit.n:
                return True
            elif self.n == orbit.n:
                if self.l < orbit.l:
                    return True
                elif self.l == orbit.l:
                    return self.j < orbit.j
                else:
                    return False
            else:
                return False

    orbit_map = {0: 's', 1: 'p', 2: 'd', 3: 'f'}
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
        sorted_pswfc = sorted(pswfc) # will use __lt__
        # Check that for a given l (>0), there are two j: j = l - 1/2 and j = l + 1/2,
        # then we can combine these two j and form a projection orbital for wannier90.
        # e.g. {n = 1, l = 1, j = 0.5} and {n = 1, l = 1, j = 1.5} together correspond
        # to a p orbital in wannier projection block.
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
                pair_orbit = Orbit({'n': n, 'l': l, 'j': j+1})
                assert i+1 < len(sorted_pswfc)
                assert sorted_pswfc[i+1] == pair_orbit # will use __eq__
                pswfc.remove(pair_orbit) # remove uses __eq__, and remove the 1st matched element
                assert not pair_orbit in pswfc # # in uses __eq__, pswfc should contain one and only one pair_orbit
                i += 1 # skip next one
            i += 1
        # Now all the j = l + 1/2 orbitals have been removed
        for wfc in pswfc:
            wannier_projections.append(f'{upf.element}: {orbit_map[wfc.l]}')
    return wannier_projections

def get_projections(structure: orm.StructureData, pseudos: Dict_of_Upf):
    """get wannier90 projection block for the crystal structure 
    based on pseudopotential files.

    Usage:
        projs = get_projections(struct_MgO, {'Mg':UpfData_Mg, 'O':UpfData_O})

    :param structure: crystal structure
    :type structure: aiida.orm.StructureData
    :param pseudos: a dictionary contains orm.UpfData of the structure
    :type pseudos: dict
    :return: wannier90 projection block
    :rtype: list
    """
    if not isinstance(structure, orm.StructureData):
        raise ValueError(f'The type of structure is {type(structure)}, only aiida.orm.StructureData is accepted')
    if not isinstance(pseudos, dict):
        raise ValueError(f'The type of pseudos is {type(pseudos)}, only dict is accepted')
    for k, v in pseudos.items():
        if not isinstance(k, str) or not isinstance(v, orm.UpfData):
            raise ValueError(f'The type of <{k}, {v}> in pseudos is <{type(k)}, {type(v)}>, only <str, aiida.orm.UpfData> type is accepted')

    projections = []
    # e.g. composition = {'Ga': 1, 'As': 1}
    composition = structure.get_composition()
    for kind in composition:
        upf = pseudos[kind]
        projs = get_projections_from_upf(upf)
        projections.extend(projs)
    return projections

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
            l = wfc['l']
            num_projections += 2 * l + 1
    else:
        pswfc = parse_pswfc_soc(upf_content)
        # For a given quantum number l, there are 2 cases:
        # 1. j = l - 1/2 then there are 2*j + 1 = 2l states
        # 2. j = l + 1/2 then there are 2*j + 1 = 2l + 2 states so we have to add another 2
        # This follows the logic in q-e/Modules/uspp.f90:n_atom_wfc
        for wfc in pswfc:
            l = wfc['l']
            j = wfc['j']
            num_projections += 2 * l
            if abs(j - l - 0.5) < 1e-6:
                num_projections += 2
    return num_projections

def get_number_of_projections_from_upf(upf: orm.UpfData) -> int:
    """aiida wrapper for `parse_number_of_pswfc`.

    :param upf: the UPF file
    :type upf: aiida.orm.UpfData
    :return: number of projections in the UPF file
    :rtype: int
    """
    upf_content = get_upf_content(upf)
    return parse_number_of_pswfc(upf_content)

def get_number_of_projections(structure: orm.StructureData, pseudos: Dict_of_Upf) -> int:
    """get number of projections for the crystal structure 
    based on pseudopotential files.

    Usage:
        nprojs = get_number_of_projections(struct_MgO, {'Mg':UpfData_Mg, 'O':UpfData_O})

    :param structure: crystal structure
    :type structure: aiida.orm.StructureData
    :param pseudos: a dictionary contains orm.UpfData of the structure
    :type pseudos: dict
    :return: number of projections
    :rtype: int
    """
    if not isinstance(structure, orm.StructureData):
        raise ValueError(f'The type of structure is {type(structure)}, only aiida.orm.StructureData is accepted')
    if not isinstance(pseudos, dict):
        raise ValueError(f'The type of pseudos is {type(pseudos)}, only dict is accepted')
    for k, v in pseudos.items():
        if not isinstance(k, str) or not isinstance(v, orm.UpfData):
            raise ValueError(f'The type of <{k}, {v}> in pseudos is <{type(k)}, {type(v)}>, only <str, aiida.orm.UpfData> type is accepted')

    tot_nprojs = 0
    # e.g. composition = {'Ga': 1, 'As': 1}
    composition = structure.get_composition()
    for kind in composition:
        upf = pseudos[kind]
        nprojs = get_number_of_projections_from_upf(upf)
        tot_nprojs += nprojs * composition[kind]
    return tot_nprojs

def get_wannier_number_of_bands(structure, pseudos, only_valence=False, spin_polarized=False):
    """estimate number of bands for a Wannier90 calculation.

    :param structure: crystal structure
    :type structure: aiida.orm.StructureData
    :param pseudos: dictionary of pseudopotentials
    :type pseudos: dict of aiida.orm.UpfData
    :param only_valence: return only occupied number of badns
    :type only_valence: bool
    :param spin_polarized: magnetic calculation?
    :type spin_polarized: bool
    :return: number of bands for Wannier90 SCDM
    :rtype: int
    """
    num_electrons = get_number_of_electrons(structure, pseudos)
    num_projections = get_number_of_projections(structure, pseudos)
    nspin = 2 if spin_polarized else 1
    # TODO check nospin, spin, soc
    if only_valence:
        num_bands = int(0.5 * num_electrons * nspin)
    else:
        # nbands must > num_projections = num_wann
        factor = 1.2
        num_bands = max(int(0.5 * num_electrons * nspin * factor), 
                    int(0.5 * num_electrons * nspin + 4 * nspin), 
                    int(num_projections * factor), 
                    int(num_projections + 4))
    return num_bands

def _load_pseudo_metadata(filename):
    """Load from the current folder a json file containing metadata (incl. suggested cutoffs) for a library of pseudopotentials.
    """
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)) as handle:
        return json.load(handle)

def md5(filename):
    hash_md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_metadata(filename):
    result = {
        "filename": filename,
        "md5": md5(filename),
        "pseudopotential": "100PAW"
        }
    with open(filename) as f:
        for line in f:
            if "Suggested minimum cutoff for wavefunctions" in line:
                wave = float(line.strip().split()[-2])
            if "Suggested minimum cutoff for charge density" in line:
                charge = float(line.strip().split()[-2])
        result["cutoff"] = wave
        result["dual"] = charge / wave
    return result

def generate_pslibrary_metadata(dirname=None):
    """Scan the folder and generate a json file containing metainfo of pseudos of pslibrary.

    :param dirname: folder to be scanned, if None download from QE website
    :type dirname: str
    """
    import urllib.request

    output_filename = 'pslibrary_paw_relpbe_1.0.0.json'
    qe_site = 'https://www.quantum-espresso.org/upf_files/'

    # these are the suggested PP from https://dalcorso.github.io/pslibrary/PP_list.html (2020.07.21)
    suggested = r"""H:  H.$fct-*_psl.1.0.0 
He: He.$fct-*_psl.1.0.0    
Li: Li.$fct-sl-*_psl.1.0.0    
Be: Be.$fct-sl-*_psl.1.0.0    
B:  B.$fct-n-*_psl.1.0.0    
C:  C.$fct-n-*_psl.1.0.0    
N:  N.$fct-n-*_psl.1.0.0    
O:  O.$fct-n-*_psl.1.0.0    
F:  F.$fct-n-*_psl.1.0.0    
Ne: Ne.$fct-n-*_psl.1.0.0    
Na: Na.$fct-spnl-*_psl.1.0.0    
Mg: Mg.$fct-spnl-*_psl.1.0.0    
Al: Al.$fct-nl-*_psl.1.0.0    
Si: Si.$fct-nl-*_psl.1.0.0    
P:  P.$fct-nl-*_psl.1.0.0    
S:  S.$fct-nl-*_psl.1.0.0    
Cl: Cl.$fct-nl-*_psl.1.0.0    
Ar: Ar.$fct-nl-*_psl.1.0.0    
K:  K.$fct-spn-*_psl.1.0.0    
Ca: Ca.$fct-spn-*_psl.1.0.0    
Sc: Sc.$fct-spn-*_psl.1.0.0    
Ti: Ti.$fct-spn-*_psl.1.0.0    
V:  V.$fct-spnl-*_psl.1.0.0    
Cr: Cr.$fct-spn-*_psl.1.0.0    
Mn: Mn.$fct-spn-*_psl.0.3.1    
Fe: Fe.$fct-n-*_psl.1.0.0    
Co: Co.$fct-n-*_psl.0.3.1    
Ni: Ni.$fct-n-*_psl.1.0.0    
Cu: Cu.$fct-dn-*_psl.1.0.0    
Zn: Zn.$fct-dn-*_psl.1.0.0    
Ga: Ga.$fct-dnl-*_psl.1.0.0    
Ge: Ge.$fct-n-*_psl.1.0.0    
As: As.$fct-n-*_psl.1.0.0    
Se: Se.$fct-n-*_psl.1.0.0    
Br: Br.$fct-n-*_psl.1.0.0    
Kr: Kr.$fct-dn-*_psl.1.0.0    
Rb: Rb.$fct-spn-*_psl.1.0.0    
Sr: Sr.$fct-spn-*_psl.1.0.0 
Y:  Y.$fct-spn-*_psl.1.0.0    
Zr: Zr.$fct-spn-*_psl.1.0.0    
Nb: Nb.$fct-spn-*_psl.1.0.0    
Mo: Mo.$fct-spn-*_psl.1.0.0    
Tc: Tc.$fct-spn-*_psl.0.3.0    
Ru: Ru.$fct-spn-*_psl.1.0.0    
Rh: Rh.$fct-spn-*_psl.1.0.0    
Pd: Pd.$fct-n-*_psl.1.0.0    
Ag: Ag.$fct-n-*_psl.1.0.0    
Cd: Cd.$fct-n-*_psl.1.0.0 
In: In.$fct-dn-*_psl.1.0.0    
Sn: Sn.$fct-dn-*_psl.1.0.0    
Sb: Sb.$fct-n-*_psl.1.0.0    
Te: Te.$fct-n-*_psl.1.0.0    
I:  I.$fct-n-*_psl.1.0.0    
Xe: Xe.$fct-dn-*_psl.1.0.0    
Cs: Cs.$fct-spnl-*_psl.1.0.0    
Ba: Ba.$fct-spn-*_psl.1.0.0 
La: La.$fct-spfn-*_psl.1.0.0    
Ce: Ce.$fct-spdn-*_psl.1.0.0    
Pr: Pr.$fct-spdn-*_psl.1.0.0    
Nd: Nd.$fct-spdn-*_psl.1.0.0    
Pm: Pm.$fct-spdn-*_psl.1.0.0    
Sm: Sm.$fct-spdn-*_psl.1.0.0    
Eu: Eu.$fct-spn-*_psl.1.0.0 
Gd: Gd.$fct-spdn-*_psl.1.0.0    
Tb: Tb.$fct-spdn-*_psl.1.0.0    
Dy: Dy.$fct-spdn-*_psl.1.0.0    
Ho: Ho.$fct-spdn-*_psl.1.0.0    
Er: Er.$fct-spdn-*_psl.1.0.0    
Tm: Tm.$fct-spdn-*_psl.1.0.0    
Yb: Yb.$fct-spn-*_psl.1.0.0    
Lu: Lu.$fct-spdn-*_psl.1.0.0 
Hf: Hf.$fct-spn-*_psl.1.0.0    
Ta: Ta.$fct-spn-*_psl.1.0.0    
W:  W.$fct-spn-*_psl.1.0.0    
Re: Re.$fct-spn-*_psl.1.0.0    
Os: Os.$fct-spn-*_psl.1.0.0    
Ir: Ir.$fct-n-*_psl.0.2.3    
Pt: Pt.$fct-n-*_psl.1.0.0    
Au: Au.$fct-n-*_psl.1.0.1    
Hg: Hg.$fct-n-*_psl.1.0.0 
Tl: Tl.$fct-dn-*_psl.1.0.0    
Pb: Pb.$fct-dn-*_psl.1.0.0    
Bi: Bi.$fct-dn-*_psl.1.0.0    
Po: Po.$fct-dn-*_psl.1.0.0    
At: At.$fct-dn-*_psl.1.0.0    
Rn: Rn.$fct-dn-*_psl.1.0.0
Fr: Fr.$fct-spdn-*_psl.1.0.0    
Ra: Ra.$fct-spdn-*_psl.1.0.0    
Ac: Ac.$fct-spfn-*_psl.1.0.0    
Th: Th.$fct-spfn-*_psl.1.0.0    
Pa: Pa.$fct-spfn-*_psl.1.0.0    
U:  U.$fct-spfn-*_psl.1.0.0
Np: Np.$fct-spfn-*_psl.1.0.0    
Pu: Pu.$fct-spfn-*_psl.1.0.0"""
    # use PAW, PBE, SOC
    star = 'kjpaw'
    fct = 'rel-pbe'
    result = {}
    suggested = suggested.replace('*', star).replace('$fct', fct)
    suggested = suggested.split('\n')
    for line in suggested:
        element, filename = line.strip().split(":")
        element = element.strip()
        filename = filename.strip() + '.UPF'
        # I cannot find these UPF, temporarily replace it
        if filename == 'Co.rel-pbe-n-kjpaw_psl.0.3.1.UPF':
            filename = 'Co.rel-pbe-spn-kjpaw_psl.0.3.1.UPF'
        if filename == 'Au.rel-pbe-n-kjpaw_psl.1.0.1.UPF':
            filename = 'Au.rel-pbe-n-kjpaw_psl.1.0.0.UPF'
        print(filename)
        if dirname is not None:
            filename = os.path.join(dirname, filename)
        else:
            if not os.path.exists(filename):
                # download from QE website
                url = qe_site + filename
                urllib.request.urlretrieve(url, filename)
        result[element] = get_metadata(filename)

    with open(output_filename, 'w') as f:
        json.dump(result, f, indent=2)

def generate_dojo_metadata():
    """generate metadata for pseduo-dojo SOC pseudos, from
    http://www.pseudo-dojo.org/nc-fr-04_pbe_stringent.json
    """
    dojo_json = 'nc-fr-04_pbe_standard.json'
    with open(dojo_json) as f:
        dojo = json.load(f)

    result = {}
    for element in dojo:
        # in pseudo-dojo standard accurary, there is no UPF endswith '_r',
        # in stringent accruray, there are UPF endswith '_r'.
        # Not sure what '_r' means, but if a element endswith '_r', then
        # its cutoff is not shown in the HTML page.
        if element.endswith('_r'):
            continue
        filename = element + '.upf'
        result[element] = {
            "filename": filename,
            "md5": md5(filename),
            "pseudopotential": "Dojo",
            # use normal accurary, and the original unit is Hartree - convert to Rydberg
            "cutoff": 2.0 * dojo[element]['hn'],
            "dual": 4.0
        }

    with open('dojo_nc_fr.json', 'w') as f:
        json.dump(result, f, indent=2)

if __name__ == '__main__':
    # generate_pslibrary_metadata()
    generate_dojo_metadata()
