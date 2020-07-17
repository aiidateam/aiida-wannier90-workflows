import typing
from aiida import orm
import xml.etree.ElementTree as ET

__all__ = ('parse_zvalence', 'get_number_of_electrons_from_upf', 'get_number_of_electrons',
           'parse_number_of_pswfc', 'get_number_of_projections_from_upf', 'get_number_of_projections')

Dict_of_Upf = typing.Dict[str, orm.UpfData]

def parse_zvalence(upf_content: str) -> float:
    """get z_valcence from a UPF file. No AiiDA dependencies.
    Works for both UPF v1 & v2. Tested on all the SSSP pseudos.

    :param upf_content: the content of the UPF file
    :type upf_content: str
    :return: z_valence of the UPF file
    :rtype: float
    """
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
                continue
        if found_begin and ('/>' in line or '</PP_HEADER>' in line):
            ppheader_block += line + '\n'
            if not found_end:
                found_end = True
                break
        if found_begin:
            ppheader_block += line + '\n'
    # print(ppheader_block)

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

def get_number_of_electrons_from_upf(upf: orm.UpfData) -> float:
    """AiiDA wrapper for `parse_zvalence'

    :param upf: pseudo
    :type upf: aiida.orm.UpfData
    :return: number of electrons
    :rtype: float
    """
    if not isinstance(upf, orm.UpfData):
        raise ValueError(f'The type of upf is {type(upf)}, only aiida.orm.UpfData is accepted')
    upf_name = upf.list_object_names()[0]
    upf_content = upf.get_object_content(upf_name)
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

def parse_number_of_pswfc(upf_content: str) -> int:
    """get the number orbitals in the PSWFC block, 
    i.e. number of occuppied atomic orbitals in the UPF file.
    This is also the number of orbitals used for projections in projwfc.x.
    No AiiDA dependencies.
    Support both UPF v1 and v2 format. Tested on all the SSSP pseudos.

    :param upf_content: the content of the UPF file
    :type upf_content: str
    :return: number of PSWFC 
    :rtype: int
    """
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

    num_projections = 0
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
            num_projections += 2 * l + 1
    else:
        # upf format 2.0.1
        for child in PP_PSWFC:
            l = int(child.get('l'))
            num_projections += 2 * l + 1
    return num_projections

def get_number_of_projections_from_upf(upf: orm.UpfData) -> int:
    """aiida wrapper for `parse_number_of_pswfc`.

    :param upf: the UPF file
    :type upf: aiida.orm.UpfData
    :return: number of projections in the UPF file
    :rtype: int
    """
    if not isinstance(upf, orm.UpfData):
        raise ValueError(f'The type of upf is {type(upf)}, only aiida.orm.UpfData is accepted')
    upf_name = upf.list_object_names()[0]
    upf_content = upf.get_object_content(upf_name)
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