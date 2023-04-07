from aiida.orm import StructureData
# def guess_projection(pmg_structure, policy='light'):
#     if policy.lower() == 'light':
#         wtb_proj=javis_projection()
#     else:
#         raise ValueError(f'Projection policy {policy} is undefined.')

#     projections=[str(elem)+': '+wtb_proj[str(elem)] for elem in pmg_structure.species]
#     print('Guess projection:',projections)
#     return projections

# def guess_num_wann(pmg_structure, policy='light'):
#     if policy.lower() == 'light':
#         wtb_numwan=javis_num_wann()
#     else:
#         raise ValueError(f'Projection policy {policy} is undefined.')

#     ntot=sum([wtb_numwan[str(elem)] for elem in pmg_structure.species])
#     return ntot

def guess_projection(structure:StructureData, policy='light'):
    if policy.lower() == 'light':
        wtb_proj=javis_projection()
    else:
        raise ValueError(f'Projection policy {policy} is undefined.')
    projections=[str(kind.name)+': '+wtb_proj[str(kind.symbol)] for kind in structure.kinds]
    print('Guess projection:',projections)
    return projections

def guess_num_wann(structure:StructureData, policy='light'):
    if policy.lower() == 'light':
        wtb_numwan=javis_num_wann()
    else:
        raise ValueError(f'Projection policy {policy} is undefined.')

    symbols=[structure.get_kind(s.kind_name).get_symbols_string() for s in structure.sites]
    ntot=sum([wtb_numwan[str(elem)] for elem in symbols])
    return ntot

def javis_projection():
    '''
    Projection list from JARVIS Database (https://github.com/usnistgov/jarvis)
    '''
    return {
        'Ag': 's;d',
        'Al': 's;p',
        'Ar': 's;p',
        'As': 's;p',
        'Au': 's;d',
        'B': 's;p',
        'Ba': 's;d',
        'Be': 's;p',
        'Bi': 's;p',
        'Br': 's;p',
        'C': 's;p',
        'Ca': 's;d',
        'Cd': 's;d',
        'Ce': 'f;d;s',
        'Cl': 'p',
        'Co': 's;d',
        'Cr': 's;d',
        'Cs': 's;d',
        'Cu': 's;d',
        'Dy': 's;f',
        'Er': 'f;s',
        'Eu': 'f;s',
        'F': 'p',
        'Fe': 's;d',
        'Ga': 's;p',
        'Gd': 'f;d;s',
        'Ge': 's;p',
        'H': 's',
        'He': 's',
        'Hf': 's;d',
        'Hg': 's;p;d',
        'I': 's;p',
        'In': 's;p',
        'Ir': 's;d',
        'K': 's;d',
        'Kr': 's;p',
        'La': 's;d;f',
        'Li': 's',
        'Lu': 'f;d;s',
        'Mg': 's;p',
        'Mn': 's;d',
        'Mo': 's;d',
        'N': 'p',
        'Na': 's;p',
        'Nb': 's;d',
        'Nd': 'f;s',
        'Ne': 's;p',
        'Ni': 's;d',
        'O': 'p',
        'Os': 's;d',
        'P': 's;p',
        'Pb': 's;p',
        'Pd': 's;d',
        'Pt': 's;d',
        'Rb': 's;d',
        'Re': 's;d',
        'Rh': 's;d',
        'Ru': 's;d',
        'S': 'p',
        'Sb': 's;p',
        'Sc': 's;d',
        'Se': 's;p',
        'Si': 's;p',
        'Sm': 'f;s',
        'Sn': 's;p',
        'Sr': 's;d',
        'Ta': 's;d',
        'Tb': 'f;s',
        'Tc': 's;d',
        'Te': 's;p',
        'Th': 'd;s',
        'Ti': 's;d',
        'Tl': 's;p',
        'U': 'f;s',
        'V': 's;d',
        'W': 's;d',
        'Xe': 's;p',
        'Y': 's;d',
        'Zn': 's;p;d',
        'Zr': 's;d'
    }

def javis_num_wann():
    return {
        'Ag': 6,
        'Al': 4,
        'Ar': 4,
        'As': 4,
        'Au': 6,
        'B': 4,
        'Ba': 6,
        'Be': 4,
        'Bi': 4,
        'Br': 4,
        'C': 4,
        'Ca': 6,
        'Cd': 6,
        'Ce': 13,
        'Cl': 3,
        'Co': 6,
        'Cr': 6,
        'Cs': 6,
        'Cu': 6,
        'Dy': 8,
        'Er': 8,
        'Eu': 8,
        'F': 3,
        'Fe': 6,
        'Ga': 4,
        'Gd': 8,
        'Ge': 4,
        'H': 1,
        'He': 1,
        'Hf': 6,
        'Hg': 9,
        'I': 4,
        'In': 4,
        'Ir': 6,
        'K': 6,
        'Kr': 4,
        'La': 13,
        'Li': 1,
        'Lu': 13,
        'Mg': 4,
        'Mn': 6,
        'Mo': 6,
        'N': 3,
        'Na': 4,
        'Nb': 6,
        'Nd': 8,
        'Ne': 4,
        'Ni': 6,
        'O': 3,
        'Os': 6,
        'P': 4,
        'Pb': 4,
        'Pd': 6,
        'Pt': 6,
        'Rb': 6,
        'Re': 6,
        'Rh': 6,
        'Ru': 6,
        'S': 3,
        'Sb': 4,
        'Sc': 6,
        'Se': 4,
        'Si': 4,
        'Sm': 8,
        'Sn': 4,
        'Sr': 6,
        'Ta': 6,
        'Tb': 8,
        'Tc': 6,
        'Te': 4,
        'Th': 6,
        'Ti': 6,
        'Tl': 4,
        'U': 8,
        'V': 6,
        'W': 6,
        'Xe': 4,
        'Y': 6,
        'Zn': 9,
        'Zr': 6
    }