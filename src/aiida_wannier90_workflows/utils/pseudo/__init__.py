"""Utility functions for pseudo potential family."""
import typing as ty

from aiida import orm
from aiida.common import exceptions
from aiida.plugins import DataFactory, GroupFactory

PseudoPotentialData = DataFactory("pseudo")
SsspFamily = GroupFactory("pseudo.family.sssp")
PseudoDojoFamily = GroupFactory("pseudo.family.pseudo_dojo")
CutoffsPseudoPotentialFamily = GroupFactory("pseudo.family.cutoffs")


def get_pseudo_and_cutoff(
    pseudo_family: str, structure: orm.StructureData
) -> ty.Tuple[ty.Mapping[str, PseudoPotentialData], float, float]:
    """Get pseudo potential and cutoffs of a given pseudo family and structure.

    :param pseudo_family: [description]
    :param structure: [description]
    :raises ValueError: [description]
    :raises ValueError: [description]
    :return: [description]
    """
    try:
        pseudo_set = (PseudoDojoFamily, SsspFamily, CutoffsPseudoPotentialFamily)
        pseudo_family = (
            orm.QueryBuilder()
            .append(pseudo_set, filters={"label": pseudo_family})
            .one()[0]
        )
    except exceptions.NotExistent as exception:
        raise ValueError(
            f"required pseudo family `{pseudo_family}` is not installed. Please use `aiida-pseudo install` to"
            "install it."
        ) from exception

    try:
        cutoff_wfc, cutoff_rho = pseudo_family.get_recommended_cutoffs(
            structure=structure, unit="Ry"
        )
        pseudos = pseudo_family.get_pseudos(structure=structure)
    except ValueError as exception:
        raise ValueError(
            f"failed to obtain recommended cutoffs for pseudo family `{pseudo_family}`: {exception}"
        ) from exception

    return pseudos, cutoff_wfc, cutoff_rho


def get_pseudo_orbitals(pseudos: ty.Mapping[str, PseudoPotentialData]) -> dict:
    """Get the pseudo wave functions contained in the pseudo potential.

    Currently only support the following pseudopotentials installed by `aiida-pseudo`:
        * SSSP/1.1/PBE/efficiency
        * SSSP/1.1/PBEsol/efficiency
        * PseudoDojo/0.4/LDA/SR/standard/upf
        * PseudoDojo/0.4/LDA/SR/stringent/upf
        * PseudoDojo/0.4/PBE/SR/standard/upf
        * PseudoDojo/0.4/PBE/SR/stringent/upf
        * PseudoDojo/0.5/PBE/SR/standard/upf
        * PseudoDojo/0.5/PBE/SR/stringent/upf
    """
    from .data import load_pseudo_metadata

    pseudo_data = []
    pseudo_data.append(load_pseudo_metadata("semicore/SSSP_1.1_PBEsol_efficiency.json"))
    pseudo_data.append(load_pseudo_metadata("semicore/SSSP_1.1_PBE_efficiency.json"))
    pseudo_data.append(
        load_pseudo_metadata("semicore/PseudoDojo_0.4_PBE_SR_standard_upf.json")
    )
    pseudo_data.append(
        load_pseudo_metadata("semicore/PseudoDojo_0.4_PBE_SR_stringent_upf.json")
    )
    pseudo_data.append(
        load_pseudo_metadata("semicore/PseudoDojo_0.5_PBE_SR_standard_upf.json")
    )
    pseudo_data.append(
        load_pseudo_metadata("semicore/PseudoDojo_0.5_PBE_SR_stringent_upf.json")
    )
    pseudo_data.append(
        load_pseudo_metadata("semicore/PseudoDojo_0.4_LDA_SR_standard_upf.json")
    )
    pseudo_data.append(
        load_pseudo_metadata("semicore/PseudoDojo_0.4_LDA_SR_stringent_upf.json")
    )

    pseudo_orbitals = {}
    for element in pseudos:
        for data in pseudo_data:
            if data.get(element, {}).get("md5", "") == pseudos[element].md5:
                pseudo_orbitals[element] = data[element]
                break
        else:
            raise ValueError(
                f"Cannot find pseudopotential {element} with md5 {pseudos[element].md5}"
            )

    return pseudo_orbitals


def get_semicore_list(structure: orm.StructureData, pseudo_orbitals: dict) -> list:
    """Get semicore states (a subset of pseudo wavefunctions) in the pseudopotential.

    :param structure: [description]
    :param pseudo_orbitals: [description]
    :return: [description]
    """
    from copy import deepcopy

    # pw2wannier90.x/projwfc.x store pseudo-wavefunctions in the same order
    # as ATOMIC_POSITIONS in pw.x input file; aiida-quantumespresso writes
    # ATOMIC_POSITIONS in the order of StructureData.sites.
    # Note some times the PSWFC in UPF files are not ordered, i.e. it's not
    # always true that the first several PSWFC are semicores states, the
    # json file which we loaded in the self.ctx.pseudo_pswfcs already
    # consider this ordering, e.g.
    # "Ce": {
    #     "filename": "Ce.GGA-PBE-paw-v1.0.UPF",
    #     "md5": "c46c5ce91c1b1c29a1e5d4b97f9db5f7",
    #     "pswfcs": ["5S", "6S", "5P", "6P", "5D", "6D", "4F", "5F"],
    #     "semicores": ["5S", "5P"]
    # }
    label2num = {"S": 1, "P": 3, "D": 5, "F": 7}

    semicore_list = []  # index should start from 1
    num_pswfcs = 0

    for site in structure.sites:
        # Here I use deepcopy to make sure list.remove() does not interfere with the original list.
        site_pswfcs = deepcopy(pseudo_orbitals[site.kind_name]["pswfcs"])
        site_semicores = deepcopy(pseudo_orbitals[site.kind_name]["semicores"])

        for orb in site_pswfcs:
            num_orbs = label2num[orb[-1]]
            if orb in site_semicores:
                site_semicores.remove(orb)
                semicore_list.extend(
                    list(range(num_pswfcs + 1, num_pswfcs + num_orbs + 1))
                )
            num_pswfcs += num_orbs

        if len(site_semicores) != 0:
            return ValueError(
                f"Error when processing pseudo {site.kind_name} with orbitals {pseudo_orbitals}"
            )

    return semicore_list


def get_wannier_number_of_bands(
    structure,
    pseudos,
    factor=1.2,
    only_valence=False,
    spin_polarized=False,
    spin_orbit_coupling: bool = False,
):
    """Estimate number of bands for a Wannier90 calculation.

    :param structure: crystal structure
    :param pseudos: dictionary of pseudopotentials
    :type pseudos: dict of aiida.orm.UpfData
    :param only_valence: return only occupied number of badns
    :type only_valence: bool
    :param spin_polarized: magnetic calculation?
    :type spin_polarized: bool
    :param spin_orbit_coupling: spin orbit coupling calculation?
    :type spin_orbit_coupling: bool
    :return: number of bands for Wannier90 SCDM
    :rtype: int
    """
    from .upf import get_upf_content, is_soc_pseudo

    if spin_orbit_coupling:
        composition = structure.get_composition()
        for kind in composition:
            upf = pseudos[kind]
            upf_content = get_upf_content(upf)
            if not is_soc_pseudo(upf_content):
                raise ValueError("Should use SOC pseudo for SOC calculation")

    num_electrons = get_number_of_electrons(structure, pseudos)
    num_projections = get_number_of_projections(structure, pseudos, spin_orbit_coupling)
    nspin = 2 if spin_polarized else 1
    # TODO check nospin, spin, soc  # pylint: disable=fixme
    if only_valence:
        num_bands = int(0.5 * num_electrons * nspin)
    else:
        # nbands must > num_projections = num_wann
        num_bands = max(
            int(0.5 * num_electrons * nspin * factor),
            int(0.5 * num_electrons * nspin + 4 * nspin),
            int(num_projections * factor),
            int(num_projections + 4),
        )
    return num_bands


def get_number_of_projections(
    structure: orm.StructureData,
    pseudos: ty.Mapping[str, orm.UpfData],
    spin_orbit_coupling: ty.Optional[bool] = None,
) -> int:
    """Get number of projections for the structure with the given pseudopotential files.

    Usage:
        nprojs = get_number_of_projections(struct_MgO, {'Mg':UpfData_Mg, 'O':UpfData_O})

    :param structure: crystal structure
    :type structure: aiida.orm.StructureData
    :param pseudos: a dictionary contains orm.UpfData of the structure
    :type pseudos: dict
    :return: number of projections
    :rtype: int
    """
    import aiida_pseudo.data.pseudo.upf

    from .upf import get_number_of_projections_from_upf, get_upf_content, is_soc_pseudo

    if not isinstance(structure, orm.StructureData):
        raise ValueError(
            f"The type of structure is {type(structure)}, only aiida.orm.StructureData is accepted"
        )
    if not isinstance(pseudos, ty.Mapping):
        raise ValueError(
            f"The type of pseudos is {type(pseudos)}, only dict is accepted"
        )
    for key, val in pseudos.items():
        if not isinstance(key, str) or not isinstance(
            val, (orm.UpfData, aiida_pseudo.data.pseudo.upf.UpfData)
        ):
            raise ValueError(
                f"The type of <{key}, {val}> in pseudos is <{type(key)}, {type(val)}>, "
                "only <str, aiida.orm.UpfData> type is accepted"
            )

    # e.g. composition = {'Ga': 1, 'As': 1}
    composition = structure.get_composition()

    if spin_orbit_coupling is None:
        # I use the first pseudo to detect SOCs
        kind = list(composition.keys())[0]
        spin_orbit_coupling = is_soc_pseudo(get_upf_content(pseudos[kind]))

    tot_nprojs = 0
    for kind in composition:
        upf = pseudos[kind]
        nprojs = get_number_of_projections_from_upf(upf)
        soc = is_soc_pseudo(get_upf_content(pseudos[kind]))
        if spin_orbit_coupling and not soc:
            # For SOC calculation with non-SOC pseudo, QE will generate
            # 2 PSWFCs from each one PSWFC in the pseudo
            nprojs *= 2
        elif not spin_orbit_coupling and soc:
            # For non-SOC calculation with SOC pseudo, QE will average
            # the 2 PSWFCs into one
            nprojs //= 2
        tot_nprojs += nprojs * composition[kind]

    return tot_nprojs


def get_projections(
    structure: orm.StructureData, pseudos: ty.Mapping[str, orm.UpfData]
):
    """Get wannier90 projection block for the structure with a given pseudopotential files.

    Usage:
        projs = get_projections(struct_MgO, {'Mg':UpfData_Mg, 'O':UpfData_O})

    :param structure: crystal structure
    :type structure: aiida.orm.StructureData
    :param pseudos: a dictionary contains orm.UpfData of the structure
    :type pseudos: dict
    :return: wannier90 projection block
    :rtype: list
    """
    import aiida_pseudo.data.pseudo.upf

    from .upf import get_projections_from_upf

    if not isinstance(structure, orm.StructureData):
        raise ValueError(
            f"The type of structure is {type(structure)}, only aiida.orm.StructureData is accepted"
        )
    if not isinstance(pseudos, ty.Mapping):
        raise ValueError(
            f"The type of pseudos is {type(pseudos)}, only dict is accepted"
        )
    for key, val in pseudos.items():
        if not isinstance(key, str) or not isinstance(
            val, (orm.UpfData, aiida_pseudo.data.pseudo.upf.UpfData)
        ):
            raise ValueError(
                f"The type of <{key}, {val}> in pseudos is <{type(key)}, {type(val)}>, "
                "only <str, aiida.orm.UpfData> type is accepted"
            )

    projections = []
    # e.g. composition = {'Ga': 1, 'As': 1}
    composition = structure.get_composition()
    for kind in composition:
        upf = pseudos[kind]
        projs = get_projections_from_upf(upf)
        projections.extend(projs)
    return projections


def get_number_of_electrons(
    structure: orm.StructureData, pseudos: ty.Mapping[str, orm.UpfData]
) -> float:
    """Get number of electrons for the structure based on pseudopotentials.

    Usage:
        nprojs = get_number_of_electrons(struct_MgO, {'Mg':UpfData_Mg, 'O':UpfData_O})

    :param structure: crystal structure
    :type structure: aiida.orm.StructureData
    :param pseudos: a dictionary contains orm.UpfData of the structure
    :type pseudos: dict
    :return: number of electrons
    :rtype: float
    """
    import aiida_pseudo.data.pseudo.upf

    from .upf import get_number_of_electrons_from_upf

    if not isinstance(structure, orm.StructureData):
        raise ValueError(
            f"The type of structure is {type(structure)}, only aiida.orm.StructureData is accepted"
        )
    if not isinstance(pseudos, ty.Mapping):
        raise ValueError(
            f"The type of pseudos is {type(pseudos)}, only dict is accepted"
        )
    for key, val in pseudos.items():
        if not isinstance(key, str) or not isinstance(
            val, (orm.UpfData, aiida_pseudo.data.pseudo.upf.UpfData)
        ):
            raise ValueError(
                f"The type of <{key}, {val}> in pseudos is <{type(key)}, {type(val)}>, "
                "only <str, aiida.orm.UpfData> type is accepted"
            )

    tot_nelecs = 0
    # e.g. composition = {'Ga': 1, 'As': 1}
    composition = structure.get_composition()
    for kind in composition:
        upf = pseudos[kind]
        nelecs = get_number_of_electrons_from_upf(upf)
        tot_nelecs += nelecs * composition[kind]

    return tot_nelecs
