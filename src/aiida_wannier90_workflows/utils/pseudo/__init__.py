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
    pseudo_data.append(
        load_pseudo_metadata("semicore/PseudoDojo_0.4_PBE_FR_standard_upf.json")
    )
    pseudo_data.append(
        load_pseudo_metadata("semicore/PseudoDojo_0.4_PBEsol_FR_standard_upf.json")
    )
    pseudo_data.append(load_pseudo_metadata("semicore/pslibrary_paw_relpbe_1.0.0.json"))

    pseudo_orbitals = {}
    # pseudos dictionary will contain kinds as keys, which may change
    # e.g. when including Hubbard corrections 'Mn'->'Mn3d'
    for kind in pseudos:
        for data in pseudo_data:
            if data.get(pseudos[kind].element, {}).get("md5", "") == pseudos[kind].md5:
                pseudo_orbitals[kind] = data[pseudos[kind].element]
                break
        else:
            raise ValueError(
                f"Cannot find pseudopotential {kind} with md5 {pseudos[kind].md5}"
            )

    return pseudo_orbitals


def get_semicore_list(
    structure: orm.StructureData, pseudo_orbitals: dict, spin_non_collinear: bool
) -> list:
    """Get semicore states (a subset of pseudo wavefunctions) in the pseudopotential.

    :param structure: [description]
    :param pseudo_orbitals: [description]
    :param spin_non_collinear: [description]
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
    # for spin-orbit-coupling, every orbit contains 2 electrons
    nspin = 2 if spin_non_collinear else 1

    semicore_list = []  # index should start from 1
    num_pswfcs = 0

    for site in structure.sites:
        # Here I use deepcopy to make sure list.remove() does not interfere with the original list.
        site_pswfcs = deepcopy(pseudo_orbitals[site.kind_name]["pswfcs"])
        site_semicores = deepcopy(pseudo_orbitals[site.kind_name]["semicores"])

        for orb in site_pswfcs:
            num_orbs = label2num[orb[-1]] * nspin
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


def get_semicore_list_ext(
    structure: orm.StructureData,
    external_projectors: dict,
    pseudo_orbitals: dict,
    spin_non_collinear: bool,
) -> list:
    """Get semicore states (a subset of pseudo wavefunctions) in the external_projectors.

    :param structure: [description]
    :param external_projectors: dict of external projectors, where every external projector
    contains the `label`, `l` and `j`(optional).
    :param pseudo_orbitals: [description]
    :param spin_non_collinear: [description]
    :return: [description]
    """
    from copy import deepcopy

    nspin = 2 if spin_non_collinear else 1

    semicore_list = []  # index should start from 1
    num_projs = 0

    for site in structure.sites:
        # Here I use deepcopy to make sure list.remove() does not interfere with the original list.
        # site_pswfcs = deepcopy(pseudo_orbitals[site.kind_name]["pswfcs"])
        site_semicores = deepcopy(pseudo_orbitals[site.kind_name]["semicores"])

        for orb in external_projectors[structure.get_kind(site.kind_name).symbol]:
            if "j" in orb:
                num_orbs = round(2 * orb["j"] + 1)
            else:
                num_orbs = (2 * orb["l"] + 1) * nspin

            if orb["label"].upper() in site_semicores:
                semicore_list.extend(
                    list(range(num_projs + 1, num_projs + num_orbs + 1))
                )
            num_projs += num_orbs

    return semicore_list


def get_frozen_list_ext(
    structure: orm.StructureData,
    external_projectors: dict,
    spin_non_collinear: bool,
) -> list:
    """Get frozen states (a subset of pseudo wavefunctions) in the external_projectors.

    :param structure: [description]
    :param external_projectors: dict of external projectors, where every external projector
    contains the `label`, `l` and `j`(optional).
    :param pseudo_orbitals: [description]
    :param spin_non_collinear: [description]
    :return: [description]
    """

    nspin = 2 if spin_non_collinear else 1

    frozen_list = []  # index should start from 1
    num_projs = 0

    for site in structure.sites:
        for orb in external_projectors[structure.get_kind(site.kind_name).symbol]:
            if "j" in orb:
                num_orbs = round(2 * orb["j"] + 1)
            else:
                num_orbs = (2 * orb["l"] + 1) * nspin

            alpha = orb.get(
                "alpha", "UPF"
            )  # if not defined, it is better to Lowdin all projectors
            if alpha == "UPF":
                frozen_list.extend(list(range(num_projs + 1, num_projs + num_orbs + 1)))
            num_projs += num_orbs

    return frozen_list


def get_wannier_number_of_bands(  # pylint: disable=too-many-positional-arguments
    structure,
    pseudos,
    factor=1.2,
    only_valence=False,
    spin_polarized=False,
    spin_non_collinear: bool = False,
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
    :param spin_non_collinear: non-collinear or spin-orbit-coupling
    :type spin_non_collinear: bool
    :param spin_orbit_coupling: spin orbit coupling calculation?
    :type spin_orbit_coupling: bool
    :return: number of bands for Wannier90 SCDM
    :rtype: int
    """
    from .upf import get_upf_content, is_soc_pseudo

    if spin_orbit_coupling:
        for site in structure.sites:
            upf = pseudos[site.kind_name]
            upf_content = get_upf_content(upf)
            if not is_soc_pseudo(upf_content):
                raise ValueError("Should use SOC pseudo for SOC calculation")

    num_electrons = get_number_of_electrons(structure, pseudos)
    num_projections = get_number_of_projections(
        structure, pseudos, spin_non_collinear, spin_orbit_coupling
    )
    nspin = 2 if (spin_polarized or spin_non_collinear) else 1
    # TODO check nospin, spin, soc  # pylint: disable=fixme
    if only_valence:
        num_bands = int(0.5 * num_electrons * nspin)
    else:
        # nbands must > num_projections = num_wann
        num_bands = max(
            int(0.5 * num_electrons * nspin * factor),
            int(0.5 * num_electrons * nspin + 4 * nspin),
            int(num_projections * factor),
            int(num_projections + 4 * nspin),
        )
    return num_bands


def get_wannier_number_of_bands_ext(  # pylint: disable=too-many-positional-arguments
    structure,
    pseudos,
    external_projectors,
    factor=1.2,
    only_valence=False,
    spin_polarized=False,
    spin_non_collinear: bool = False,
    spin_orbit_coupling: bool = False,
):
    """Estimate number of bands for a Wannier90 calculation.

    :param structure: crystal structure
    :param pseudos: dictionary of pseudopotentials
    :type pseudos: dict of aiida.orm.UpfData
    :param external_projectors: dict of external projectors
    :type external_projectors: dict
    :param only_valence: return only occupied number of badns
    :type only_valence: bool
    :param spin_polarized: magnetic calculation?
    :type spin_polarized: bool
    :param spin_non_collinear: non-collinear or spin-orbit-coupling
    :type spin_non_collinear: bool
    :param spin_orbit_coupling: spin orbit coupling calculation?
    :type spin_orbit_coupling: bool
    :return: number of bands for Wannier90 SCDM
    :rtype: int
    """
    from .upf import get_upf_content, is_soc_pseudo

    if spin_orbit_coupling:
        for site in structure.sites:
            upf = pseudos[site.kind_name]
            upf_content = get_upf_content(upf)
            if not is_soc_pseudo(upf_content):
                raise ValueError("Should use SOC pseudo for SOC calculation")

    num_electrons = get_number_of_electrons(structure, pseudos)
    num_projections = get_number_of_projections_ext(
        structure, external_projectors, spin_non_collinear, spin_orbit_coupling
    )
    nspin = 2 if (spin_polarized or spin_non_collinear) else 1
    # TODO check nospin, spin, soc  # pylint: disable=fixme
    if only_valence:
        num_bands = int(0.5 * num_electrons * nspin)
    else:
        # nbands must > num_projections = num_wann
        num_bands = max(
            int(0.5 * num_electrons * nspin * factor),
            int(0.5 * num_electrons * nspin + 4 * nspin),
            int(num_projections * factor),
            int(num_projections + 4 * nspin),
        )
    return num_bands


def get_number_of_projections(
    structure: orm.StructureData,
    pseudos: ty.Mapping[str, orm.UpfData],
    spin_non_collinear: bool,
    spin_orbit_coupling: ty.Optional[bool] = None,
) -> int:
    """Get number of projections for the structure with the given pseudopotential files.

    Usage:
        nprojs = get_number_of_projections(struct_MgO, {'Mg':UpfData_Mg, 'O':UpfData_O})

    :param structure: crystal structure
    :type structure: aiida.orm.StructureData
    :param pseudos: a dictionary contains orm.UpfData of the structure
    :type pseudos: dict
    :param spin_non_collinear: non-collinear or spin-orbit-coupling
    :type spin_non_collinear: bool
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

    if spin_orbit_coupling is None:
        # I use the first pseudo to detect SOCs
        kind = structure.get_kind_names()[0]
        spin_orbit_coupling = is_soc_pseudo(get_upf_content(pseudos[kind]))

    tot_nprojs = 0
    for site in structure.sites:
        upf = pseudos[site.kind_name]
        nprojs = get_number_of_projections_from_upf(upf)
        if spin_non_collinear and not spin_orbit_coupling:
            # For magnetic calculation with non-SOC pseudo, QE will generate
            # 2 PSWFCs from each one PSWFC in the pseudo
            # For collinear-magnetic calculation, spin up and down will calc
            # seperately, so nprojs do not times 2
            nprojs *= 2
        elif not spin_non_collinear and spin_orbit_coupling:
            # For non-magnetic calculation with SOC pseudo, QE will average
            # the 2 PSWFCs into one
            nprojs //= 2
        tot_nprojs += nprojs

    return tot_nprojs


def get_number_of_projections_ext(
    structure: orm.StructureData,
    external_projectors: dict,
    spin_non_collinear: bool,
    spin_orbit_coupling: bool = False,
) -> int:
    """Get number of projections for the structure with the given projector dict.

    :param structure: crystal structure
    :type structure: aiida.orm.StructureData
    :param projectors: a dictionary contains projector list of the structure
    :type pseudos: dict
    :param spin_non_collinear: non-collinear or spin-orbit-coupling
    :type spin_non_collinear: bool
    :return: number of projections
    :rtype: int
    """
    if not isinstance(structure, orm.StructureData):
        raise ValueError(
            f"The type of structure is {type(structure)}, only aiida.orm.StructureData is accepted"
        )

    # I use the first projector to detect SOCs
    kind = structure.get_kind_names()[0]
    spin_orbit_coupling_proj = "j" in external_projectors[kind][0]
    if spin_orbit_coupling and not spin_orbit_coupling_proj:
        raise ValueError(
            "SOC spin type is specified, however spin-unpolarized projectors are being used."
        )
    if spin_orbit_coupling_proj and not spin_orbit_coupling:
        raise ValueError(
            "spin-unpolarized spin type is specified, however SOC projectors are being used"
        )

    tot_nprojs = 0
    for site in structure.sites:
        nprojs = 0
        element = structure.get_kind(site.kind_name).symbol
        for orb in external_projectors[element]:
            if spin_orbit_coupling:
                nprojs += round(2 * orb["j"]) + 1
            else:
                nprojs += 2 * orb["l"] + 1
        if spin_non_collinear and not spin_orbit_coupling:
            nprojs *= 2
        tot_nprojs += nprojs

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
    # I am not sure if it will cause some issues,
    # since `get_projections_from_upf` will return the element name as key
    # instead of `kind_name`. It may repreatly extend the `projections` list.
    # e.g. get_projections_from_upf(Mn0) = ['Mn: s', 'Mn: p', 'Mn: d'],
    # and get_projections_from_upf(Mn1) = ['Mn: s', 'Mn: p', 'Mn: d'] will repeat.
    # However, get_projections do not used in AWW,
    # I do not know where it may be used and can not do any test.
    for kind_name in structure.get_kind_names():
        upf = pseudos[kind_name]
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

    for site in structure.sites:
        upf = pseudos[site.kind_name]
        nelecs = get_number_of_electrons_from_upf(upf)
        tot_nelecs += nelecs

    return tot_nelecs


# def reduce_pseudos(
#     pseudos: ty.Mapping[str, orm.UpfData],
#     structure: ty.Optional[orm.StructureData] = None,
# ):
#     """Reduce pseudos from merging kinds with different kind names.

#     Considering pseudofamily.get_pseudos may give
#     {'Fe0': <UpfData>,
#      'Fe1': <UpfData>}
#     However the structure.get_composion will give
#     {'Fe': 2}
#     This function will reduce pseudos to
#     {'Fe': <UpfData>}
#     """

#     reduced_pseudos = {}
#     if structure is not None:
#         for kind in structure.kinds:
#             if kind.name not in pseudos:
#                 raise ValueError(f"{kind.name} do not match the pesudos.\n", pseudos)
#             if kind.symbol not in reduced_pseudos:
#                 reduced_pseudos[kind.symbol] = pseudos[kind.name]
#             else:
#                 if reduced_pseudos[kind.symbol].md5 != pseudos[kind.name].md5:
#                     raise NotImplementedError(
#                         "Same element different pseudos is not implement."
#                     )
#     else:
#         for _, val in pseudos.items():
#             if val.element not in reduced_pseudos:
#                 reduced_pseudos[val.element] = val
#             else:
#                 if val != reduced_pseudos[val.element]:
#                     raise NotImplementedError(
#                         "Same element different pseudos is not implement."
#                     )
#     return reduced_pseudos
