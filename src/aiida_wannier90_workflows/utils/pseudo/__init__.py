"""Utility functions for pseudo potential family."""

import re
import typing as ty
import warnings

from upf_tools import UPFDict

from aiida import orm
from aiida.common import exceptions
from aiida.plugins import DataFactory, GroupFactory

PseudoPotentialData = DataFactory("pseudo")
PseudoPotentialFamily = GroupFactory("pseudo.family")
SsspFamily = GroupFactory("pseudo.family.sssp")
PseudoDojoFamily = GroupFactory("pseudo.family.pseudo_dojo")
CutoffsPseudoPotentialFamily = GroupFactory("pseudo.family.cutoffs")


# Two-class split only because requires-python is 3.9: merge into a single
# TypedDict with NotRequired[...] fields once python >= 3.11.
class _PseudoOrbitalsTableMeta(ty.TypedDict, total=False):
    """Optional provenance metadata carried by the bundled semicore tables."""

    filename: str
    md5: str


class PseudoOrbitals(_PseudoOrbitalsTableMeta):
    """Valence-orbital description of one pseudopotential.

    * ``pswfcs`` — labels of the pseudo wave functions, e.g. ``["3S", "3P"]``.
    * ``semicores`` — the subset of ``pswfcs`` to treat as semicore states
      (excluded from the Wannier manifold when ``exclude_semicore`` is on).
    """

    pswfcs: ty.List[str]
    semicores: ty.List[str]


def _load_pseudo_family(
    label: str, pseudo_set: ty.Tuple[type, ...] = (PseudoPotentialFamily,)
):
    """Return the stored pseudopotential family with the given label.

    :param pseudo_set: the family classes to search. The default accepts every
        ``aiida-pseudo`` family; pass a narrower set when the caller needs a
        capability only some families have.
    :raises ValueError: no family in ``pseudo_set`` carries that label.
    """
    try:
        return orm.QueryBuilder().append(pseudo_set, filters={"label": label}).one()[0]
    except exceptions.NotExistent as exception:
        raise ValueError(
            f"required pseudo family `{label}` is not installed. Please use `aiida-pseudo install` to"
            "install it."
        ) from exception


def get_pseudos(
    pseudo_family: str, structure: orm.StructureData
) -> ty.Mapping[str, PseudoPotentialData]:
    """Get the pseudo potential of each kind of a structure, from a pseudo family.

    Any ``aiida-pseudo`` family serves, including one built by the user with
    ``aiida-pseudo install family``: no recommended cutoffs are needed.

    :param pseudo_family: label of the family to take the pseudos from.
    :param structure: the structure whose kinds are to be covered.
    :raises ValueError: the family is not installed, or does not cover every
        kind of the structure.
    """
    return _load_pseudo_family(pseudo_family).get_pseudos(structure=structure)


def get_pseudo_and_cutoff(
    pseudo_family: str, structure: orm.StructureData
) -> ty.Tuple[ty.Mapping[str, PseudoPotentialData], float, float]:
    """Get the pseudo potentials and the recommended cutoffs of a pseudo family.

    Only families that recommend cutoffs qualify, i.e. SSSP, PseudoDojo and any
    family with a cutoff stringency defined. Use :func:`get_pseudos` when the
    cutoffs are not needed.

    :param pseudo_family: label of the family to take the pseudos from.
    :param structure: the structure whose kinds are to be covered.
    :raises ValueError: the family is not installed among those that recommend
        cutoffs.
    :raises ValueError: the family recommends no cutoffs for this structure.
    :return: the pseudos per kind, the wave-function cutoff and the charge
        density cutoff, both in Ry.
    """
    pseudo_family = _load_pseudo_family(
        pseudo_family, (PseudoDojoFamily, SsspFamily, CutoffsPseudoPotentialFamily)
    )

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


def get_pseudo_orbitals(
    pseudos: ty.Mapping[str, PseudoPotentialData],
    overrides: ty.Optional[ty.Mapping[str, PseudoOrbitals]] = None,
) -> ty.Dict[str, PseudoOrbitals]:
    """Get the valence orbitals (pseudo wave functions) of each pseudopotential.

    Resolution proceeds in three tiers, per kind:

    1. an entry in ``overrides``, when given;
    2. the bundled semicore tables (matched by md5), which carry curated
       ``semicores`` lists for the following families installed by
       ``aiida-pseudo``:

       * SSSP/1.3/PBE/efficiency
       * SSSP/1.3/PBEsol/efficiency
       * SSSP/1.1/PBE/efficiency
       * SSSP/1.1/PBEsol/efficiency
       * PseudoDojo/0.4/LDA/SR/standard/upf
       * PseudoDojo/0.4/LDA/SR/stringent/upf
       * PseudoDojo/0.4/PBE/SR/standard/upf
       * PseudoDojo/0.4/PBE/SR/stringent/upf
       * PseudoDojo/0.5/PBE/SR/standard/upf
       * PseudoDojo/0.5/PBE/SR/stringent/upf
       * PseudoDojo/0.4/PBE/FR/standard/upf
       * Pslibrary/1.0.0/relPBE/PAW

    3. the pseudo's own ``PP_PSWFC`` block, read via ``upf-tools``: the labels
       of its atomic wave functions are the valence orbitals, with an empty
       ``semicores`` list (nothing is auto-excluded) and a warning.

    A ``ValueError`` is raised only when none of the tiers can resolve a
    pseudopotential; its message explains what to pass via ``overrides``.

    :param overrides: optional per-kind :class:`PseudoOrbitals` entries that
        take precedence over the bundled tables, e.g. ``{"Ti": {"pswfcs":
        ["3S", "3P", "4S", "3D"], "semicores": ["3S", "3P"]}}``.
    """
    from .data import load_pseudo_metadata

    pseudo_data = []
    pseudo_data.append(load_pseudo_metadata("semicore/SSSP_1.3_PBE_efficiency.json"))
    pseudo_data.append(load_pseudo_metadata("semicore/SSSP_1.3_PBEsol_efficiency.json"))
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
        if overrides is not None and kind in overrides:
            pseudo_orbitals[kind] = overrides[kind]
            continue
        for data in pseudo_data:
            if data.get(pseudos[kind].element, {}).get("md5", "") == pseudos[kind].md5:
                pseudo_orbitals[kind] = data[pseudos[kind].element]
                break
        else:
            derived = _derive_pseudo_orbitals_from_upf(pseudos[kind])
            if derived is None:
                raise ValueError(
                    f"Cannot find pseudopotential {kind} with md5 {pseudos[kind].md5}, "
                    "and its valence orbitals could not be read from the pseudopotential "
                    "file. Provide the entry explicitly via the `overrides` argument, e.g. "
                    '{"' + str(kind) + '": {"pswfcs": ["3S", "3P"], "semicores": []}}.'
                )
            warnings.warn(
                f"Pseudopotential {kind} (md5 {pseudos[kind].md5}) is not in the bundled "
                "semicore tables; its valence orbitals were read directly from the "
                "pseudopotential file's atomic wave functions "
                f"({derived['pswfcs']}), and no semicore states will be excluded. Pass "
                "`overrides` to specify them explicitly."
            )
            pseudo_orbitals[kind] = derived

    return pseudo_orbitals


# Angular-momentum quantum number to spectroscopic letter, used to build a
# pseudo wave function's label from its (n, l) when the UPF's own label field
# is missing or unusable.
_L_TO_LETTER = {0: "S", 1: "P", 2: "D", 3: "F", 4: "G", 5: "H"}


def _chi_label(entry: ty.Mapping) -> ty.Optional[str]:
    """Build the ``n`` + spectroscopic-letter label of one ``PP_CHI`` entry.

    ``l`` (the angular momentum) is the physical ground truth and is always
    carried by a ``PP_CHI`` entry, so the spectroscopic letter is taken from
    it. The principal quantum number ``n`` is taken from the leading digits of
    the UPF's own ``label`` field, which carries the true atomic ``n`` (``3S``,
    ``6P``, ...) across generators. The ``n`` *attribute* is not reliable for
    this: some ultrasoft/PAW generators write the *pseudo* principal quantum
    number there -- radial nodes + l + 1 -- so a node-free 6s/6p/5d projector
    is labelled n=1/2/3 rather than 6/6/5. The attribute is used only as a
    fallback when the label has no leading digit (empty or non-standard label
    but intact ``n``/``l``).

    Returns ``None`` when neither ``l`` nor ``n`` can be determined, so the
    caller can fail closed.
    """
    try:
        l = int(entry.get("l"))
    except (TypeError, ValueError):
        return None
    letter = _L_TO_LETTER.get(l)
    if letter is None:
        return None
    label = entry.get("label")
    match = re.match(r"\s*(\d+)", label) if isinstance(label, str) else None
    n = match.group(1) if match is not None else entry.get("n")
    try:
        n = int(n)
    except (TypeError, ValueError):
        return None
    return f"{n}{letter}"


def _derive_pseudo_orbitals_from_upf(
    pseudo: PseudoPotentialData,
) -> ty.Optional[PseudoOrbitals]:
    """Read the valence orbitals of a pseudo directly from its ``PP_PSWFC`` block.

    The UPF states its own valence: each ``PP_CHI`` atomic wave function carries
    the angular momentum ``l`` and (via its label or ``n`` attribute) the
    principal quantum number, so the pseudo wave function labels are read off
    rather than guessed. ``semicores`` is left empty -- which shells to
    auto-exclude is a curated judgement, so nothing is excluded for
    UPF-derived entries.

    Fully-relativistic pseudos list one ``PP_CHI`` per j channel (each l > 0
    shell splits into j = l +/- 1/2); the split is collapsed by keeping each
    (n, l) label once, in first-appearance order -- the order in which
    ``pw2wannier90.x``/``projwfc.x`` emit the projections.

    Returns ``None`` -- so ``get_pseudo_orbitals`` fails closed and asks for an
    explicit ``overrides`` entry -- when the ``PP_PSWFC`` block cannot be read
    (unparseable content, no atomic wave functions, or an entry whose (n, l)
    cannot be recovered) or when the wave function occupations do not sum to
    ``z_valence``, which can indicate a missing valence orbital.
    """
    try:
        upf = UPFDict.from_str(pseudo.get_content())
        chi = upf["pswfc"]["chi"]
    except (KeyError, TypeError, ValueError, IndexError, AttributeError, SyntaxError):
        # xml.etree ParseError subclasses SyntaxError; a missing PP_PSWFC or
        # PP_HEADER section surfaces as KeyError once parsed; unparseable
        # content yields a dict without the expected keys.
        return None
    if not chi:
        return None

    labels = []
    for entry in chi:
        label = _chi_label(entry)
        if label is None:
            # A wave function whose (n, l) cannot be recovered makes the whole
            # derivation untrustworthy; fail closed rather than emit a partial
            # or wrongly-ordered orbital set.
            return None
        labels.append(label)
    # Collapse fully-relativistic j-splits: keep one label per unique (n, l),
    # preserving first-appearance order.
    pswfcs = list(dict.fromkeys(labels))

    if _occupations_disagree(pseudo, chi):
        return None

    return PseudoOrbitals(
        filename=getattr(pseudo, "filename", f"{pseudo.element}.upf"),
        md5=pseudo.md5,
        pswfcs=pswfcs,
        semicores=[],
    )


def _occupations_disagree(
    pseudo: PseudoPotentialData, chi: ty.Sequence[ty.Mapping]
) -> bool:
    """Return True when the ``PP_CHI`` occupations do not sum to ``z_valence``.

    The atomic wave function occupations must account for the pseudo's valence
    electrons; a disagreement can mean a valence orbital is missing from
    ``PP_PSWFC``, in which case the derived labels would be silently
    incomplete. The derivation therefore fails closed on any disagreement
    beyond numerical noise. The check is skipped when either quantity is
    unavailable.
    """
    z_valence = getattr(pseudo, "z_valence", None)
    if z_valence is None:
        return False
    try:
        occupations = [
            float(entry["occupation"])
            for entry in chi
            if entry.get("occupation") is not None
        ]
        z_valence = float(z_valence)
    except (KeyError, TypeError, ValueError):
        return False
    if not occupations:
        return False
    occ_sum = sum(occupations)
    if abs(occ_sum - z_valence) > 1.0e-3:
        warnings.warn(
            f"Pseudopotential {pseudo.element} (md5 {pseudo.md5}): its atomic "
            f"wave function occupations sum to {occ_sum:g}, which disagrees with "
            f"its z_valence {z_valence:g}, so a valence orbital may be missing "
            "from PP_PSWFC. Refusing the derived orbitals; provide an explicit "
            "`overrides` entry instead."
        )
        return True
    return False


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

    Whether a projector is frozen is decided by, in order of precedence:

    1. an explicit boolean ``frozen`` key on the projector, if present;
    2. otherwise the ``alpha`` key: ``"UPF"`` (or a missing ``alpha``) marks a
       pseudo-atomic orbital, which is frozen; a numeric ``alpha`` marks a
       generated (e.g. hydrogenic) orbital, which is not.

    :param structure: [description]
    :param external_projectors: dict of external projectors, where every external projector
    contains the `label`, `l` and `j`(optional), and optionally `frozen` and/or `alpha`.
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

            # if neither key is defined, it is better to Lowdin all projectors
            frozen = orb.get("frozen", orb.get("alpha", "UPF") == "UPF")
            if frozen:
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
    kind_name = structure.get_kind_names()[0]
    kind = structure.get_kind(kind_name).symbol
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
