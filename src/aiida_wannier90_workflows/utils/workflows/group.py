"""Functions for group of workchains."""

import typing as ty

from aiida import orm


def get_mapping_for_group(
    wan_group: ty.Union[str, orm.Group],
    dft_group: ty.Union[str, orm.Group],
    match_by_formula: bool = False,
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
    from aiida_quantumespresso.workflows.pw.bands import PwBandsWorkChain

    if isinstance(wan_group, str):
        wan_group = orm.load_group(wan_group)
    if isinstance(dft_group, str):
        dft_group = orm.load_group(dft_group)

    if len(dft_group.nodes) == 0:
        print(f"DFT group<{dft_group.pk}> is empty")
        return None

    if len(wan_group.nodes) == 0:
        print(f"Wannier group<{wan_group.pk}> is empty")
        return None

    # PwBandsWorkChain calls seekpath, I need to use seekpath reduced structure
    if dft_group.nodes[0].process_class == PwBandsWorkChain:
        dft_structures = {_.outputs.primitive_structure: _ for _ in dft_group.nodes}
    else:
        # Just check the input structure
        if "structure" in dft_group.nodes[0].inputs:
            dft_structures = {_.inputs.structure: _ for _ in dft_group.nodes}
        elif "structure" in dft_group.nodes[0].inputs["pw"]:
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


def standardize_groupname(label: str) -> str:
    """Replace ``/`` in group label to ``_``, to be used as filename.

    :param label: [description]
    :type label: str
    :return: [description]
    :rtype: str
    """
    new_label = label.replace("/", "-")
    return new_label
