#!/usr/bin/env runaiida
"""Prepare meta data for SSSP."""
import yaml

from aiida import orm

from aiida_wannier90_workflows.utils.pseudo.upf import (
    get_number_of_electrons_from_upf,
    get_number_of_projections_from_upf,
    get_upf_content,
    parse_pswfc_energy_nosoc,
    parse_pswfc_nosoc,
    parse_pswfc_soc,
)


def write_yaml(group_label: str, filename: str) -> None:
    """Write a yaml file containing meta data for a group of pseudopotentials.

    The yaml file will be used to generate pytest fixtures for SSSP.
    The script searches your AiiDA database for the SSSP, and write to a yaml file which contains
    the number of electrons and number of pseudo wavefunctions of SSSP pseudopotentials.

    :param group_label: [description]
    :type group_label: str
    :param filename: [description]
    :type filename: str
    """

    pseudo_group = orm.load_group(group_label)
    results = {}

    for upf_data in pseudo_group.nodes:
        pswfc = parse_pswfc_nosoc(get_upf_content(upf_data))
        # e.g. [{'l': 0}, {'l': 1}]
        pswfc = [_["l"] for _ in pswfc]

        upf_dict = {
            "filename": upf_data.filename,
            "md5": upf_data.md5,
            "has_so": "F",
            "z_valence": get_number_of_electrons_from_upf(upf_data),
            "number_of_wfc": get_number_of_projections_from_upf(upf_data),
            "pswfc": pswfc,
        }

        results[upf_data.element] = upf_dict

    with open(filename, "w", encoding="utf-8") as file:
        yaml.dump(results, file)

    print(f'Written to file "{filename}"')


def write_yaml_soc(group_label: str, filename: str) -> None:
    """Write a yaml file containing meta data for a group of pseudopotentials.

    The yaml file will be used to generate pytest fixtures for SSSP.
    The script searches your AiiDA database for the SSSP, and write to a yaml file which contains
    the number of electrons and number of pseudo wavefunctions of SSSP pseudopotentials.

    :param group_label: [description]
    :type group_label: str
    :param filename: [description]
    :type filename: str
    """

    pseudo_group = orm.load_group(group_label)
    results = {}

    for upf_data in pseudo_group.nodes:
        pswfc = parse_pswfc_soc(get_upf_content(upf_data))
        # e.g. [{'n': 1, 'l': 0, 'j': 0.5}]
        lchi = [_["l"] for _ in pswfc]
        jchi = [_["j"] for _ in pswfc]
        nn = [_["n"] for _ in pswfc]

        upf_dict = {
            "filename": upf_data.filename,
            "md5": upf_data.md5,
            "has_so": "T",
            "z_valence": get_number_of_electrons_from_upf(upf_data),
            "number_of_wfc": get_number_of_projections_from_upf(upf_data),
            "ppspinorb": {"lchi": lchi, "jchi": jchi, "nn": nn},
        }

        results[upf_data.element] = upf_dict

    with open(filename, "w", encoding="utf-8") as file:
        yaml.dump(results, file)

    print(f'Written to file "{filename}"')


def write_json(group_label: str, filename: str) -> None:
    """Write a yaml file containing meta data for a group of pseudopotentials.

    The yaml file will be used to generate pytest fixtures for SSSP.
    The script searches your AiiDA database for the SSSP, and write to a yaml file which contains
    the number of electrons and number of pseudo wavefunctions of SSSP pseudopotentials.

    :param group_label: [description]
    :type group_label: str
    :param filename: [description]
    :type filename: str
    """
    import json

    pseudo_group = orm.load_group(group_label)
    results = {}

    for upf_data in pseudo_group.nodes:
        upf_content = get_upf_content(upf_data)
        energy = parse_pswfc_energy_nosoc(upf_content)

        pswfcs = [_["label"] for _ in energy]
        # pseudo_energy = [_['pseudo_energy'] for _ in energy]
        # If pseudo_energy < energy_threshold, treat it as semicore,
        # a crude estimate, need to refine manually.
        energy_threshold = -1
        semicores = [
            _["label"] for _ in energy if _["pseudo_energy"] < energy_threshold
        ]

        upf_dict = {
            "filename": upf_data.filename,
            "md5": upf_data.md5,
            "pswfcs": pswfcs,
            "semicores": semicores,
            # 'pseudo_energy': pseudo_energy,
        }

        results[upf_data.element] = upf_dict

    with open(filename, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, sort_keys=True)

    print(f'Written to file "{filename}"')


if __name__ == "__main__":
    #  pylint: disable=invalid-name
    # pseudo_group_label = 'SSSP/1.1/PBE/efficiency'
    # out_file = 'SSSP_1.1_PBE_efficiency.yaml'

    # write_yaml(pseudo_group_label, out_file)

    # from aiida_wannier90_workflows.utils.workflows.group import standardize_groupname
    # pseudo_group_label = 'PseudoDojo/0.4/PBE/SR/standard/upf'
    # pseudo_group_label = "PseudoDojo/0.4/LDA/SR/standard/upf"
    pseudo_group_label = "PseudoDojo/0.4/PBE/FR/standard/upf"
    # out_file = standardize_groupname(group_label) + '.json'
    # out_file = pseudo_group_label.replace("/", "_") + ".json"
    out_file = pseudo_group_label.replace("/", "_") + ".yaml"
    write_yaml_soc(pseudo_group_label, out_file)
