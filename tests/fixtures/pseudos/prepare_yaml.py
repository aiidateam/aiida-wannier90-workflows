#!/usr/bin/env runaiida
# -*- coding: utf-8 -*-
"""Prepare meta data for SSSP."""
import yaml

from aiida import orm
from aiida_wannier90_workflows.utils.pseudo.upf import (
    get_number_of_electrons_from_upf, get_number_of_projections_from_upf, get_upf_content, parse_pswfc_nosoc
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
        pswfc = [_['l'] for _ in pswfc]

        upf_dict = {
            'filename': upf_data.filename,
            'md5': upf_data.md5,
            'has_so': 'F',
            'z_valence': get_number_of_electrons_from_upf(upf_data),
            'number_of_wfc': get_number_of_projections_from_upf(upf_data),
            'pswfc': pswfc
        }

        results[upf_data.element] = upf_dict

    with open(filename, 'w') as file:
        yaml.dump(results, file)

    print(f'Written to file "{filename}"')


if __name__ == '__main__':
    SSSP_GROUP_LABEL = 'SSSP/1.1/PBE/efficiency'
    OUT_FILE = 'SSSP_1.1_PBE_efficiency.yaml'

    write_yaml(SSSP_GROUP_LABEL, OUT_FILE)
