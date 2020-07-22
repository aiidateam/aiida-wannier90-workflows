#!/usr/bin/env runaiida
import argparse
import json
from aiida import orm
from aiida_wannier90_workflows.utils.upf import get_number_of_electrons_from_upf, get_number_of_projections_from_upf

if __name__ == '__main__':
    OUTPUT_FILENAME = 'sssp_nelec_nproj.json'

    parser = argparse.ArgumentParser(description=
    f'Parse number of electrons and number of projection orbitals of SSSP pseudopotentials. '
    'The script will read from a json file and search your AiiDA database for the pseudopotentials, '
    'and write to a file `{OUTPUT_FILENAME}` which contains the number of electrons '
    'and number of projection orbitals of SSSP pseudopotentials.')
    parser.add_argument('file', metavar='JSON_FILE', type=str, help='the json file of SSSP, e.g. sssp_efficiency_1.1.json.')
    args = parser.parse_args()

    with open(args.file) as f:
        sssp = json.load(f)

    results = {}
    for element in sssp:
        md5 = sssp[element]['md5']
        builder = orm.QueryBuilder()
        builder.append(orm.UpfData, filters={'attributes.md5': md5}, project=['uuid', 'attributes.element'])
        res = builder.all()
        if len(res) >= 1:
            this_uuid, this_element = res[0]
            this_upf = orm.load_node(this_uuid)
            if element == this_element:
                results[element] = dict(
                    filename=sssp[element]['filename'],
                    md5=md5,
                    num_elec=get_number_of_electrons_from_upf(this_upf),
                    num_proj=get_number_of_projections_from_upf(this_upf),
                )
        else:
            raise Exception('Upf of {} not found'.format(element))

    with open(OUTPUT_FILENAME, 'w') as f:
        json.dump(results, f, indent=2)

    print(f'written to file {OUTPUT_FILENAME}')