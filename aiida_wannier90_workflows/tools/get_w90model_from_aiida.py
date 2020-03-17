import pythtb
import tempfile
import os
import shutil
from aiida_wannier90.parsers import raw_wout_parser


def get_pythtb(wan_calculation, xyz_workaround=False):
    tmpdir = tempfile.mkdtemp()

    seedname = os.path.splitext(wan_calculation._INPUT_FILE)[0]
    out_dir = os.path.join(
        wan_calculation.out.retrieved.folder.abspath, "path"
    )
    for filepath in [
        os.path.join(
            wan_calculation.folder.abspath, "raw_input",
            "{}.win".format(seedname)
        ),
        os.path.join(out_dir, "{}.wout".format(seedname)),
        os.path.join(out_dir, "{}_hr.dat".format(seedname)),
        os.path.join(out_dir, "{}_wsvec.dat".format(seedname)),
        os.path.join(out_dir, "{}_centres.xyz".format(seedname)),
        os.path.join(out_dir, "{}_band.kpt".format(seedname)),
        os.path.join(out_dir, "{}_band.dat".format(seedname)),
    ]:
        if os.path.exists(filepath):
            shutil.copy(
                filepath, os.path.join(tmpdir, os.path.basename(filepath))
            )
        else:
            print("Skipping {}".format(os.path.basename(filepath)))

    if xyz_workaround:
        parsed = raw_wout_parser(
            open(os.path.join(out_dir, "{}.wout".format(seedname))).readlines()
        )
        num_atoms = len(wan_calculation.inp.structure.sites)
        num_wf = len(parsed['wannier_functions_output'])

        xyz = []
        xyz.append(str(num_atoms + num_wf))
        xyz.append(str(num_atoms + num_wf))
        for wf in parsed['wannier_functions_output']:
            xyz.append(
                "X {} {} {}".format(
                    wf['coordinates'][0], wf['coordinates'][1],
                    wf['coordinates'][2]
                )
            )
        for site in wan_calculation.inp.structure.sites:
            xyz.append(
                "{} {} {} {}".format(
                    site.kind_name, site.position[0], site.position[1],
                    site.position[2]
                )
            )
        with open(
            os.path.join(tmpdir, "{}_centres.xyz".format(seedname)), 'w'
        ) as f:
            f.write("\n".join(xyz))

    tb_w90 = pythtb.w90(tmpdir, seedname)
    # Remove the folder
    shutil.rmtree(tmpdir)

    return tb_w90


if __name__ == "__main__":
    c = load_node(16515)
    tb_w90 = get_pythtb(c, xyz_workaround=True)
    print(tb_w90.model())
