# Extend a existed aiida-pseudo pseudo family
from aiida_wannier90_workflows.utils.projectors.fit_hydrogenics import fit_rsq_projector, r_hydrogenic, fit_ortho_projectors
from aiida_wannier90_workflows.utils.projectors.projectors import newProjectors, newProjector
from aiida_wannier90_workflows.utils.projectors.upfdict import newUPFDict

from aiida import orm, load_profile
import aiida_wannier90_workflows

import os
import re
from pathlib import Path
import json

aww_path = Path(aiida_wannier90_workflows.__file__).parent.resolve()

def main():
    load_profile()
    projectors = {}
    # Generate a list of required atomic orbitals
    with open(aww_path/"utils/projectors/required_orbitals.json", "r") as fp:
        required_orbital_list = json.load(fp)
    ###-> Choose the pseudo family (aiida-pseudo family) <-###
    upfs = orm.load_group('PseudoDojo/0.4/PBE/FR/standard/upf')
    with open(
        aww_path/
        "utils/pseudo/data/semicore/"
        ###-> Choose the semicore list of the selected package <-###
        "PseudoDojo_0.4_PBE_FR_standard_upf.json",
        "r"
    ) as fin:
        pswfc = json.load(fin)
    
    for element in pswfc:
        pswfc[element]["additional"] = []
        if not element in required_orbital_list:
            continue
        else:
            for ao in required_orbital_list[element]:
                if ao not in pswfc[element]["pswfcs"]:
                    pswfc[element]["additional"].append(ao)
    # If you want to add orbitals to specific element:
    # pswfc["P"].update({"additional": ["3D"]})
    # pswfc["Ba"].update({"additional": ["5D", "6P"]})

    # Use fit_projector to get fine alfa, and add additional Projector
    str2l = {"s": 0, "p": 1, "d": 2, "f": 3}
    ###-> Set OpenMX location <-###
    pao_path = Path("/Users/yuhao/Softwares/openmx3.9/DFT_DATA19/PAO/")
    for upf in upfs.nodes:
        element = upf.element
        print(element)

        proj_ele = []
        if element not in pswfc.keys():
            continue
        upfdict = newUPFDict.from_str(upf.get_content())
        proj = upfdict.to_projectors()
        try:
            proj[0].j
        except AttributeError:
            spin_orbit = False
        else:
            spin_orbit = True
        # Add additional projectors
        # Create a directory to store projectors
        os.makedirs(aww_path/'utils/projectors/external_projector/', exist_ok=True)
        for addit_orb in pswfc[element]["additional"]:
            print(addit_orb)
            l = str2l[addit_orb[1]]
            n = len([_ for _ in pswfc[element]["pswfcs"] if addit_orb[1] in _])
            if n == 0:
                # no inner shell found, can only find orbitals from third-party PAO library
                pao_file = pao_path / [_ for  _ in os.listdir(pao_path) if re.match(f"{element}[0-9]*\.0.*\.pao", _)][0]
                pao = newProjectors.from_pao(pao_file, n, l)[0]
                alfa = fit_rsq_projector(pao, n)
                print(pao_file,":", element, n, l, alfa)
                x = proj[0].x
                r = proj[0].r
                y = r_hydrogenic(r, l, n, alfa)
                if spin_orbit:
                    proj.add_projector_soc(newProjector(x, y, l, label=addit_orb, alfa=alfa))
                else:
                    proj.add_projector(newProjector(x, y, l, label=addit_orb, alfa=alfa))
            else:
                if not spin_orbit:
                    ref = None
                    for p in proj:
                        if (int(p.label[0]) == int(addit_orb[0])-1) and (p.label[1].lower() == addit_orb[1].lower()):
                            ref = p
                else: # spin_orbit
                    ref = []
                    for p in proj:
                        if (int(p.label[0]) == int(addit_orb[0])-1) and (p.label[1].lower() == addit_orb[1].lower()):
                            ref.append(p)

                if ref is None:
                    raise ValueError(f"Cant find inner projectors for {addit_orb}")
                if isinstance(ref, list):
                    if not len(ref) in [1, 2]:
                        raise ValueError(f"Wrong inner projectors for {addit_orb}, found {len(ref)}")
                print("fit from pao ortho")
                if spin_orbit:
                    for ref_ in ref:
                        alfa = fit_ortho_projectors(ref_, n, element)
                        x = proj[0].x
                        r = proj[0].r
                        y = r_hydrogenic(r, l, n, alfa)
                        proj.add_projector(newProjector(x, y, l, j=ref_.j, label=addit_orb, alfa=alfa))
                else:
                    alfa = fit_ortho_projectors(ref, n, element)
                    x = proj[0].x
                    r = proj[0].r
                    y = r_hydrogenic(r, l, n, alfa)
                    proj.add_projector(newProjector(x, y, l, label=addit_orb, alfa=alfa))


        proj.to_file(aww_path/f"utils/projectors/external_projector/{element}.dat")
        for projector in proj:
            if spin_orbit:
                proj_ele.append(
                    {"label": projector.label, "l": projector.l, "j": projector.j, "alfa": projector.alfa}
                )
            else:
                proj_ele.append(
                    {"label": projector.label, "l": projector.l, "alfa": projector.alfa}
                )
        projectors[element] = proj_ele
        with open(aww_path/"utils/projectors/external_projector/projectors.json", "w") as fp:
            # Output .dat file
            json.dump(projectors, fp, indent=2)



if __name__ == "__main__":
    main()
