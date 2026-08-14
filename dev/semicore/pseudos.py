#!/usr/bin/env python
"""Generate the semicore JSON for a given pseudo family."""
from collections import OrderedDict
import hashlib
from importlib.resources import files
import json

from aiida_pseudo.data.pseudo.upf import UpfData
from rich import print  # pylint: disable=redefined-builtin
from rich.progress import track
import typer
from upf_tools import UPFDict

from aiida import load_profile, orm

from aiida_wannier90_workflows.utils.pseudo.data import semicore

load_profile()


def generate_semicore_dict(pseudo: UpfData, semicore_threshold=-1.8) -> dict:
    """Generate the semicore dict for a given pseudo potential.

    Fails if there is no `pseudo_energy` in the UPF file.
    In this case, use the `generate_semicore_dict_from_orbitals` function instead.
    """
    upf = UPFDict.from_str(pseudo.get_content())
    filename = pseudo.filename
    md5_hash = hashlib.md5(pseudo.get_content().encode()).hexdigest()

    label_energy_dict = {
        wfc["label"]: wfc["pseudo_energy"] for wfc in upf["pswfc"]["chi"]
    }

    threshold_modifier = {
        "2S": -1.0,
        "3P": 0.5,
        "3D": 0.4,
        "4P": 0.7,
        "4D": 0.5,
        "5P": 0.9,
        "5D": 0.8,
    }
    return {
        upf["header"]["element"].rstrip(): {
            "filename": filename,
            "md5": md5_hash,
            "pswfcs": list(label_energy_dict.keys()),
            "semicores": [
                label
                for label, energy in label_energy_dict.items()
                if energy < semicore_threshold + threshold_modifier.get(label, 0)
            ],
        }
    }


def generate_semicore_dict_from_orbitals(pseudo: UpfData) -> dict:
    """Generate the semicore dict based on the orbitals in the pseudo.

    Take into account the element's position in the periodic table.
    More robust way to determine semicores, works also for UPF version 1.
    """
    from ase.data import atomic_numbers

    # P-block elements
    pblock = (
        list(range(31, 37))
        + list(range(49, 55))
        + list(range(81, 87))
        + list(range(113, 119))
    )

    upf = UPFDict.from_str(pseudo.get_content())
    filename = pseudo.filename
    md5_hash = hashlib.md5(pseudo.get_content().encode()).hexdigest()
    znum = atomic_numbers[upf["header"]["element"].strip()]
    pswfcs_list = []
    for wfc in upf["pswfc"]["chi"]:
        # Calculate the rank based on the atomic number (Z) and orbital type
        # for S and P orbitals of elements in P-block we add 1 to the rank,
        # for D and F orbitals we add 1 and 2 respectively (l-1)
        rank = int(wfc["label"][0])  # Extract the main quantum number
        l = int(wfc["l"])
        if znum in pblock and l in (0, 1):
            rank += 1
        elif l > 1:
            rank += l - 1

        pswfcs_list.append(
            {
                "label": wfc["label"],
                "rank": rank,
            }
        )
    return {
        upf["header"]["element"].strip(): {
            "filename": filename,
            "md5": md5_hash,
            "pswfcs": [wfc["label"] for wfc in pswfcs_list],
            "semicores": [
                wfc["label"]
                for wfc in pswfcs_list
                if wfc["rank"] < max(wfc["rank"] for wfc in pswfcs_list)
            ],
        }
    }


def generate_semicore_dict_auto(
    pseudo: UpfData, method: str = "energy_threshold", semicore_threshold: float = -1.8
) -> dict:
    """Generate the semicore dict for a given pseudo potential using the specified method."""
    if method == "energy_threshold":
        return generate_semicore_dict(pseudo, semicore_threshold)
    if method == "orbitals":
        return generate_semicore_dict_from_orbitals(pseudo)
    raise ValueError(
        f"Unknown method: {method}\nAvailable methods: 'energy_threshold', 'orbitals'."
    )


def cli(
    pseudo_family: str,
    method: str = "energy_threshold",
    semicore_threshold: float = -1.8,
):
    """Generate the semicore JSON for a given pseudo family."""
    pseudo_group = orm.load_group(pseudo_family)

    semicore_dict = {}

    for pseudo in track(
        pseudo_group.pseudos.values(),
        description=f"Generating semicore dict for {pseudo_family}",
    ):
        semicore_dict.update(
            generate_semicore_dict_auto(pseudo, method, semicore_threshold)
        )

    semicore_path = files(semicore).joinpath(f"{pseudo_family.replace('/', '_')}.json")
    with open(semicore_path, "w", encoding="utf8") as handle:
        json.dump(OrderedDict(sorted(semicore_dict.items())), handle, indent=4)
        handle.write("\n")

    print("[bold green]Success:[/] the semicore JSON has been generated at:\n")
    print(f"  {semicore_path.absolute()}\n")


if __name__ == "__main__":
    typer.run(cli)
