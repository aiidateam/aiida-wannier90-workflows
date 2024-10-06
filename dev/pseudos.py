#!/usr/bin/env python
"""Generate the semicore JSON for a given pseudo family."""
from collections import OrderedDict
import hashlib
from importlib.resources import path
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
    """Generate the semicore dict for a given pseudo potential."""
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


def cli(pseudo_family: str, semicore_threshold: float = -1.8):
    """Generate the semicore JSON for a given pseudo family."""
    pseudo_group = orm.load_group(pseudo_family)

    semicore_dict = {}

    for pseudo in track(
        pseudo_group.pseudos.values(),
        description=f"Generating semicore dict for {pseudo_family}",
    ):
        semicore_dict.update(generate_semicore_dict(pseudo, semicore_threshold))

    with path(semicore, f"{pseudo_family.replace('/', '_')}.json") as semicore_path:
        with semicore_path.open("w", encoding="utf8") as handle:
            json.dump(OrderedDict(sorted(semicore_dict.items())), handle, indent=4)

        print("[bold green]Success:[/] the semicore JSON has been generated at:\n")
        print(f"  {semicore_path.absolute()}\n")


if __name__ == "__main__":
    typer.run(cli)
