"""Utility functions for processing pw.x related workchains."""

import typing as ty

from aiida import orm

from aiida_quantumespresso.calculations.pw import PwCalculation
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain


def get_fermi_energy(output_parameters: orm.Dict) -> ty.Optional[float]:
    """Get Fermi energy from scf output parameters.

    :param output_parameters: scf output parameters
    :type output_parameters: orm.Dict
    :return: if found return Fermi energy, else None. Unit is eV.
    :rtype: float, None
    """
    out_dict = output_parameters.get_dict()
    fermi = out_dict.get("fermi_energy", None)
    fermi_units = out_dict.get("fermi_energy_units", None)

    if fermi_units != "eV":
        return None

    return fermi


def get_fermi_energy_from_nscf(
    calc_nscf: ty.Union[PwBaseWorkChain, PwCalculation]
) -> float:
    """Parse nscf output to get the scf Fermi energy.

    :param calc_nscf: a nscf PwBaseWorkChain or PwCalculation
    :type calc_nscf: ty.Union[PwBaseWorkChain, PwCalculation]
    :return: scf Fermi energy
    :rtype: float
    """
    import re

    from aiida_wannier90_workflows.utils.workflows import get_last_calcjob

    valid_inputs = (PwBaseWorkChain, PwCalculation)
    if calc_nscf.process_class not in valid_inputs:
        raise ValueError(f"Only support {valid_inputs}, input is {calc_nscf}")

    if not calc_nscf.is_finished_ok:
        raise ValueError(f"Input {calc_nscf} has not finished successfully")

    if calc_nscf.process_class == PwBaseWorkChain:
        calc_nscf = get_last_calcjob(calc_nscf)

    if calc_nscf.process_class != PwCalculation:
        raise ValueError(f"Input {calc_nscf} is not a PwCalculation")

    out = calc_nscf.outputs.retrieved.get_object_content("aiida.out")
    lines = out.split("\n")

    # QE 6.8 output scf Fermi energy in nscf run:
    #  the Fermi energy is     5.9816 ev
    #  (compare with:     5.9034 eV, computed in scf)
    fermi_energy = None
    regex = re.compile(
        r"\s*\(compare with:\s*([+-]?(?:[0-9]+(?:[.][0-9]*)?|[.][0-9]+))\s*eV, computed in scf\)"
    )
    for line in lines:
        match = regex.match(line)
        if match:
            fermi_energy = float(match.group(1))
            break

    return fermi_energy
