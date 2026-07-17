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
) -> ty.Optional[float]:
    """Get a Fermi energy from an nscf run.

    Prefer the scf Fermi energy reported in the nscf stdout via the
    ``(compare with: ... computed in scf)`` marker. When that marker is absent
    (see the fallback below), return the nscf's own Fermi energy instead, taken
    from the parsed ``output_parameters``. The two can differ; the scf value is
    kept as the first choice to preserve existing behaviour.

    :param calc_nscf: a nscf PwBaseWorkChain or PwCalculation
    :type calc_nscf: ty.Union[PwBaseWorkChain, PwCalculation]
    :return: the scf Fermi energy if the stdout marker is present, otherwise the
        nscf Fermi energy from ``output_parameters``, else None. Unit is eV.
    :rtype: float, None
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

    if fermi_energy is None:
        # The regex above only matches the single-value "(compare with: X eV,
        # computed in scf)" marker. QE prints that marker only for a metallic
        # (smearing/tetrahedra), single-Fermi-energy nscf run: insulators print
        # HOMO/LUMO instead, and constrained-magnetization runs print a
        # two-value variant the regex does not match (see QE
        # PW/src/print_ks_energies.f90). In those cases fall back to the Fermi
        # energy the parser stored from this nscf run.
        output_parameters = calc_nscf.outputs.output_parameters.get_dict()
        # The aiida-quantumespresso parser always stores Fermi energies in eV,
        # but guard on the units regardless, to mirror `get_fermi_energy` and
        # avoid silently returning a value in the wrong unit.
        if output_parameters.get("fermi_energy_units") == "eV":
            fermi_energy = output_parameters.get("fermi_energy")
            if fermi_energy is None:
                # Spin-polarised runs with a constrained total magnetization
                # report one Fermi level per channel and have no single chemical
                # potential. Take the higher of the two as a conservative
                # reference for the (frozen) energy windows.
                up = output_parameters.get("fermi_energy_up")
                down = output_parameters.get("fermi_energy_down")
                if up is not None and down is not None:
                    fermi_energy = max(up, down)

    return fermi_energy
