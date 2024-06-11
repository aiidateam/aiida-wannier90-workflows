"""Functions to guess Wannier90 projections."""

import typing as ty

from aiida_quantumespresso.common.types import ElectronicType

from aiida_wannier90_workflows.common.types import (
    WannierDisentanglementType,
    WannierFrozenType,
    WannierProjectionType,
)


def guess_wannier_projection_types(
    electronic_type: ElectronicType,
    projection_type: WannierProjectionType = None,
    disentanglement_type: WannierDisentanglementType = None,
    frozen_type: WannierFrozenType = None,
) -> ty.Tuple[WannierProjectionType, WannierDisentanglementType, WannierFrozenType]:
    """Automatically guess Wannier projection, disentanglement, and frozen types."""
    # pylint: disable=too-many-branches

    if electronic_type == ElectronicType.INSULATOR:
        if disentanglement_type is None:
            disentanglement_type = WannierDisentanglementType.NONE
        elif disentanglement_type == WannierDisentanglementType.NONE:
            pass
        else:
            raise ValueError(
                "For insulators there should be no disentanglement, "
                f"current disentanglement type: {disentanglement_type}"
            )
        if frozen_type is None:
            frozen_type = WannierFrozenType.NONE
        elif frozen_type == WannierFrozenType.NONE:
            pass
        else:
            raise ValueError(
                f"For insulators there should be no frozen states, current frozen type: {frozen_type}"
            )
    elif electronic_type == ElectronicType.METAL:
        if projection_type == WannierProjectionType.SCDM:
            if disentanglement_type is None:
                # No disentanglement when using SCDM, otherwise the wannier interpolated bands are wrong
                disentanglement_type = WannierDisentanglementType.NONE
            elif disentanglement_type == WannierDisentanglementType.NONE:
                pass
            else:
                raise ValueError(
                    "For SCDM there should be no disentanglement, "
                    f"current disentanglement type: {disentanglement_type}"
                )
            if frozen_type is None:
                frozen_type = WannierFrozenType.NONE
            elif frozen_type == WannierFrozenType.NONE:
                pass
            else:
                raise ValueError(
                    f"For SCDM there should be no frozen states, current frozen type: {frozen_type}"
                )
        elif projection_type in [
            WannierProjectionType.ANALYTIC,
            WannierProjectionType.RANDOM,
        ]:
            if disentanglement_type is None:
                disentanglement_type = WannierDisentanglementType.SMV
            if frozen_type is None:
                frozen_type = WannierFrozenType.ENERGY_FIXED
            if (
                disentanglement_type == WannierDisentanglementType.NONE
                and frozen_type != WannierFrozenType.NONE
            ):
                raise ValueError(
                    f"Disentanglement is explicitly disabled but frozen type {frozen_type} is required"
                )
        elif projection_type in [
            WannierProjectionType.ATOMIC_PROJECTORS_QE,
            WannierProjectionType.ATOMIC_PROJECTORS_EXTERNAL,
        ]:
            if disentanglement_type is None:
                disentanglement_type = WannierDisentanglementType.SMV
            if frozen_type is None:
                frozen_type = WannierFrozenType.FIXED_PLUS_PROJECTABILITY
            if (
                disentanglement_type == WannierDisentanglementType.NONE
                and frozen_type != WannierFrozenType.NONE
            ):
                raise ValueError(
                    f"Disentanglement is explicitly disabled but frozen type {frozen_type} is required"
                )
        else:
            if disentanglement_type is None or frozen_type is None:
                raise ValueError(
                    "Cannot automatically guess disentanglement and frozen types "
                    f"from projection type: {projection_type}"
                )
    else:
        raise ValueError(f"Not supported electronic type {electronic_type}")

    return projection_type, disentanglement_type, frozen_type
