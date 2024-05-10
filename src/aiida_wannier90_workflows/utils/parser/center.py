#!/usr/bin/env python
"""Processthe Wannier function centers."""
import typing as ty

import numpy as np

from aiida import orm

from aiida_wannier90.calculations import Wannier90Calculation


def get_wf_centers(calculation: Wannier90Calculation, initial: bool = False) -> tuple:
    """Get Wannier function centers.

    :param calculation: A finished ``Wannier90Calculation``.
    :type calculation: Wannier90Calculation
    :param initial: Get initial or final WF center.
    :type initial: bool
    :return: ``cell``, ``atoms``, ``wf_centers``. ``atoms`` are the atomic positions,
    both ``atoms`` and ``wf_centers`` are in Cartesian coordinates and are translated back into the cell.
    :rtype: tuple
    """
    import ase

    structure = calculation.inputs.structure

    if initial:
        wf_outputs_key = "wannier_functions_initial"
    else:
        wf_outputs_key = "wannier_functions_output"

    wf_outputs = calculation.outputs.output_parameters[wf_outputs_key]
    wf_centers = np.zeros(shape=(len(wf_outputs), 3))
    for wf in wf_outputs:  # pylint: disable=invalid-name
        wf_id = wf["wf_ids"] - 1
        wf_centers[wf_id] = wf["wf_centres"]

    structure_ase = structure.get_ase()
    cell = structure_ase.get_cell()
    # Transform all coordinates into Cartesian, translate atoms to the cell at origin
    atoms = structure_ase.get_positions(wrap=True)

    # ase Atoms for representing Wannier function centers
    wf_ase = ase.Atoms()
    wf_ase.set_cell(cell)
    wf_ase.set_pbc(True)
    for wf in wf_centers:  # pylint: disable=invalid-name
        # position is in angstrom unit
        atom = ase.Atom("H", position=wf)
        wf_ase.append(atom)
    # Transform all coordinates into Cartesian, translate Wannier function to the cell at origin
    wf_centers = wf_ase.get_positions(wrap=True)

    return cell, atoms, wf_centers


def generate_supercell(
    cell: np.array, size: ty.Union[int, list, np.array] = 2
) -> ty.Tuple[np.array, np.array]:
    """Generate a supercell for finding nearest neighbours.

    :param cell: each row is a lattice vector
    :param size: number of repetitions = 2*size + 1, defaults to 2
    :return: supercell and the translation index
    """
    # Generate a supercell (2D: square, 3D: cube).
    # If the angles between lattice vectors are small, a 3*3*3 supercell is not enough
    # for finding minimum distance.

    # Handle both list & np.array
    dimension = len(cell[0])
    if dimension not in (2, 3):
        raise NotImplementedError(
            f"Only support dimension of 2 or 3, input dimension is {dimension}"
        )

    a1 = np.array(cell[0])  # pylint: disable=invalid-name
    a2 = np.array(cell[1])  # pylint: disable=invalid-name
    if dimension == 3:
        a3 = np.array(cell[2])  # pylint: disable=invalid-name

    if isinstance(size, int):
        size = [size for _ in range(dimension)]

    supercell_range0 = range(-size[0], size[0] + 1)
    supercell_range1 = range(-size[1], size[1] + 1)
    if dimension == 2:
        num_pts = len(supercell_range0) * len(supercell_range1)
    elif dimension == 3:
        supercell_range2 = range(-size[2], size[2] + 1)
        num_pts = len(supercell_range0) * len(supercell_range1) * len(supercell_range2)
    supercell = np.zeros((num_pts, dimension))
    supercell_translations = np.zeros_like(supercell, dtype=int)

    if dimension == 2:
        counter = 0
        for i in supercell_range0:
            for j in supercell_range1:
                x, y = i * a1 + j * a2
                supercell[counter, :] = [x, y]
                supercell_translations[counter, :] = [i, j]
                counter += 1
    elif dimension == 3:
        counter = 0
        for i in supercell_range0:
            for j in supercell_range1:
                for k in supercell_range2:
                    x, y, z = i * a1 + j * a2 + k * a3
                    supercell[counter, :] = [x, y, z]
                    supercell_translations[counter, :] = [i, j, k]
                    counter += 1

    return supercell, supercell_translations


def find_wf_nearest_atom(
    cell: np.array,
    atoms: np.array,
    wf_centers: np.array,
    *,
    nth_neighbour: int = 1,
) -> ty.Tuple[np.array, np.array]:
    """Find the nearest atom for each Wannier function center.

    :param cell: each row is a lattice vector
    :type cell: np.array, 3 x 3
    :param atoms: atomic positions, in Cartesian coordinates.
    :type atoms: np.array, num_atoms x 3
    :param wf_centers: Wannier function centers, in Cartesian coordinates.
    :type wf_centers: np.array, num_wf x 3
    :param nth_neighbour: Get 1st, 2nd, ... nth neighbouring atom.
    :type nth_neighbour: int >= 1
    :return: nearest atom distance, nearest atom index.
    nearest atom distance: num_wf x 3
    nearest atom index: num_wf x 4, 0th column is the atom index of ``atoms``,
    1-3th columns are the cell translation for this atom (the equivalent atom (with this
    translation applied) is the atom which is nearest to the Wannier function).
    :rtype: tuple[np.array, np.array]
    """
    from scipy.spatial import cKDTree

    if not isinstance(nth_neighbour, int) and nth_neighbour < 1:
        raise ValueError(f"nth_neighbour {nth_neighbour} not integer or < 1")

    num_atoms = atoms.shape[0]

    supercell, supercell_translations = generate_supercell(cell)
    num_supercell, dimension = supercell.shape

    # Generate a supercell of atoms
    supercell_with_atoms = np.zeros((num_atoms * num_supercell, dimension))
    supercell_translation_with_atoms = np.zeros(
        (num_atoms * num_supercell, dimension + 1), dtype=int
    )
    for iatom in range(num_atoms):
        idx_start = iatom * num_supercell
        idx_stop = (iatom + 1) * num_supercell
        supercell_with_atoms[idx_start:idx_stop, :] = supercell + atoms[iatom, :]
        # 0th: atom index
        supercell_translation_with_atoms[idx_start:idx_stop, 0] = iatom
        # 1-3th: supercell translation
        supercell_translation_with_atoms[idx_start:idx_stop, 1:] = (
            supercell_translations
        )

    # KD tree for to find nearest neighbours
    kdtree = cKDTree(supercell_with_atoms)
    # neighbour_distance, neighbour_indexes = kdtree.query(wf_centers, k=1)
    neighbour_distance, neighbour_indexes = kdtree.query(wf_centers, k=[nth_neighbour])
    # print(f"{neighbour_distance = } {neighbour_indexes = }")
    neighbour_distance = neighbour_distance.flatten()
    neighbour_indexes = neighbour_indexes.flatten()
    num_centers = len(wf_centers)

    neighbour_atom = np.zeros(
        (num_centers, supercell_translation_with_atoms.shape[1]), dtype=int
    )
    for i in range(num_centers):
        neighbour_atom[i] = supercell_translation_with_atoms[neighbour_indexes[i]]

    return neighbour_distance, neighbour_atom


def get_wf_center_distances(
    calculation: Wannier90Calculation,
    *,
    nth_neighbour: int = 1,
    initial: bool = False,
) -> tuple:
    """Calculate distances between Wannier function centers and its nearest neighbour.

    :param calculation: a ``Wannier90Calculation``.
    :type calculation: Wannier90Calculation
    :param nth_neighbour: Get 1st, 2nd, ... nth neighbouring atom.
    :type nth_neighbour: int >= 1
    :param initial: Get initial or final WF center.
    :type initial: bool
    :return: distance, nearest_atom, cell_translation, structure_ase
    :rtype: tuple
    """
    if nth_neighbour < 1:
        raise ValueError(f"nth_neighbour {nth_neighbour} < 1")

    cell, atoms, wf_centers = get_wf_centers(calculation, initial=initial)

    distance, atom_translation = find_wf_nearest_atom(
        cell, atoms, wf_centers, nth_neighbour=nth_neighbour
    )

    nearest_atom = atom_translation[:, 0]
    cell_translation = atom_translation[:, 1:]
    structure_ase = calculation.inputs.structure.get_ase()

    return distance, nearest_atom, cell_translation, structure_ase


def get_wigner_seitz(cell: np.array, search_size: int = 2) -> np.array:
    """Get Wigner-Seitz cell.

    :param cell: each row is a lattice vector.
    :return: Wigner-Seitz cell
    """
    import itertools

    from scipy.spatial import Voronoi  # pylint: disable=no-name-in-module

    # Initially I tried to find the Wigner-Seitz cell the Wannier function belongs to,
    # and calculate the distance between Wannier function and the cell center.
    # Then get the minimum of the calculated distances (among all atomic positions in the WS cell),
    # thus we find the nearest atom of the Wannier function.
    # But later on I just use a supercell approach since that is easier to do.

    dimension = cell.shape[0]
    search_range = range(-search_size, search_size + 1)

    # points = []
    # for i, j, k in itertools.product(search_range, repeat=dimension):
    #     points.append(i * cell[0] + j * cell[1] + k * cell[2])

    # A bit faster
    points = np.array(list(itertools.product(search_range, repeat=dimension)))
    points = points @ cell

    vor = Voronoi(points)

    ws_cell = None
    for region in vor.regions:
        if len(region) != 0 and np.all(np.array(region) >= 0):
            ws_cell = np.zeros(shape=(len(region), 3))
            for i, reg in enumerate(region):
                ws_cell[i] = vor.vertices[reg]
            break
    # pylint: disable=fixme
    # TODO: There might be multiple closed regions, I need to ensure the region containing
    # the origin is returned.

    return ws_cell


def test_plot_voronoi():
    """Plot a test Voronoi diagram for a very oblique cell."""
    import matplotlib.pyplot as plt
    from scipy.spatial import (  # pylint: disable=no-name-in-module
        Voronoi,
        voronoi_plot_2d,
    )

    cell_angle = 10
    cell = np.array(
        [[1, 0], [np.cos(cell_angle / 180 * np.pi), np.sin(cell_angle / 180 * np.pi)]]
    )

    supercell, _ = generate_supercell(cell)

    vor = Voronoi(supercell)
    voronoi_plot_2d(vor)

    plt.axis("equal")
    plt.show()


def test_find_nearest():
    """Test the function."""
    # cell, atoms, wf_centers = get_wf_centers(orm.load_node(124310))
    cell = np.array(
        [
            [0.0, 2.6988037626031, 2.6988037626031],
            [2.6988037626031, 0.0, 2.6988037626031],
            [2.6988037626031, 2.6988037626031, 0.0],
        ]
    )
    atoms = np.array(
        [
            [1.34940188, 1.34940188, 1.34940188],
            [0.0, 0.0, 0.0],
        ]
    )
    wf_centers = np.array(
        [
            [1.34939724e00, 1.34939400e00, 1.34937924e00],
            [2.69880776e00, 2.69881176e00, 2.30000000e-05],
            [1.34940924e00, 1.34932000e00, 1.34946624e00],
            [1.34942924e00, 1.34935000e00, 1.34936224e00],
            [1.34932524e00, 1.34944400e00, 1.34935224e00],
            [2.69887476e00, 2.69885476e00, 5.39754153e00],
            [2.69877776e00, 7.30000000e-05, 2.69883976e00],
            [2.69880876e00, 2.69875976e00, 5.70000000e-05],
        ]
    )
    # wannier_function_coordinates = [[3.20056284, 3.20056284, 3.20056284]])
    distance, nearest_atoms = find_wf_nearest_atom(
        cell, atoms, wf_centers, nth_neighbour=2
    )

    print("==== cell ====")
    print(cell)
    print("==== atoms ====")
    print(atoms)
    print("==== wf_centers ====")
    print(wf_centers)
    print("==== distance ====")
    print(distance)
    print("==== nearest_atoms ====")
    print(nearest_atoms)


def get_last_wan_calc(
    node: ty.Union[orm.WorkChainNode, orm.CalcJobNode]
) -> Wannier90Calculation:
    """Get the last ``Wannier90Calculation`` of a workchain.

    :param node: [description]
    :type node: ty.Union[orm.WorkChainNode, orm.CalcJobNode]
    :return: [description]
    :rtype: Wannier90Calculation
    """
    from aiida.common.links import LinkType

    from aiida_wannier90_workflows.utils.workflows import get_last_calcjob
    from aiida_wannier90_workflows.workflows.bands import Wannier90BandsWorkChain
    from aiida_wannier90_workflows.workflows.base.wannier90 import (
        Wannier90BaseWorkChain,
    )
    from aiida_wannier90_workflows.workflows.open_grid import Wannier90OpenGridWorkChain
    from aiida_wannier90_workflows.workflows.optimize import Wannier90OptimizeWorkChain
    from aiida_wannier90_workflows.workflows.wannier90 import Wannier90WorkChain

    supported_workchains = (
        Wannier90BandsWorkChain,
        Wannier90OpenGridWorkChain,
        Wannier90WorkChain,
    )

    if isinstance(node, orm.WorkChainNode):
        if node.process_class == Wannier90OptimizeWorkChain:
            calc = node.outputs.wannier90_optimal.output_parameters.creator
        elif node.process_class in supported_workchains:
            calc = (
                node.base.links.get_outgoing(
                    link_type=(LinkType.CALL_CALC, LinkType.CALL_WORK),
                    link_label_filter="wannier90",
                )
                .one()
                .node
            )
            if calc.process_class == Wannier90BaseWorkChain:
                calc = get_last_calcjob(calc)
        elif node.process_class == Wannier90BaseWorkChain:
            calc = get_last_calcjob(node)
        else:
            raise ValueError(
                f"Supported WorkChain type {supported_workchains}, current WorkChain {node}"
            )
    elif (
        isinstance(node, orm.CalcJobNode) and node.process_class == Wannier90Calculation
    ):
        calc = node
    else:
        raise ValueError(f"Unsupported type {node}")

    return calc


def wf_center_distances_for_group(group: ty.Union[orm.Group, str, int]) -> np.array:
    """Calculate distance of Wannier function center to nearest atom for a group of WorkChain.

    :param group: [description]
    :type group: orm.Group, str, int
    :return: [description]
    :rtype: np.array
    """
    distances = []

    if not isinstance(group, orm.Group):
        group = orm.load_group(group)

    for node in group.nodes:
        if not node.is_finished_ok:
            print(f"Skip unfinished node: {node}")
            continue

        calc = get_last_wan_calc(node)
        dist, _, _, _ = get_wf_center_distances(calc)
        distances.extend(dist)

    distances = np.array(distances)

    return distances


def export_wf_centers_to_xyz(calculation: Wannier90Calculation, filename: str = None):
    """Export a XSF file to visualize Wannier function centers.

    :param calculation: [description]
    :type calculation: Wannier90Calculation
    """
    import ase

    structure = calculation.inputs.structure.get_ase()
    new_structure = structure.copy()

    _, _, wf_centers = get_wf_centers(calculation)

    for coord in wf_centers:
        new_structure.append(ase.Atom("X", coord))

    if not filename:
        filename = f"{structure.get_formula()}_{calculation.pk}_wf_centers.xyz"

    new_structure.write(filename)


def export_wf_centers_for_group(group: orm.Group, save_dir: str = "."):
    """Export Wannier function centers to XYZ file for a group of WorkChain.

    :param group: [description]
    :type group: orm.Group
    """
    from pathlib import Path

    for node in group.nodes:
        if not node.is_finished_ok:
            print(f"Skip unfinished node: {node}")
            continue

        calc = get_last_wan_calc(node)

        filename = f"{calc.inputs.structure.get_formula()}_{calc.pk}_wf_centers.xyz"
        filename = Path(save_dir) / filename

        export_wf_centers_to_xyz(calc, filename)


def plot_histogram(distances: np.array, title: str = None):
    """Plot a histogram of Wannier function centers to nearest atom distances.

    :param distances: [description]
    :type distances: np.array
    """
    import matplotlib.pyplot as plt

    plt.hist(distances, 100)

    plt.xlabel("distance(WF center, nearest atom) / Angstrom")
    plt.ylabel("Count")
    pre_title = f"Histogram of {len(distances)} WFs"
    if title is not None:
        full_title = f"{pre_title}, {title}"
        plt.title(full_title)
    plt.grid(True)
    plt.annotate(
        f"average = {np.average(distances):.4f}",
        xy=(0.7, 0.9),
        xycoords="axes fraction",
    )

    # plt.savefig('distances.png')
    plt.show()
