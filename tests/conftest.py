# pylint: disable=redefined-outer-name,too-many-statements
"""Initialise a text database and profile for pytest."""
from collections.abc import Mapping
import io
import pathlib
import shutil

import pytest

pytest_plugins = ["aiida.manage.tests.pytest_fixtures"]  # pylint: disable=invalid-name


@pytest.fixture(scope="session")
def filepath_tests():
    """Return the absolute filepath of the `tests` folder.

    .. warning:: if this file moves with respect to the `tests` folder, the implementation should change.

    :return: absolute filepath of `tests` folder which is the basepath for all test resources.
    """
    return pathlib.Path(__file__).resolve().parent


@pytest.fixture(scope="session")
def filepath_fixtures(filepath_tests):
    """Return the absolute filepath to the directory containing the file `fixtures`."""
    return filepath_tests / "fixtures"


@pytest.fixture(scope="function")
def fixture_sandbox():
    """Return a `SandboxFolder`."""
    from aiida.common.folders import SandboxFolder

    with SandboxFolder() as folder:
        yield folder


@pytest.fixture
def fixture_localhost(aiida_localhost):
    """Return a localhost `Computer`."""
    localhost = aiida_localhost
    localhost.set_default_mpiprocs_per_machine(1)
    return localhost


@pytest.fixture
def fixture_code(fixture_localhost):
    """Return a ``Code`` instance configured to run calculations of given entry point on localhost ``Computer``."""

    def _fixture_code(entry_point_name):
        from aiida.common import exceptions
        from aiida.orm import Code

        label = f"test.{entry_point_name}"

        try:
            return Code.objects.get(label=label)  # pylint: disable=no-member
        except exceptions.NotExistent:
            return Code(
                label=label,
                input_plugin_name=entry_point_name,
                remote_computer_exec=[fixture_localhost, "/bin/true"],
            )

    return _fixture_code


@pytest.fixture
def serialize_builder():
    """Serialize the given process builder into a dictionary with nodes turned into their value representation.

    :param builder: the process builder to serialize
    :return: dictionary
    """
    from aiida_wannier90_workflows.utils.workflows.builder.serializer import serialize

    def _serializer(node):
        return serialize(node, show_pk=False)

    return _serializer


@pytest.fixture(scope="session", autouse=True)
def pseudos(aiida_profile, generate_upf_data, generate_upf_data_soc):
    """Create pseudo potential families from scratch."""
    from aiida.common.constants import elements
    from aiida.plugins import GroupFactory

    aiida_profile.clear_profile()

    # Create an SSSP pseudo potential family from scratch.
    SsspFamily = GroupFactory("pseudo.family.sssp")

    stringency = "standard"
    label = "SSSP/1.3/PBEsol/efficiency"
    sssp = SsspFamily(label=label)
    sssp.store()

    cutoffs = {}
    upfs = []

    for values in elements.values():
        element = values["symbol"]
        if element in ["X"]:
            continue
        try:
            upf = generate_upf_data(element)
        except ValueError:
            continue

        upfs.append(upf)

        cutoffs[element] = {
            "cutoff_wfc": 30.0,
            "cutoff_rho": 240.0,
        }

    sssp.add_nodes(upfs)
    sssp.set_cutoffs(cutoffs, stringency, unit="Ry")

    # Create an pseudoDojo pseudo potential family from scratch.
    DojoFamily = GroupFactory("pseudo.family.pseudo_dojo")

    stringency = "standard"
    label = "PseudoDojo/0.4/PBE/FR/standard/upf"
    dojo = DojoFamily(label=label)
    dojo.store()

    cutoffs = {}
    upfs = []

    for values in elements.values():
        element = values["symbol"]
        if element in ["X"]:
            continue
        try:
            upf = generate_upf_data_soc(element)
        except ValueError:
            continue

        upfs.append(upf)

        cutoffs[element] = {
            "cutoff_wfc": 40.0,
            "cutoff_rho": 300.0,
        }

    dojo.add_nodes(upfs)
    dojo.set_cutoffs(cutoffs, stringency, unit="Ry")

    return sssp, dojo


@pytest.fixture(scope="session")
def generate_upf_data(filepath_fixtures):
    """Return a `UpfData` instance for the given element a file for which should exist in `tests/fixtures/pseudos`."""

    def _generate_upf_data(element):
        """Return `UpfData` node."""
        from aiida_pseudo.data.pseudo import PseudoPotentialData, UpfData
        import yaml

        yaml_file = filepath_fixtures / "pseudos" / "SSSP_1.1_PBE_efficiency.yaml"
        with open(yaml_file, encoding="utf-8") as file:
            upf_metadata = yaml.load(file, Loader=yaml.FullLoader)

        if element not in upf_metadata:
            raise ValueError(f"Element {element} not found in {yaml_file}")

        filename = upf_metadata[element]["filename"]
        md5 = upf_metadata[element]["md5"]
        z_valence = upf_metadata[element]["z_valence"]
        number_of_wfc = upf_metadata[element]["number_of_wfc"]
        has_so = upf_metadata[element]["has_so"]
        pswfc = upf_metadata[element]["pswfc"]
        ppchi = ""
        for i, l in enumerate(pswfc):  # pylint: disable=invalid-name
            ppchi += f'<PP_CHI.{i+1} l="{l}"/>\n'

        content = (
            '<UPF version="2.0.1">\n'
            "<PP_HEADER\n"
            f'element="{element}"\n'
            f'z_valence="{z_valence}"\n'
            f'has_so="{has_so}"\n'
            f'number_of_wfc="{number_of_wfc}"\n'
            "/>\n"
            "<PP_PSWFC>\n"
            f"{ppchi}"
            "</PP_PSWFC>\n"
            "</UPF>\n"
        )
        stream = io.BytesIO(content.encode("utf-8"))
        upf = UpfData(stream, filename=f"{filename}")

        # I need to hack the md5
        # upf.md5 = md5
        upf.base.attributes.set(upf._key_md5, md5)  # pylint: disable=protected-access
        # UpfData.store will check md5
        # `PseudoPotentialData` is the parent class of `UpfData`, this will skip md5 check
        super(PseudoPotentialData, upf).store()

        return upf

    return _generate_upf_data


@pytest.fixture(scope="session")
def generate_upf_data_soc(filepath_fixtures):
    """Return a `UpfData` instance for the given element a file for which should exist in `tests/fixtures/pseudos`."""

    def _generate_upf_data_soc(element):
        """Return `UpfData` node."""
        from aiida_pseudo.data.pseudo import PseudoPotentialData, UpfData
        import yaml

        yaml_file = (
            filepath_fixtures / "pseudos" / "PseudoDojo_0.4_PBE_FR_standard_upf.yaml"
        )
        with open(yaml_file, encoding="utf-8") as file:
            upf_metadata = yaml.load(file, Loader=yaml.FullLoader)

        if element not in upf_metadata:
            raise ValueError(f"Element {element} not found in {yaml_file}")

        filename = upf_metadata[element]["filename"]
        md5 = upf_metadata[element]["md5"]
        z_valence = upf_metadata[element]["z_valence"]
        number_of_wfc = upf_metadata[element]["number_of_wfc"]
        has_so = upf_metadata[element]["has_so"]
        ppspinorb = upf_metadata[element]["ppspinorb"]
        jchi = ppspinorb["jchi"]
        lchi = ppspinorb["lchi"]
        nn = ppspinorb["nn"]
        pprelwfc = ""

        for i, l in enumerate(lchi):  # pylint: disable=invalid-name
            pprelwfc += (
                f'<PP_RELWFC.{i+1} index="{i+1}" '
                f'lchi="{l}" '
                f'jchi="{jchi[i]}" '
                f'nn="{nn[i]}"/>\n'
            )

        content = (
            '<UPF version="2.0.1">\n'
            "<PP_HEADER\n"
            f'element="{element}"\n'
            f'z_valence="{z_valence}"\n'
            f'has_so="{has_so}"\n'
            f'number_of_wfc="{number_of_wfc}"\n'
            "/>\n"
            "<PP_SPIN_ORB>\n"
            f"{pprelwfc}"
            "</PP_SPIN_ORB>\n"
            "</UPF>\n"
        )
        stream = io.BytesIO(content.encode("utf-8"))
        upf = UpfData(stream, filename=f"{filename}")

        # I need to hack the md5
        # upf.md5 = md5
        upf.base.attributes.set(upf._key_md5, md5)  # pylint: disable=protected-access
        # UpfData.store will check md5
        # `PseudoPotentialData` is the parent class of `UpfData`, this will skip md5 check
        super(PseudoPotentialData, upf).store()

        return upf

    return _generate_upf_data_soc


@pytest.fixture(scope="session")
def get_sssp_upf():
    """Return a SSSP pseudo with a given element name."""

    def _get_sssp_upf(element):
        """Return SSSP pseudo."""
        from aiida.orm import QueryBuilder
        from aiida.plugins import GroupFactory

        SsspFamily = GroupFactory("pseudo.family.sssp")

        label = "SSSP/1.3/PBEsol/efficiency"
        pseudo_family = (
            QueryBuilder().append(SsspFamily, filters={"label": label}).one()[0]
        )

        return pseudo_family.get_pseudo(element=element)

    return _get_sssp_upf


@pytest.fixture(scope="session")
def get_dojo_upf():
    """Returen a pseudoDojo pseudo with a given element name."""

    def _get_dojo_upf(element):
        """Returen pesudoDojo pseudo."""
        from aiida.orm import QueryBuilder
        from aiida.plugins import GroupFactory

        DojoFamily = GroupFactory("pseudo.family.pseudo_dojo")

        label = "PseudoDojo/0.4/PBE/FR/standard/upf"
        pseudo_family = (
            QueryBuilder().append(DojoFamily, filters={"label": label}).one()[0]
        )

        return pseudo_family.get_pseudo(element=element)

    return _get_dojo_upf


@pytest.fixture
def generate_calc_job():
    """Fixture to construct a new `CalcJob` instance and call `prepare_for_submission` for testing `CalcJob` classes.

    The fixture will return the `CalcInfo` returned by `prepare_for_submission` and the temporary folder that was passed
    to it, into which the raw input files will have been written.
    """

    def _generate_calc_job(folder, entry_point_name, inputs=None):
        """Fixture to generate a mock `CalcInfo` for testing calculation jobs."""
        from aiida.engine.utils import instantiate_process
        from aiida.manage.manager import get_manager
        from aiida.plugins import CalculationFactory

        manager = get_manager()
        runner = manager.get_runner()

        process_class = CalculationFactory(entry_point_name)
        process = instantiate_process(runner, process_class, **inputs)

        calc_info = process.prepare_for_submission(folder)

        return calc_info

    return _generate_calc_job


@pytest.fixture
def generate_calc_job_node(fixture_localhost, filepath_fixtures):
    """Fixture to generate a mock `CalcJobNode` for testing parsers."""

    def flatten_inputs(inputs, prefix=""):
        """Flatten inputs recursively like :meth:`aiida.engine.processes.process::Process._flatten_inputs`."""
        flat_inputs = []
        for key, value in inputs.items():
            if isinstance(value, Mapping):
                flat_inputs.extend(flatten_inputs(value, prefix=prefix + key + "__"))
            else:
                flat_inputs.append((prefix + key, value))
        return flat_inputs

    def _generate_calc_job_node(  # pylint: disable=too-many-positional-arguments
        entry_point_name="base",
        computer=None,
        test_name=None,
        inputs=None,
        attributes=None,
        retrieve_temporary=None,
        store=True,
    ):
        """Fixture to generate a mock `CalcJobNode` for testing parsers.

        :param entry_point_name: entry point name of the calculation class
        :param computer: a `Computer` instance
        :param test_name: relative path of directory with test output files in the `fixtures/{entry_point_name}` folder.
        :param inputs: any optional nodes to add as input links to the corrent CalcJobNode
        :param attributes: any optional attributes to set on the node
        :param retrieve_temporary: optional tuple of an absolute filepath of a temporary directory and a list of
            filenames that should be written to this directory, which will serve as the `retrieved_temporary_folder`.
            For now this only works with top-level files and does not support files nested in directories.
        :return: `CalcJobNode` instance with an attached `FolderData` as the `retrieved` node.
        """
        from aiida import orm
        from aiida.common import LinkType
        from aiida.plugins.entry_point import format_entry_point_string

        if computer is None:
            computer = fixture_localhost

        filepath_folder = None

        if test_name is not None:
            for name in ("quantumespresso.", "wannier90."):
                if name in entry_point_name:
                    plugin_name = entry_point_name[len(name) :]
                    break
            filepath_folder = filepath_fixtures / "calcjob" / plugin_name / test_name
            filepath_input = filepath_folder / "aiida.in"

        entry_point = format_entry_point_string("aiida.calculations", entry_point_name)

        node = orm.CalcJobNode(computer=computer, process_type=entry_point)
        node.base.attributes.set("input_filename", "aiida.in")
        node.base.attributes.set("output_filename", "aiida.out")
        node.base.attributes.set("error_filename", "aiida.err")
        node.set_option("resources", {"num_machines": 1, "num_mpiprocs_per_machine": 1})
        node.set_option("max_wallclock_seconds", 1800)

        if attributes:
            node.base.attributes.set_many(attributes)

        if filepath_folder:
            from qe_tools.exceptions import ParsingError

            from aiida_quantumespresso.tools.pwinputparser import PwInputFile

            try:
                with open(filepath_input, encoding="utf-8") as input_file:
                    parsed_input = PwInputFile(input_file.read())
            except (ParsingError, FileNotFoundError):
                pass
            else:
                inputs["structure"] = parsed_input.get_structuredata()
                inputs["parameters"] = orm.Dict(parsed_input.namelists)

        if inputs:
            metadata = inputs.pop("metadata", {})
            options = metadata.get("options", {})

            for name, option in options.items():
                node.set_option(name, option)

            for link_label, input_node in flatten_inputs(inputs):
                input_node.store()
                node.base.links.add_incoming(
                    input_node, link_type=LinkType.INPUT_CALC, link_label=link_label
                )

        if store:
            node.store()

        if retrieve_temporary:
            dirpath, filenames = retrieve_temporary
            for filename in filenames:
                shutil.copy(
                    filepath_folder / filename, pathlib.Path(dirpath) / filename
                )

        if filepath_folder:
            retrieved = orm.FolderData()
            retrieved.put_object_from_tree(filepath_folder)

            # Remove files that are supposed to be only present in the retrieved temporary folder
            if retrieve_temporary:
                for filename in filenames:
                    retrieved.delete_object(filename)

            retrieved.base.links.add_incoming(
                node, link_type=LinkType.CREATE, link_label="retrieved"
            )
            retrieved.store()

            remote_folder = orm.RemoteData(computer=computer, remote_path="/tmp")
            remote_folder.base.links.add_incoming(
                node, link_type=LinkType.CREATE, link_label="remote_folder"
            )
            remote_folder.store()

        return node

    return _generate_calc_job_node


@pytest.fixture
def generate_structure():
    """Return a ``StructureData`` representing either bulk silicon or a water molecule."""

    def _generate_structure(structure_id="Si"):
        """Return a ``StructureData`` representing bulk silicon or a snapshot of a single water molecule dynamics.

        :param structure_id: identifies the ``StructureData`` you want to generate. Either 'Si' or 'H2O' or 'GaAs'.
        """
        from aiida.orm import StructureData

        if structure_id == "Si":
            param = 5.43
            cell = [
                [param / 2.0, param / 2.0, 0],
                [param / 2.0, 0, param / 2.0],
                [0, param / 2.0, param / 2.0],
            ]
            structure = StructureData(cell=cell)
            structure.append_atom(position=(0.0, 0.0, 0.0), symbols="Si", name="Si")
            structure.append_atom(
                position=(param / 4.0, param / 4.0, param / 4.0),
                symbols="Si",
                name="Si",
            )
        elif structure_id == "H2O":
            structure = StructureData(
                cell=[
                    [5.29177209, 0.0, 0.0],
                    [0.0, 5.29177209, 0.0],
                    [0.0, 0.0, 5.29177209],
                ]
            )
            structure.append_atom(
                position=[12.73464656, 16.7741411, 24.35076238], symbols="H", name="H"
            )
            structure.append_atom(
                position=[-29.3865565, 9.51707929, -4.02515904], symbols="H", name="H"
            )
            structure.append_atom(
                position=[1.04074437, -1.64320127, -1.27035021], symbols="O", name="O"
            )
        elif structure_id == "GaAs":
            structure = StructureData(
                cell=[
                    [0.0, 2.8400940897, 2.8400940897],
                    [2.8400940897, 0.0, 2.8400940897],
                    [2.8400940897, 2.8400940897, 0.0],
                ]
            )
            structure.append_atom(position=[0.0, 0.0, 0.0], symbols="Ga", name="Ga")
            structure.append_atom(
                position=[1.42004704485, 1.42004704485, 4.26014113455],
                symbols="As",
                name="As",
            )
        elif structure_id == "BaTiO3":
            structure = StructureData(
                cell=[
                    [3.93848606, 0.0, 0.0],
                    [0.0, 3.93848606, 0.0],
                    [0.0, 0.0, 3.93848606],
                ]
            )
            structure.append_atom(position=[0.0, 0.0, 0.0], symbols="Ba", name="Ba")
            structure.append_atom(
                position=[1.969243028987539, 1.969243028987539, 1.969243028987539],
                symbols="Ti",
                name="Ti",
            )
            structure.append_atom(
                position=[0.0, 1.969243028987539, 1.969243028987539],
                symbols="O",
                name="O",
            )
            structure.append_atom(
                position=[1.969243028987539, 1.969243028987539, 0.0],
                symbols="O",
                name="O",
            )
            structure.append_atom(
                position=[1.969243028987539, 0.0, 1.969243028987539],
                symbols="O",
                name="O",
            )
        else:
            raise KeyError(f"Unknown structure_id='{structure_id}'")
        return structure

    return _generate_structure


@pytest.fixture
def generate_kpoints_mesh():
    """Return a `KpointsData` node."""

    def _generate_kpoints_mesh(npoints):
        """Return a `KpointsData` with a mesh of npoints in each direction."""
        from aiida.orm import KpointsData

        kpoints = KpointsData()
        kpoints.set_kpoints_mesh([npoints] * 3)

        return kpoints

    return _generate_kpoints_mesh


@pytest.fixture(scope="session")
def generate_parser():
    """Fixture to load a parser class for testing parsers."""

    def _generate_parser(entry_point_name):
        """Fixture to load a parser class for testing parsers.

        :param entry_point_name: entry point name of the parser class
        :return: the `Parser` sub class
        """
        from aiida.plugins import ParserFactory

        return ParserFactory(entry_point_name)

    return _generate_parser


@pytest.fixture
def generate_remote_data():
    """Return a `RemoteData` node."""

    def _generate_remote_data(computer, remote_path, entry_point_name=None):
        """Return a `RemoteData`."""
        from aiida.common.links import LinkType
        from aiida.orm import CalcJobNode, RemoteData
        from aiida.plugins.entry_point import format_entry_point_string

        entry_point = format_entry_point_string("aiida.calculations", entry_point_name)

        remote = RemoteData(remote_path=remote_path)
        remote.computer = computer

        if entry_point_name is not None:
            creator = CalcJobNode(computer=computer, process_type=entry_point)
            creator.set_option(
                "resources", {"num_machines": 1, "num_mpiprocs_per_machine": 1}
            )
            remote.base.links.add_incoming(
                creator, link_type=LinkType.CREATE, link_label="remote_folder"
            )
            creator.store()

        return remote

    return _generate_remote_data


@pytest.fixture
def generate_bands_data():
    """Return a `BandsData` node."""

    def _generate_bands_data():
        """Return a `BandsData` instance with some basic `kpoints` and `bands` arrays."""
        import numpy as np

        from aiida.plugins import DataFactory

        BandsData = DataFactory("core.array.bands")  # pylint: disable=invalid-name
        bands_data = BandsData()

        bands_data.set_kpoints(np.array([[0.0, 0.0, 0.0], [0.625, 0.25, 0.625]]))

        bands_data.set_bands(
            np.array(
                [
                    [-5.64024889, 6.66929678, 6.66929678, 6.66929678, 8.91047649],
                    [-1.71354964, -0.74425095, 1.82242466, 3.98697455, 7.37979746],
                ]
            ),
            units="eV",
        )

        return bands_data

    return _generate_bands_data


@pytest.fixture
def generate_projection_data():
    """Return a `ProjectionData` node."""

    def _generate_projection_data(num_orbs=1):
        """Return a `ProjectionData` instance with some basic `orbitals` and `projections` arrays."""
        import numpy as np

        from aiida.orm import ProjectionData
        from aiida.tools.data.orbital.realhydrogen import RealhydrogenOrbital

        orbital_dict = {
            "position": [0.0, 0.0, 0.0],
            "angular_momentum": 0,
            "magnetic_number": 0,
            "radial_nodes": 0,
            "kind_name": "He",
            "spin": 0,
            "x_orientation": None,
            "z_orientation": None,
            "spin_orientation": None,
            "diffusivity": None,
        }
        # Cannot use [] * num_orbs because it is shallow copy
        orbitals = [RealhydrogenOrbital(**orbital_dict) for _ in range(num_orbs)]

        projection = np.array([[1.0, 0.5, 0.5, 0.5, 0.0], [1.0, 1.0, 1.0, 0.9, 0.0]])
        projections = [projection] * num_orbs

        projection_data = ProjectionData()
        projection_data.set_projectiondata(
            list_of_orbitals=orbitals,
            list_of_projections=projections,
            bands_check=False,
        )

        return projection_data

    return _generate_projection_data


@pytest.fixture
def generate_workchain():
    """Generate an instance of a `WorkChain`."""

    def _generate_workchain(entry_point, inputs):
        """Generate an instance of a `WorkChain` with the given entry point and inputs.

        :param entry_point: entry point name of the work chain subclass.
        :param inputs: inputs to be passed to process construction.
        :return: a `WorkChain` instance.
        """
        from aiida.engine.utils import instantiate_process
        from aiida.manage.manager import get_manager
        from aiida.plugins import WorkflowFactory

        process_class = WorkflowFactory(entry_point)
        runner = get_manager().get_runner()
        process = instantiate_process(runner, process_class, **inputs)

        return process

    return _generate_workchain


@pytest.fixture
def generate_inputs_pw(
    fixture_code, generate_structure, generate_kpoints_mesh, get_sssp_upf
):
    """Generate default inputs for a `PwCalculation."""

    def _generate_inputs_pw():
        """Generate default inputs for a `PwCalculation."""
        from aiida.orm import Dict

        from aiida_quantumespresso.utils.resources import get_default_options

        parameters = Dict(
            {
                "CONTROL": {"calculation": "scf"},
                "SYSTEM": {"ecutrho": 240.0, "ecutwfc": 30.0},
                "ELECTRONS": {
                    "electron_maxstep": 60,
                },
            }
        )
        inputs = {
            "code": fixture_code("quantumespresso.pw"),
            "structure": generate_structure(),
            "kpoints": generate_kpoints_mesh(2),
            "parameters": parameters,
            "pseudos": {"Si": get_sssp_upf("Si")},
            "metadata": {"options": get_default_options()},
        }
        return inputs

    return _generate_inputs_pw


@pytest.fixture
def generate_workchain_pw(
    generate_workchain, generate_inputs_pw, generate_calc_job_node
):
    """Generate an instance of a ``PwBaseWorkChain``."""

    def _generate_workchain_pw(
        exit_code=None, inputs=None, return_inputs=False, pw_outputs=None
    ):
        """Generate an instance of a ``PwBaseWorkChain``.

        :param exit_code: exit code for the ``PwCalculation``.
        :param inputs: inputs for the ``PwBaseWorkChain``.
        :param return_inputs: return the inputs of the ``PwBaseWorkChain``.
        :param pw_outputs: ``dict`` of outputs for the ``PwCalculation``. The keys must correspond to the link labels
            and the values to the output nodes.
        """
        from plumpy import ProcessState

        from aiida.common import LinkType
        from aiida.orm import Dict

        entry_point = "quantumespresso.pw.base"

        if inputs is None:
            pw_inputs = generate_inputs_pw()
            kpoints = pw_inputs.pop("kpoints")
            inputs = {"pw": pw_inputs, "kpoints": kpoints}

        if return_inputs:
            return inputs

        process = generate_workchain(entry_point, inputs)

        pw_node = generate_calc_job_node(inputs={"parameters": Dict()})
        process.ctx.iteration = 1
        process.ctx.children = [pw_node]

        if pw_outputs is not None:
            for link_label, output_node in pw_outputs.items():
                output_node.base.links.add_incoming(
                    pw_node, link_type=LinkType.CREATE, link_label=link_label
                )
                output_node.store()

        if exit_code is not None:
            pw_node.set_process_state(ProcessState.FINISHED)
            pw_node.set_exit_status(exit_code.status)

        return process

    return _generate_workchain_pw
