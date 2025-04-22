"""Self modified MagnetStructureData including magmom."""

from copy import deepcopy

import numpy as np

from aiida.orm.nodes.data.structure import Kind as oldKind
from aiida.orm.nodes.data.structure import Site, StructureData

__all__ = ("MagneticStructureData", "Kind", "Site")

_MASS_THRESHOLD = 1.0e-3
# Threshold to check if the magmom(or diff of magmom) is zero or not
_MAGMOM_THRESHOLD = 1.0e-4


def get_unique_magmoms(sites: list) -> dict:
    """Loop over all sites in pymatgen.core.Structure to find all unique magmoms."""
    from pymatgen.electronic_structure.core import Magmom

    unique_magmoms = {}
    for site in sites:
        symbol = site.specie.symbol
        if symbol not in unique_magmoms:
            unique_magmoms[symbol] = []
        try:
            magmom = site.properties["magmom"].moment
        except KeyError:  # magmom not in site.properties
            magmom = np.array([0.0, 0.0, 0.0])
            site.properties["magmom"] = Magmom(magmom)
        if not any(
            np.linalg.norm(magmom - m) < _MAGMOM_THRESHOLD
            for m in unique_magmoms[symbol]
        ):
            unique_magmoms[symbol].append(magmom)

    return unique_magmoms


class MagneticStructureData(StructureData):
    """Override aiida.orm.StructureData.

    Enable to generate StructureData that give atoms with different magnetic moment
    different Kinds.name and Sites.kind_name.
    Only support set_pymatgen now.
    """

    # Override
    def set_pymatgen_structure(self, struct):
        """Load the structure from a pymatgen Structure object.

        .. note:: periodic boundary conditions are set to True in all
            three directions.
        .. note:: Requires the pymatgen module (version >= 3.3.5, usage
            of earlier versions may cause errors).

        :raise ValueError: if there are partial occupancies together with spins.
        """

        def build_kind_name(species_and_occu):
            """Build a kind name from a pymatgen Composition.

            Including an additional ordinal if spin is included,
            e.g. it returns '<specie>1' for an atom with spin < 0 and '<specie>2' for an atom with spin > 0,
            otherwise (no spin) it returns None

            :param species_and_occu: a pymatgen species and occupations dictionary
            :return: a string representing the kind name or None
            """
            from aiida.orm.nodes.data.structure import create_automatic_kind_name

            species = list(species_and_occu.keys())
            occupations = list(species_and_occu.values())

            # As of v2023.9.2, the ``properties`` argument is removed and the ``spin`` argument should be used.
            # See: https://github.com/materialsproject/pymatgen/commit/118c245d6082fe0b13e19d348fc1db9c0d512019
            # The ``spin`` argument was introduced in v2023.6.28.
            # See: https://github.com/materialsproject/pymatgen/commit/9f2b3939af45d5129e0778d371d814811924aeb6
            has_spin_attribute = hasattr(species[0], "_spin")

            if has_spin_attribute:
                has_spin = any(specie.spin != 0 for specie in species)
            else:
                has_spin = any(
                    specie.as_dict().get("properties", {}).get("spin", 0) != 0
                    for specie in species
                )

            has_partial_occupancies = len(occupations) != 1 or occupations[0] != 1.0

            if has_partial_occupancies and has_spin:
                raise ValueError(
                    "Cannot set partial occupancies and spins at the same time"
                )

            if has_spin:
                symbols = [specie.symbol for specie in species]
                kind_name = create_automatic_kind_name(symbols, occupations)

                # If there is spin, we can only have a single specie, otherwise we would have raised above
                specie = species[0]
                if has_spin_attribute:
                    spin = specie.spin
                else:
                    spin = specie.as_dict().get("properties", {}).get("spin", 0)

                if spin < 0:
                    kind_name += "1"
                else:
                    kind_name += "2"

                return kind_name

            return None

        self.cell = struct.lattice.matrix.tolist()
        self.pbc = [True, True, True]
        self.clear_kinds()
        struct = deepcopy(struct)

        hasmag = "magmom" in struct.site_properties
        if hasmag:
            unique_magmoms = get_unique_magmoms(struct.sites)
        for site in struct.sites:
            species_and_occu = site.species

            if "kind_name" in site.properties:
                kind_name = site.properties["kind_name"]
            else:
                kind_name = build_kind_name(species_and_occu)

            inputs = {
                "symbols": [x.symbol for x in species_and_occu.keys()],
                "weights": list(species_and_occu.values()),
                "position": site.coords.tolist(),
            }
            if hasmag:
                magmom_site = site.properties["magmom"].moment
                # Magnetic system, the kind_name should override the spin
                if len(unique_magmoms[site.specie.symbol]) == 1:
                    kind_name = site.specie.symbol
                else:
                    kind_name = site.specie.symbol + str(
                        [
                            np.linalg.norm(magmom_site - m) < _MAGMOM_THRESHOLD
                            for m in unique_magmoms[site.specie.symbol]
                        ].index(True)
                    )
                inputs["magmom"] = magmom_site
            if kind_name is not None:
                inputs["name"] = kind_name
            self.append_atom(**inputs)

    def append_kind(self, kind):
        """Override: Append a kind to the `StructureData <aiida.orm.nodes.data.structure.StructureData>`.

        It makes a copy of the kind.
        :param kind: the site to append, must be a Kind object.
        """
        from aiida.common.exceptions import ModificationNotAllowed

        if self.is_stored:
            raise ModificationNotAllowed(
                "The StructureData object cannot be modified, it has already been stored"
            )

        new_kind = Kind(kind=kind)  # So we make a copy

        if kind.name in [k.name for k in self.kinds]:
            raise ValueError(f"A kind with the same name ({kind.name}) already exists.")

        # If here, no exceptions have been raised, so I add the site.
        self.base.attributes.all.setdefault("kinds", []).append(new_kind.get_raw())
        # Note, this is a dict (with integer keys) so it allows for empty spots!
        if self._internal_kind_tags is None:
            self._internal_kind_tags = {}

        self._internal_kind_tags[len(self.base.attributes.get("kinds")) - 1] = (
            kind._internal_tag
        )

    def append_atom(self, **kwargs):
        """Append an atom to the Structure, taking care of creating the corresponding kind.

        :param ase: the ase Atom object from which we want to create a new atom
                (if present, this must be the only parameter)
        :param position: the position of the atom (three numbers in angstrom)
        :param symbols: passed to the constructor of the Kind object.
        :param weights: passed to the constructor of the Kind object.
        :param name: passed to the constructor of the Kind object. See also the note below.
        :param magmom: passed to the constuctor of the Kind object.

        .. note :: Note on the 'name' parameter (that is, the name of the kind):

            * if specified, no checks are done on existing species. Simply,
                a new kind with that name is created. If there is a name
                clash, a check is done: if the kinds are identical, no error
                is issued; otherwise, an error is issued because you are trying
                to store two different kinds with the same name.

            * if not specified, the name is automatically generated. Before
                adding the kind, a check is done. If other species with the
                same properties already exist, no new kinds are created, but
                the site is added to the existing (identical) kind.
                (Actually, the first kind that is encountered).
                Otherwise, the name is made unique first, by adding to the string
                containing the list of chemical symbols a number starting from 1,
                until an unique name is found

        .. note :: checks of equality of species are done using
            the :py:meth:`~aiida.orm.nodes.data.structure.Kind.compare_with` method.
        """
        aseatom = kwargs.pop("ase", None)
        if aseatom is not None:
            if kwargs:
                raise ValueError(
                    "If you pass 'ase' as a parameter to "
                    "append_atom, you cannot pass any further"
                    "parameter"
                )
            position = aseatom.position
            kind = Kind(ase=aseatom)
        else:
            position = kwargs.pop("position", None)
            if position is None:
                raise ValueError("You have to specify the position of the new atom")
            # all remaining parameters
            kind = Kind(**kwargs)

        # I look for identical species only if the name is not specified
        _kinds = self.kinds

        if "name" not in kwargs:
            # If the kind is identical to an existing one, I use the existing
            # one, otherwise I replace it
            exists_already = False
            for idx, existing_kind in enumerate(_kinds):
                try:
                    existing_kind._internal_tag = self._internal_kind_tags[idx]
                except KeyError:
                    # self._internal_kind_tags does not contain any info for
                    # the kind in position idx: I don't have to add anything
                    # then, and I continue
                    pass
                if kind.compare_with(existing_kind)[0]:
                    kind = existing_kind
                    exists_already = True
                    break
            if not exists_already:
                # There is not an identical kind.
                # By default, the name of 'kind' just contains the elements.
                # I then check that the name of 'kind' does not already exist,
                # and if it exists I add a number (starting from 1) until I
                # find a non-used name.
                existing_names = [k.name for k in _kinds]
                simplename = kind.name
                counter = 1
                while kind.name in existing_names:
                    kind.name = f"{simplename}{counter}"
                    counter += 1
                self.append_kind(kind)
        else:  # 'name' was specified
            old_kind = None
            for existing_kind in _kinds:
                if existing_kind.name == kwargs["name"]:
                    old_kind = existing_kind
                    break
            if old_kind is None:
                self.append_kind(kind)
            else:
                is_the_same, firstdiff = kind.compare_with(old_kind)
                if is_the_same:
                    kind = old_kind
                else:
                    raise ValueError(
                        "You are explicitly setting the name "
                        f"of the kind to '{kind.name}', that already "
                        "exists, but the two kinds are different!"
                        f" (first difference: {firstdiff})"
                    )

        site = Site(kind_name=kind.name, position=position)
        self.append_site(site)

    @property
    def kinds(self):
        """Returns a list of kinds."""
        try:
            raw_kinds = deepcopy(self.base.attributes.get("kinds"))
        except AttributeError:
            raw_kinds = []

        return [Kind(raw=i) for i in raw_kinds]

    def to_aiida_structure(self):
        """Convert the MagnetStructureData to core.StructureData."""
        structure = StructureData(cell=self.cell, pbc=self.pbc)
        structure.clear_kinds()
        for kind in self.kinds:
            structure.append_kind(kind=kind)
        for site in self.sites:
            structure.append_site(site=site)
        return structure

    def has_magmom(self):
        """Return True if all kinds has magmom property."""
        return all(k._magmom for k in self.kinds)

    def is_collin_mag(self):
        """Check if all magentic moments are in fact collinear.

        Num of kinds would not be too large, so I may check twice
        M_A // M_B and M_B // M_A
        """
        for kind in self.kinds:
            if kind.get_magmom_coord()[0] < _MAGMOM_THRESHOLD:
                continue
            magmom = kind.get_magmom_coord(coord="cartesian")
            for kind_compare in self.kinds:
                magmom_compare = kind_compare.get_magmom_coord(coord="cartesian")
                if (
                    not (
                        np.linalg.norm(magmom_compare)
                        - np.dot(magmom, magmom_compare) / np.linalg.norm(magmom)
                    )
                    < _MAGMOM_THRESHOLD
                ):
                    return False
        return True  # if never return False, return True


class Kind(oldKind):
    """Override and add magmom."""

    def __init__(self, **kwargs):
        """Create a site.

        One can either pass:

        :param raw: the raw python dictionary that will be converted to a
               Kind object.
        :param ase: an ase Atom object
        :param kind: a Kind object (to get a copy)

        Or alternatively the following parameters:

        :param symbols: a single string for the symbol of this site, or a list
                   of symbol strings
        :param weights: (optional) the weights for each atomic species of
                   this site.
                   If only a single symbol is provided, then this value is
                   optional and the weight is set to 1.
        :param mass: (optional) the mass for this site in atomic mass units.
                   If not provided, the mass is set by the
                   self.reset_mass() function.
        :param name: a string that uniquely identifies the kind, and that
                   is used to identify the sites.
        """
        # Internal variables
        self._magmom = None
        if "raw" in kwargs:
            magmom_raw = kwargs["raw"].pop("magmom", None)
        elif "kind" in kwargs:
            pass
        elif "ase" in kwargs:
            pass
        else:
            magmom = kwargs.pop("magmom", None)
        super().__init__(**kwargs)

        # It will be remain to None in general; it is used to further
        # identify this species. At the moment, it is used only when importing
        # from ASE, if the species had a tag (different from zero).
        ## NOTE! This is not persisted on DB but only used while the class
        # is loaded in memory (i.e., it is not output with the get_raw() method)
        self._internal_tag = None

        # Logic to create the site from the raw format
        if "raw" in kwargs:
            if len(kwargs) != 1:
                raise ValueError(
                    "If you pass 'raw', then you cannot pass any other parameter."
                )
            if magmom_raw is not None:
                self.magmom = magmom_raw
            else:
                self.magmom = None  # no magnetic system

        elif "kind" in kwargs:
            if len(kwargs) != 1:
                raise ValueError(
                    "If you pass 'kind', then you cannot pass any other parameter."
                )
            oldkind = kwargs["kind"]

            try:
                self.magmom = oldkind.magmom
            except AttributeError:
                self.magmom = None

        elif "ase" in kwargs:
            # do not know how to get magmom from ase
            pass
        else:
            if "symbols" not in kwargs:
                raise ValueError(
                    "'symbols' need to be "
                    "specified (at least) to create a Site object. Otherwise, "
                    "pass a raw site using the 'raw' parameter."
                )
            weights = kwargs.pop("weights", None)
            self.set_symbols_and_weights(kwargs.pop("symbols"), weights)
            try:
                self.mass = kwargs.pop("mass")
            except KeyError:
                self.reset_mass()
            try:
                self.name = kwargs.pop("name")
            except KeyError:
                self.set_automatic_kind_name()
            if kwargs:
                raise ValueError(
                    f"Unrecognized parameters passed to Kind constructor: {kwargs.keys()}"
                )
            if magmom is not None:
                self.magmom = magmom
            else:
                self.magmom = None

    def get_raw(self):
        """Return the raw version of the site, mapped to a suitable dictionary.

        This is the format that is actually used to store each kind of the
        structure in the DB.

        :return: a python dictionary with the kind.
        """
        if self._magmom is None:
            raw_out = {
                "symbols": self.symbols,
                "weights": self.weights,
                "mass": self.mass,
                "name": self.name,
            }
        else:
            raw_out = {
                "symbols": self.symbols,
                "weights": self.weights,
                "mass": self.mass,
                "name": self.name,
                "magmom": self.magmom,
            }
        return raw_out

    def compare_with(self, other_kind):
        """Compare with another Kind object to check if they are different.

        .. note:: This does NOT check the 'type' attribute. Instead, it compares
            (with reasonable thresholds, where applicable): the mass, and the list
            of symbols and of weights. Moreover, it compares the
            ``_internal_tag``, if defined (at the moment, defined automatically
            only when importing the Kind from ASE, if the atom has a non-zero tag).
            Note that the _internal_tag is only used while the class is loaded,
            but is not persisted on the database.

        :return: A tuple with two elements.
        The first one is True if the two sites are 'equivalent' (same mass, symbols and weights), False otherwise.
        The second element of the tuple is a string, which is either None (if the first element was True), or contains
        a 'human-readable' description of the first difference encountered between the two sites.
        """  # pylint: disable=too-many-return-statements
        # Check length of symbols
        if len(self.symbols) != len(other_kind.symbols):
            return (False, "Different length of symbols list")

        # Check list of symbols
        for i, symbol in enumerate(self.symbols):
            if symbol != other_kind.symbols[i]:
                return (
                    False,
                    f"Symbol at position {i + 1:d} are different ({symbol} vs. {other_kind.symbols[i]})",
                )
        # Check weights (assuming length of weights and of symbols have same
        # length, which should be always true
        for i, weight in enumerate(self.weights):
            if weight != other_kind.weights[i]:
                return (
                    False,
                    f"Weight at position {i + 1:d} are different ({weight} vs. {other_kind.weights[i]})",
                )
        # Check masses
        if abs(self.mass - other_kind.mass) > _MASS_THRESHOLD:
            return (False, f"Masses are different ({self.mass} vs. {other_kind.mass})")

        # Check magmom
        try:
            diff = (
                np.linalg.norm(np.array(self.magmom) - np.array(other_kind.magmom))
                > _MAGMOM_THRESHOLD
            )
        except AttributeError:
            pass
        else:
            if diff:
                return (
                    False,
                    f"Magmoms are different ({self.magmom} vs. {other_kind.magmom})",
                )

        if self._internal_tag != other_kind._internal_tag:
            return (
                False,
                f"Internal tags are different ({self._internal_tag} vs. {other_kind._internal_tag})",
            )

        # If we got here, the two Site objects are similar enough
        # to be considered of the same kind
        return (True, "")

    def get_magmom_coord(self, coord="spherical"):
        """Get magnetic moment in given coordinate.

        :return: spherical theta and phi in unit rad
                 cartesian x y and z in unit ang
        """
        try:
            self.magmom
        except AttributeError as exc:
            raise AttributeError(
                f"Kind {self.name} do not have attribute `magmom`."
            ) from exc

        if coord not in ["spherical", "cartesian"]:
            raise ValueError("`coord` can only be `cartesian` or `spherical`")
        if coord == "cartesian":
            magmom_coord = self.magmom
        else:
            r = np.linalg.norm(self.magmom)
            if r < _MAGMOM_THRESHOLD:
                magmom_coord = (0.0, 0.0, 0.0)
            else:
                theta = np.arccos(self.magmom[2] / r)  # arccos(z/r)
                theta = theta / np.pi * 180
                phi = np.arctan2(self.magmom[1], self.magmom[0])  # atan2(y, x)
                phi = phi / np.pi * 180
                magmom_coord = (r, theta, phi)
                # unit always in degree to fit qe inputs.
        return magmom_coord

    @property
    def magmom(self):
        """Return the magnetic moment of this kind.

        :return: Optional[tuple(float)]
        """
        if self._magmom is None:
            raise AttributeError(f"Kind {self.name} do not have attribute `magmom`.")

        return deepcopy(self._magmom)

    @magmom.setter
    def magmom(self, value):
        """Set the magmom of this kind."""
        if value is None:
            self._magmom = None
            return

        try:
            internal_mag = tuple(float(i) for i in value)
            if len(internal_mag) != 3:
                raise ValueError
        except (ValueError, TypeError) as exc:
            raise ValueError(
                "Wrong format for magmom, must be a list of 3 float numbers."
            ) from exc
        self._magmom = internal_mag
