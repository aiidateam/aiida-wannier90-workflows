"""Utility functions for pseudo potential metadata."""

import json
import os
import typing as ty
import xml.sax

__all__ = ("load_pseudo_metadata",)


def load_pseudo_metadata(filename):
    """Load from the current folder a json file containing metadata for a library of pseudopotentials.

    incl. suggested cutoffs.
    """
    with open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), filename),
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def md5(filename):
    """Get md5 of a file."""
    import hashlib

    hash_md5 = hashlib.md5()
    with open(filename, "rb") as handle:
        for chunk in iter(lambda: handle.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


class PSHandler(xml.sax.ContentHandler):
    """Read xml files in pslibrary using xml.sax.

    This script can generate cutoff energy/rho, pswfcs and semicores for
    FULLY RELATIVISTIC pseudopotentials (rel-*).
    Use SSSP or Pesudo-dojo for scalar/non relativistic pseudos.
    """

    def __init__(self) -> None:
        super().__init__()
        # define the p-block
        self.pblock = list(range(31, 37))
        self.pblock.extend(list(range(49, 55)))
        self.pblock.extend(list(range(81, 87)))
        self.pblock.extend(list(range(113, 119)))
        # in p-block ns, np are planewaves and n-1d are semicores
        # otherwise ns, np and n-1d are pws, n-1s n-1p are semicores
        self.readWFC = False
        self.pswfcs = []
        self.pswfcs_shell = []
        self.semicores = []
        self.znum = 0

    def startElement(self, name, attrs):
        """Instructions for the beginning of different sections.

        If in PP_MESH section, read element's index in periodic table;
        If in PP_SPIN_ORB, set read wavefunctions flag to True;
        If in PP_RELWFC, read the orbital labels (5S, 5D, etc.)
        and calculate the "weight" by taking the principal quantum number
        and adding 1 for D orbitals, S and P orbitals for the elements in P block,
        and 2 for F orbitals. If not in P block, S and P weight is 0.
        This will later be used to identify semicores.
        """

        # dojo UPF does not have zmesh, use PP_HEADER to be safe
        # if name == "PP_MESH":
        #     try:
        #         self.znum = int(float(attrs["zmesh"]))
        #     except ValueError:
        #         print(f"z = {attrs['zmesh']} is not acceptable")
        if name == "PP_HEADER":
            from ase.data import atomic_numbers

            self.znum = atomic_numbers[attrs["element"].strip()]

        # Instead of PP_RELWFC, PSWFC/PP_CHI is what is used as projection functions
        # if name == "PP_SPIN_ORB":
        #     # <PP_SPIN_ORB>
        #     #   <PP_RELWFC.1 index="1" els="5S" nn="1" lchi="0" jchi="0.500000000000000" oc="2.00000000000000"/>
        #     #   <PP_RELWFC.2 index="2" els="6S" nn="2" lchi="0" jchi="0.500000000000000" oc="1.00000000000000"/>
        #     #   ...
        #     #   <PP_RELBETA.1 index="1" lll="0" jjj="0.500000000000000"/>
        #     #   <PP_RELBETA.2 index="2" lll="0" jjj="0.500000000000000"/>
        #     #   <PP_RELBETA.3 index="3" lll="0" jjj="0.500000000000000"/>
        #     #   ...
        #     # </PP_SPIN_ORB>
        if name == "PP_PSWFC":
            self.readWFC = True
        # if "PP_RELWFC" in name and self.readWFC:
        #     orb = attrs["els"]
        if "PP_CHI" in name and self.readWFC:
            orb = attrs["label"]
            if not orb in self.pswfcs:
                self.pswfcs.append(orb)
                nn = int(orb[0])
                orbtype = orb[-1]
                if orbtype in ("S", "P"):
                    ll = 1 if self.znum in self.pblock else 0
                if orbtype == "D":
                    ll = 1
                if orbtype == "F":
                    ll = 2
                self.pswfcs_shell.append(nn + ll)

    def endElement(self, name):
        """Select semicores.

        When the PP_SPIN_ORB ends, all pswfcs are recorded in list,
        then we can determine semicores using the rules introduced in init
        """

        # if name == "PP_SPIN_ORB":
        if name == "PP_PSWFC":
            self.readWFC = False
            self.znum = 0
            maxshell = max(self.pswfcs_shell)
            for iorb in enumerate(self.pswfcs_shell):
                if iorb[1] < maxshell:
                    self.semicores.append(self.pswfcs[iorb[0]])


def get_metadata(filename, cutoff: bool = True):
    """Return metadata."""

    # this part reads the upf file twice, but it do not take too much time
    result = {"filename": filename, "md5": md5(filename)}
    if cutoff:
        result["pseudopotential"] = "100PAW"
    if cutoff:
        with open(filename, encoding="utf-8") as handle:
            for line in handle:
                if "Suggested minimum cutoff for wavefunctions" in line:
                    wave = float(line.strip().split()[-2])
                if "Suggested minimum cutoff for charge density" in line:
                    charge = float(line.strip().split()[-2])
    # use xml.sax to parse upf file
    parser = xml.sax.make_parser()
    Handler = PSHandler()
    parser.setContentHandler(Handler)
    parser.parse(filename)
    if cutoff:
        # cutoffs in unit: Ry
        result["cutoff_wfc"] = wave
        result["cutoff_rho"] = charge
    result["pswfcs"] = Handler.pswfcs
    result["semicores"] = Handler.semicores
    return result


def generate_pslibrary_metadata(dirname=None):
    """Scan the folder and generate a json file containing metainfo of pseudos of pslibrary.

    :param dirname: folder to be scanned, if None download from QE website
    :type dirname: str
    """
    import shutil
    import urllib.request

    output_filename = "pslibrary_paw_relpbe_1.0.0.json"
    qe_site = "https://pseudopotentials.quantum-espresso.org/upf_files/"

    # these are the suggested PP from https://dalcorso.github.io/pslibrary/PP_list.html (2020.07.21)
    suggested = r"""H:  H.$fct-*_psl.1.0.0
He: He.$fct-*_psl.1.0.0
Li: Li.$fct-sl-*_psl.1.0.0
Be: Be.$fct-sl-*_psl.1.0.0
B:  B.$fct-n-*_psl.1.0.0
C:  C.$fct-n-*_psl.1.0.0
N:  N.$fct-n-*_psl.1.0.0
O:  O.$fct-n-*_psl.1.0.0
F:  F.$fct-n-*_psl.1.0.0
Ne: Ne.$fct-n-*_psl.1.0.0
Na: Na.$fct-spnl-*_psl.1.0.0
Mg: Mg.$fct-spnl-*_psl.1.0.0
Al: Al.$fct-nl-*_psl.1.0.0
Si: Si.$fct-nl-*_psl.1.0.0
P:  P.$fct-nl-*_psl.1.0.0
S:  S.$fct-nl-*_psl.1.0.0
Cl: Cl.$fct-nl-*_psl.1.0.0
Ar: Ar.$fct-nl-*_psl.1.0.0
K:  K.$fct-spn-*_psl.1.0.0
Ca: Ca.$fct-spn-*_psl.1.0.0
Sc: Sc.$fct-spn-*_psl.1.0.0
Ti: Ti.$fct-spn-*_psl.1.0.0
V:  V.$fct-spnl-*_psl.1.0.0
Cr: Cr.$fct-spn-*_psl.1.0.0
Mn: Mn.$fct-spn-*_psl.0.3.1
Fe: Fe.$fct-n-*_psl.1.0.0
Co: Co.$fct-n-*_psl.0.3.1
Ni: Ni.$fct-n-*_psl.1.0.0
Cu: Cu.$fct-dn-*_psl.1.0.0
Zn: Zn.$fct-dn-*_psl.1.0.0
Ga: Ga.$fct-dnl-*_psl.1.0.0
Ge: Ge.$fct-n-*_psl.1.0.0
As: As.$fct-n-*_psl.1.0.0
Se: Se.$fct-n-*_psl.1.0.0
Br: Br.$fct-n-*_psl.1.0.0
Kr: Kr.$fct-dn-*_psl.1.0.0
Rb: Rb.$fct-spn-*_psl.1.0.0
Sr: Sr.$fct-spn-*_psl.1.0.0
Y:  Y.$fct-spn-*_psl.1.0.0
Zr: Zr.$fct-spn-*_psl.1.0.0
Nb: Nb.$fct-spn-*_psl.1.0.0
Mo: Mo.$fct-spn-*_psl.1.0.0
Tc: Tc.$fct-spn-*_psl.0.3.0
Ru: Ru.$fct-spn-*_psl.1.0.0
Rh: Rh.$fct-spn-*_psl.1.0.0
Pd: Pd.$fct-n-*_psl.1.0.0
Ag: Ag.$fct-n-*_psl.1.0.0
Cd: Cd.$fct-n-*_psl.1.0.0
In: In.$fct-dn-*_psl.1.0.0
Sn: Sn.$fct-dn-*_psl.1.0.0
Sb: Sb.$fct-n-*_psl.1.0.0
Te: Te.$fct-n-*_psl.1.0.0
I:  I.$fct-n-*_psl.1.0.0
Xe: Xe.$fct-dn-*_psl.1.0.0
Cs: Cs.$fct-spnl-*_psl.1.0.0
Ba: Ba.$fct-spn-*_psl.1.0.0
La: La.$fct-spfn-*_psl.1.0.0
Ce: Ce.$fct-spdn-*_psl.1.0.0
Pr: Pr.$fct-spdn-*_psl.1.0.0
Nd: Nd.$fct-spdn-*_psl.1.0.0
Pm: Pm.$fct-spdn-*_psl.1.0.0
Sm: Sm.$fct-spdn-*_psl.1.0.0
Eu: Eu.$fct-spn-*_psl.1.0.0
Gd: Gd.$fct-spdn-*_psl.1.0.0
Tb: Tb.$fct-spdn-*_psl.1.0.0
Dy: Dy.$fct-spdn-*_psl.1.0.0
Ho: Ho.$fct-spdn-*_psl.1.0.0
Er: Er.$fct-spdn-*_psl.1.0.0
Tm: Tm.$fct-spdn-*_psl.1.0.0
Yb: Yb.$fct-spn-*_psl.1.0.0
Lu: Lu.$fct-spdn-*_psl.1.0.0
Hf: Hf.$fct-spn-*_psl.1.0.0
Ta: Ta.$fct-spn-*_psl.1.0.0
W:  W.$fct-spn-*_psl.1.0.0
Re: Re.$fct-spn-*_psl.1.0.0
Os: Os.$fct-spn-*_psl.1.0.0
Ir: Ir.$fct-n-*_psl.0.2.3
Pt: Pt.$fct-n-*_psl.1.0.0
Au: Au.$fct-n-*_psl.1.0.1
Hg: Hg.$fct-n-*_psl.1.0.0
Tl: Tl.$fct-dn-*_psl.1.0.0
Pb: Pb.$fct-dn-*_psl.1.0.0
Bi: Bi.$fct-dn-*_psl.1.0.0
Po: Po.$fct-dn-*_psl.1.0.0
At: At.$fct-dn-*_psl.1.0.0
Rn: Rn.$fct-dn-*_psl.1.0.0
Fr: Fr.$fct-spdn-*_psl.1.0.0
Ra: Ra.$fct-spdn-*_psl.1.0.0
Ac: Ac.$fct-spfn-*_psl.1.0.0
Th: Th.$fct-spfn-*_psl.1.0.0
Pa: Pa.$fct-spfn-*_psl.1.0.0
U:  U.$fct-spfn-*_psl.1.0.0
Np: Np.$fct-spfn-*_psl.1.0.0
Pu: Pu.$fct-spfn-*_psl.1.0.0"""
    # use PAW, PBE, SOC
    star = "kjpaw"
    fct = "rel-pbe"
    # For conventionally installed pslibrary, the dirname folder will contain
    # more pseudos than what we expect. It is essential to create a sub-folder
    # to store the pseudos we need if we want to install the pseudos to aiida-pseudo.
    # If dirname == None, the pslibrary_install folder will be placed in workdir.
    if dirname is not None:
        dir_pseudos_install = os.path.join(dirname, "pslibrary_install")
    else:
        dir_pseudos_install = "pslibrary_install"
    os.makedirs(dir_pseudos_install, exist_ok=True)
    result = {}
    suggested = suggested.replace("*", star).replace("$fct", fct)
    suggested = suggested.split("\n")
    for line in suggested:
        element, filename = line.strip().split(":")
        element = element.strip()
        filename = filename.strip() + ".UPF"
        # I cannot find these UPF, temporarily replace it
        if filename == "Co.rel-pbe-n-kjpaw_psl.0.3.1.UPF":
            filename = "Co.rel-pbe-spn-kjpaw_psl.0.3.1.UPF"
        if filename == "Au.rel-pbe-n-kjpaw_psl.1.0.1.UPF":
            filename = "Au.rel-pbe-n-kjpaw_psl.1.0.0.UPF"
        # Cs.rel-pbe-spnl-kjpaw_psl.1.0.0.UPF gives 'z_valence=-5.00000000000000'
        # it is not a legal value for aiida-pseudo install
        if filename == "Cs.rel-pbe-spnl-kjpaw_psl.1.0.0.UPF":
            filename = "Cs.rel-pbe-spn-kjpaw_psl.1.0.0.UPF"
        # these file do not exist in qe_site
        if filename == "Ar.rel-pbe-nl-kjpaw_psl.1.0.0.UPF":
            filename = "Ar.rel-pbe-n-kjpaw_psl.1.0.0.UPF"
        if filename == "Fe.rel-pbe-n-kjpaw_psl.1.0.0.UPF":
            filename = "Fe.rel-pbe-spn-kjpaw_psl.1.0.0.UPF"
        if filename == "Ni.rel-pbe-n-kjpaw_psl.1.0.0.UPF":
            filename = "Ni.rel-pbe-spn-kjpaw_psl.1.0.0.UPF"
        if filename == "Zn.rel-pbe-dn-kjpaw_psl.1.0.0.UPF":
            filename = "Zn.rel-pbe-dnl-kjpaw_psl.1.0.0.UPF"
        print(filename)
        # copy the pseudopotentials to another folder, which only contain the pp we need
        dst_filename = os.path.join(dir_pseudos_install, filename)
        if dirname is not None:
            shutil.copy2(os.path.join(dirname, filename), dst_filename)
        else:
            if not os.path.exists(filename):
                # download from QE website
                url = qe_site + filename
                urllib.request.urlretrieve(url, dst_filename)
        result[element] = get_metadata(dst_filename)
        result[element]["filename"] = filename
    # the output file (as well as the pseudo_install if dirname==None) will be placed in the workdir
    with open(output_filename, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(
        "Use following commands to install pslibrary as a cutoffs family in aiida-pseudo"
    )
    print(
        f"\taiida-pseudo install family '{dir_pseudos_install}' <LABEL> -F pseudo.family.cutoffs -P pseudo.upf"
    )
    print(
        f"\taiida-pseudo family cutoffs set -s <STRINGENCY(standard)> -u Ry <FAMILY> '{output_filename}'"
    )
    print(
        "Please move the json file into "
        + "'.../aiida-wannier90-workflows/src/aiida_wannier90_workflows/utils/pseudo/data/semicore/'"
    )


def generate_dojo_metadata():
    """Generate metadata for pseduo-dojo SOC pseudos.

    from http://www.pseudo-dojo.org/nc-fr-04_pbe_stringent.json
    """
    dojo_json = "nc-fr-04_pbe_standard.json"
    with open(dojo_json, encoding="utf-8") as handle:
        dojo = json.load(handle)

    result = {}
    for element in dojo:
        # in pseudo-dojo standard accuracy, there is no UPF endswith '_r',
        # in stringent accuracy, there are UPF endswith '_r'.
        # Not sure what '_r' means, but if a element endswith '_r', then
        # its cutoff is not shown in the HTML page.
        if element.endswith("_r"):
            continue
        filename = element + ".upf"
        result[element] = {
            "filename": filename,
            "md5": md5(filename),
            "pseudopotential": "Dojo",
            # use normal accurary, and the original unit is Hartree - convert to Rydberg
            "cutoff": 2.0 * dojo[element]["hn"],
            "dual": 4.0,
        }

    with open("dojo_nc_fr.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)


def _print_exclude_semicore():
    """Print semicore."""
    periodic_table = "H He Li Be B C N O F Ne "
    periodic_table += "Na Mg Al Si P S Cl Ar "
    periodic_table += "K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
    periodic_table += "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe "
    periodic_table += "Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn "
    periodic_table += "Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
    periodic_table = periodic_table.split()

    with open("semicore_sssp_efficiency_1.1.json", encoding="utf-8") as handle:
        data = json.load(handle)

    for kind in periodic_table:
        if not kind in data:
            continue
        remaining = set(data[kind]["pswfcs"]) - set(data[kind]["semicores"])
        print(f"{kind:2s} {' '.join(remaining)}")


def generate_dojo_semicore():
    """Generate semicore data for dojo fr pbesol standard."""
    with open("nc-fr-04_pbesol_standard.json", encoding="utf-8") as handle:
        data = json.load(handle)

    result = {}
    for kind in data:
        symbol = kind.removesuffix("_r")
        filename = f"{symbol}.upf"
        result[symbol] = {
            "filename": filename,
            "md5": md5(filename),
            **get_metadata(filename, cutoff=False),
        }
    with open(
        "PseudoDojo_0.4_PBEsol_FR_standard_upf.json", "w", encoding="utf-8"
    ) as handle:
        json.dump(result, handle, indent=2)


if __name__ == "__main__":
    # generate_pslibrary_metadata()
    # generate_dojo_metadata()
    generate_dojo_semicore()
