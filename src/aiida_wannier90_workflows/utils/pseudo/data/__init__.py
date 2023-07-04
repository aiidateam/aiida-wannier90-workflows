"""Utility functions for pseudo potential metadata."""
import json
import os
import typing as ty

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


def get_metadata(filename):
    """Return metadata."""
    result = {"filename": filename, "md5": md5(filename), "pseudopotential": "100PAW"}
    with open(filename, encoding="utf-8") as handle:
        for line in handle:
            if "Suggested minimum cutoff for wavefunctions" in line:
                wave = float(line.strip().split()[-2])
            if "Suggested minimum cutoff for charge density" in line:
                charge = float(line.strip().split()[-2])
        result["cutoff"] = wave
        result["dual"] = charge / wave
    return result


def generate_pslibrary_metadata(dirname=None):
    """Scan the folder and generate a json file containing metainfo of pseudos of pslibrary.

    :param dirname: folder to be scanned, if None download from QE website
    :type dirname: str
    """
    import urllib.request

    output_filename = "pslibrary_paw_relpbe_1.0.0.json"
    qe_site = "https://www.quantum-espresso.org/upf_files/"

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
        print(filename)
        if dirname is not None:
            filename = os.path.join(dirname, filename)
        else:
            if not os.path.exists(filename):
                # download from QE website
                url = qe_site + filename
                urllib.request.urlretrieve(url, filename)
        result[element] = get_metadata(filename)

    with open(output_filename, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)


def generate_dojo_metadata():
    """Generate metadata for pseduo-dojo SOC pseudos.

    from http://www.pseudo-dojo.org/nc-fr-04_pbe_stringent.json
    """
    dojo_json = "nc-fr-04_pbe_standard.json"
    with open(dojo_json, encoding="utf-8") as handle:
        dojo = json.load(handle)

    result = {}
    for element in dojo:
        # in pseudo-dojo standard accurary, there is no UPF endswith '_r',
        # in stringent accruray, there are UPF endswith '_r'.
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


# if __name__ == '__main__':
#     # generate_pslibrary_metadata()
#     generate_dojo_metadata()
