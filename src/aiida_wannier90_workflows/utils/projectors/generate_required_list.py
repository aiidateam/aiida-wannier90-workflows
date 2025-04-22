import json

def main():
    required_orbitals = {}
    for element in "H He".split(" "):
        required_orbitals[element] = ["1s"]

    for element in "Li Be B C N O F Ne".split(" "):
        required_orbitals[element] = ["2s", "2p"]

    for element in "Na Mg Al Si P S Cl Ar".split(" "):
        required_orbitals[element] = ["3s", "3p"]

    for element in "K Ca".split(" "):
        required_orbitals[element] = ["4s", "3d"]
    for element in "Sc Ti V Cr Mn Fe Co Ni Cu Zn".split(" "):
        required_orbitals[element] = ["4s", "4p", "3d"]
    for element in "Ga Ge As Se Br Kr".split(" "):
        required_orbitals[element] = ["4s", "4p"]

    for element in "Rb Sr".split(" "):
        required_orbitals[element] = ["5s", "4d"]
    for element in "Y Zr Nb Mo Tc Ru Rh Pd Ag Cd".split(" "):
        required_orbitals[element] = ["5s", "5p", "4d"]
    for element in "In Sn Sb Te I Xe".split(" "):
        required_orbitals[element] = ["5s", "5p"]

    for element in "Cs Ba".split(" "):
        required_orbitals[element] = ["6s", "5d"]
    for element in "Hf Ta W Re Os Ir Pt Au Hg".split(" "):
        required_orbitals[element] = ["6s", "6p", "5d"]
    for element in "Tl Pb Bi Po At Rn".split(" "):
        required_orbitals[element] = ["6s", "6p"]

    with open("./required_orbitals.json", "w") as fout:
        json.dump(required_orbitals, fout, indent=2)


if __name__ == "__main__":
    main()
