import sys
import ase


def plot_centres_xsf(structure,w90_calc):
    a = structure.get_ase()
    new_a = a.copy()
    out = w90_calc.out.output_parameters.get_dict()['wannier_functions_output']
    coords = [i['coordinates'] for i in out]
    for c in coords:
        new_a.append(ase.Atom('X',c))
    new_a.write('./wannier.xsf')

if __name__=='__main__':
    s_pk = int(sys.argv[1])
    w90_pk = int(sys.argv[2])
    s = load_node(s_pk)
    w90 = load_node(w90_pk)
    plot_centres_xsf(structure=s,w90_calc = w90)


