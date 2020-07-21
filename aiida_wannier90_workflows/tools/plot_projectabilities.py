#!/usr/bin/env runaiida
# -*- coding: utf-8 -*-
import argparse
from aiida import orm
from aiida_wannier90_workflows.utils.scdm import erfc_scdm, fit_scdm_mu_sigma_aiida
import matplotlib.pyplot as plt

def isNode(NodeType):
    return lambda x: x.process_label == NodeType

def findNodes(NodeType, NodeList):
    # return sorted
    nodes = list(filter(isNode(NodeType), NodeList))
    nodes.sort(key=lambda x: x.pk)
    return nodes

def findLastNode(NodeType, NodeList):
    # return last one
    return findNodes(NodeType, NodeList)[-1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script to plot the projectabilities distribution")
    parser.add_argument('pk', metavar='WORKCHAIN_PK', type=int, help="PK of Wannier90BandsWorkChain")
    args =  parser.parse_args()

    wannier90bandsworkchain = orm.load_node(args.pk)
    formula = wannier90bandsworkchain.inputs.structure.get_formula()

    if 'use_opengrid' in wannier90bandsworkchain.inputs:
        if wannier90bandsworkchain.inputs.use_opengrid:
            workchain_name = 'Wannier90OpengridWorkChain'
        else:
            workchain_name = 'Wannier90WorkChain'
    else:
        workchain_name = 'Wannier90WorkChain'
    wannier90workchain = findLastNode(workchain_name, wannier90bandsworkchain.called)

    wannier90calculation = findLastNode('Wannier90Calculation', wannier90workchain.called)
    fermi_energy = wannier90calculation.inputs.parameters['fermi_energy']

    pw2wannier90calculation = findLastNode('Pw2wannier90Calculation', wannier90workchain.called)
    sigma = pw2wannier90calculation.inputs.parameters['inputpp']['scdm_sigma']
    mu = pw2wannier90calculation.inputs.parameters['inputpp']['scdm_mu']

    projwfccalculation = findLastNode('ProjwfcCalculation', wannier90workchain.called)
    projections = projwfccalculation.outputs.projections

    print("{:6s}:".format(formula))
    print(f"        fermi_energy = {fermi_energy}, mu = {mu}, sigma = {sigma}")

    proj_bands = projwfccalculation.outputs.bands
    mu_fit, sigma_fit, data = fit_scdm_mu_sigma_aiida(proj_bands, projections, {'sigma_factor': 0}, True)
    # print(sigma, sigma_fit)
    eps = 1e-6
    assert abs(sigma - sigma_fit) < eps
    sigma_factor = wannier90workchain.inputs.scdm_thresholds.get_dict()['sigma_factor']
    assert abs(mu - (mu_fit - sigma_fit * sigma_factor)) < eps
    sorted_bands = data[0, :]
    sorted_projwfc = data[1, :]

    plt.figure()
    plt.plot(sorted_bands, sorted_projwfc, 'o')
    plt.plot(sorted_bands, erfc_scdm(sorted_bands, mu_fit, sigma_fit))
    plt.axvline([mu_fit], color='red', label=r"$\mu$")
    plt.axvline([mu_fit - sigma_factor * sigma_fit], color='orange', label=r"$\mu-"+str(sigma_factor)+r"\sigma$")
    plt.axvline([fermi_energy], color='green', label=r"$E_f$")
    plt.title(f"workchain pk {wannier90bandsworkchain.pk}, {formula}")
    plt.xlabel('Energy [eV]')
    plt.ylabel('Projectability')
    plt.legend(loc='best')
    plt.show()
    # plt.savefig('{}_proj.png'.format(formula))
