# aiida-wannier90-workflows
Advanced AiiDA workflows for automated Wannierisation.

|     | |
|-----|----------------------------------------------------------------------------|
|Latest release| [![PyPI version](https://badge.fury.io/py/aiida-wannier90-workflows.svg)](https://badge.fury.io/py/aiida-wannier90-workflows) [![PyPI pyversions](https://img.shields.io/pypi/pyversions/aiida-wannier90-workflows.svg)](https://pypi.python.org/pypi/aiida-wannier90-workflows/) |
|Getting help| [![Docs status](https://readthedocs.org/projects/aiida-wannier90-workflows/badge)](http://aiida-wannier90-workflows.readthedocs.io/) [![Google Group](https://img.shields.io/badge/-Google%20Group-lightgrey.svg)](https://groups.google.com/forum/#!forum/aiidausers)
|Build status| [![Build Status](https://github.com/aiidateam/aiida-wannier90-workflows/actions/workflows/ci.yml/badge.svg)](https://github.com/aiidateam/aiida-wannier90-workflows/actions) [![Coverage Status](https://codecov.io/gh/aiidateam/aiida-wannier90-workflows/branch/develop/graph/badge.svg)](https://codecov.io/gh/aiidateam/aiida-wannier90-workflows) |
|Activity| [![PyPI-downloads](https://img.shields.io/pypi/dm/aiida-wannier90-workflows.svg?style=flat)](https://pypistats.org/packages/aiida-wannier90-workflows) [![Commit Activity](https://img.shields.io/github/commit-activity/m/aiidateam/aiida-wannier90-workflows.svg)](https://github.com/aiidateam/aiida-wannier90-workflows/pulse)


The protocol for automating the construction of Wannier functions is discussed in the following article

* Valerio Vitale, Giovanni Pizzi, Antimo Marrazzo, Jonathan Yates, Nicola Marzari, Arash Mostofi,
  *Automated high-throughput wannierisation*, accepted in npj Computational Materials (2020);
  https://arxiv.org/abs/1909.00433; https://doi.org/10.24435/materialscloud:2019.0044/v2.

which leverages the SCDM method that was introduced in:

* Anil Damle, Lin Lin, and Lexing Ying,
  *Compressed representation of kohn–sham orbitals via selected columns of the density matrix*
  Journal of Chemical Theory and Computation 11, 1463–1469 (2015).

* Anil Damle and L. Lin,
  *Disentanglement via entanglement: A unified method for wannier localization*,
  Multiscale Modeling & Simulation 16, 1392–1410 (2018).


## Available workflows

```
aiida_wannier90_workflows/
└── workflows
    ├── bands.py
    ├── __init__.py
    └── wannier.py
```

1. `bands.py` contains `Wannier90BandsWorkChain`, the automatic workflow that handles everything
2. `wannier.py` contains `Wannier90WorkChain`, a basic workflow that requires input parameters of every step: scf, nscf, pw2wan, projwfc, w90 pp, w90

## Installation

2. install this repository

   ```
   git clone https://github.com/aiidateam/aiida-wannier90-workflows.git
   cd aiida-wannier90-workflows/
   pip install -e .
   ```

4. Run the workflow

   ```
   cd examples/workflows/
   ./run_automated_wannier.py GaAs.xsf
   ```
   this script is for Quantum Mobile VM, for other machines please update these strings in the script
   ```
   str_pw = 'qe-6.5-pw@localhost'
   str_pw2wan = 'qe-6.5-pw2wannier90@localhost'
   str_projwfc = 'qe-6.5-projwfc@localhost'
   str_wan = 'wannier90-3.1.0-wannier@localhost'
   ```
