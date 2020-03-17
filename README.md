# aiida-wannier90-workflows
Advanced AiiDA workflows developed in the THEOS group for QE+Wannier90

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

The initial workflow was written by Antimo Marrazzo (EPFL) and Giovanni Pizzi (EPFL), it was later substantially improved and upgraded to AiiDA v1.1.1 by Junfeng Qiao (EFPL). The SCDM implementation in Quantum ESPRESSO was done by Valerio Vitale (Imperial College London and University of Cambridge).

## Available workflows

```
aiida_wannier90_workflows/
└── workflows
    ├── bands.py
    ├── band_structure.py
    ├── __init__.py
    └── wannier.py
```

1. `bands.py` contains `Wannier90BandsWorkChain`, the automatic workflow that handles everything
2. `wannier.py` contains `Wannier90WorkChain`, a basic workflow that requires input parameters of every step: scf, nscf, pw2wan, projwfc, w90 pp, w90
3. `band_structure.py` customized QE `PwBandStructureWorkChain`, remove relax step, used for comparing band structure with `Wannier90BandsWorkChain`

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
