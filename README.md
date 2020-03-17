# aiida-wannier90-theosworkflows
Advanced AiiDA workflows developed in the THEOS group for QE+Wannier90

## available workflows

```
aiida_wannier90_theosworkflows/
└── workflows
    ├── bands.py
    ├── band_structure.py
    ├── __init__.py
    └── wannier.py
```

1. `bands.py` contains `Wannier90BandsWorkChain`, the automatic workflow that handles everything
2. `wannier.py` contains `Wannier90WorkChain`, a basic workflow that requires input parameters of every step: scf, nscf, pw2wan, projwfc, w90 pp, w90
3. `band_structure.py` customized QE `PwBandStructureWorkChain`, remove relax step, used for comparing band structure with `Wannier90BandsWorkChain`

## launch script

```
examples/workflows/run_automated_wannier.py
```

## Installation

1. I subclassed the `BaseRestartWorkChain` to a `Wannier90BaseWorkChain`, which can automatically handle Wannier90 error e.g. `Not enough bvectors found after several trials of kmesh_tol`, `Unable to satisfy B1`. So please first install this branch `https://github.com/qiaojunfeng/aiida-wannier90.git` before using this workflow

   ```bash
   git clone https://github.com/qiaojunfeng/aiida-wannier90.git
   cd aiida-wannier90/
   git checkout restart_workchain
   pip uninstall aiida-wannier90
   pip install -e .
   ```

2. then install this repository

   ```
   git clone git@github.com:aiidateam/aiida-wannier90-workflows.git
   cd aiida-wannier90-theosworkflows/
   pip install -e .
   ```

3. For Quantum Mobile VM
   There are some issues with the [newest VM](https://github.com/marvel-nccr/quantum-mobile/issues/107#issuecomment-596251969), I have collected my solution [here](https://github.com/marvel-nccr/quantum-mobile/issues/107#issuecomment-597140222). To run the launch script in the VM, those problems needs to be fixed.
   
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
