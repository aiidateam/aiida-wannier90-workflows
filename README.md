# aiida-wannier90-workflows

Advanced AiiDA workflows for automated Wannierisation.

|     | |
|-----|----------------------------------------------------------------------------|
|Latest release| [![PyPI version](https://badge.fury.io/py/aiida-wannier90-workflows.svg)](https://badge.fury.io/py/aiida-wannier90-workflows) [![PyPI pyversions](https://img.shields.io/pypi/pyversions/aiida-wannier90-workflows.svg)](https://pypi.python.org/pypi/aiida-wannier90-workflows/) |
|Getting help| [![Docs status](https://readthedocs.org/projects/aiida-wannier90-workflows/badge)](http://aiida-wannier90-workflows.readthedocs.io/) [![Google Group](https://img.shields.io/badge/-Google%20Group-lightgrey.svg)](https://groups.google.com/forum/#!forum/aiidausers)
|Build status| [![Build Status](https://github.com/aiidateam/aiida-wannier90-workflows/actions/workflows/ci.yml/badge.svg)](https://github.com/aiidateam/aiida-wannier90-workflows/actions) [![Coverage Status](https://codecov.io/gh/aiidateam/aiida-wannier90-workflows/branch/main/graph/badge.svg)](https://codecov.io/gh/aiidateam/aiida-wannier90-workflows/tree/main) |
|Activity| [![PyPI-downloads](https://img.shields.io/pypi/dm/aiida-wannier90-workflows.svg?style=flat)](https://pypistats.org/packages/aiida-wannier90-workflows) [![Commit Activity](https://img.shields.io/github/commit-activity/m/aiidateam/aiida-wannier90-workflows.svg)](https://github.com/aiidateam/aiida-wannier90-workflows/pulse)


The protocol for automating the construction of Wannier functions is discussed in the following articles

* Junfeng Qiao, Giovanni Pizzi, Nicola Marzari,
  *Projectability disentanglement for accurate and automated electronic-structure Hamiltonians*, arXiv:2303.07877;
  <https://arxiv.org/abs/2303.07877>, <https://archive.materialscloud.org/record/2023.44>.
* Valerio Vitale, Giovanni Pizzi, Antimo Marrazzo, Jonathan Yates, Nicola Marzari, Arash Mostofi,
  *Automated high-throughput Wannierisation*, npj Computational Materials 6, 66 (2020);
  <https://arxiv.org/abs/1909.00433>, <https://www.nature.com/articles/s41524-020-0312-y>, <https://doi.org/10.24435/materialscloud:2019.0044/v2>.

## Installation

1. Install latest release by

   ```bash
   pip install aiida-wannier90-workflows
   ```

2. Or install the development version by

   ```bash
   git clone https://github.com/aiidateam/aiida-wannier90-workflows.git
   cd aiida-wannier90-workflows/
   pip install -e .
   ```
