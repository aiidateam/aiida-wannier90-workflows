
Pseudopotentials
^^^^^^^^^^^^^^^^

Running the Wannier90 workflows requires specifying which orbitals you will consider as semicore.
To do this automatically, the pseudo wave functions and which ones are considered semicore are stored in a JSON format for each pseudopotential family in the following directory:


.. code-block::

    src/aiida_wannier90_workflows/utils/pseudo/data/semicore

Adding support for more pseudopotentials means adding a new JSON file there where the following is specified for each element in the pseudopotential family:

1. The filename of the pseudopotential.
2. The ``md5`` hash of the file contents.
3. The pseudo wave functions (``pswfcs```) and chosen semicore states (``semicores``).

Here's an example:

.. code-block::

    "Ba": {
        "filename": "Ba.upf",
        "md5": "ecceda5fc736cf81896add41b7758c6c",
        "pswfcs": [
            "5S",
            "5P",
            "6S"
        ],
        "semicores": [
            "5S",
            "5P"
        ]
    },

The ``pswfcs`` are all the labels of the wave functions stored in the pseudo.
These can be found by looking for blocks like this one in the UPF file:

.. code-block::

    <PP_CHI.1
    type="real"
    size="1950"
    columns="4"
    index="1"
    occupation=" 2.000"
    pseudo_energy="   -0.2475035028E+01"
    label="5S"
    l="0" >

.. note::

    The above example is only valid for the PseudoDojo pseudopotentials.
    Other pseudopotentials may have different formats, and you may need to look for the labels and energies in other parts of the file.

Which of the ``pswfcs`` should be treated as semicore is determined somewhat heuristically.
Simply look at their energy level (``pseudo_energy``), in case it is much lower (an order of magnitude) than other wave functions, they can be considered semicore.

However, which orbitals should be considered semicore is in principle structure-dependent, and can fail in some cases.
It can only be really tested by using the choice and comparing the wannier-interpolated band structure with one directly calculated using Quantum ESPRESSO.

For the PseudoDojo, you can find the pseudopotentials at:

http://www.pseudo-dojo.org/pseudos/

In the `dev` directory at the root of this repository, you can find a script that can help you generate the JSON files for the PseudoDojo pseudopotentials, called `pseudos.py`.
Simply run it with the label of the pseudopotential family you want to generate the JSON file for, e.g.:

.. code-block::

    python pseudos.py PseudoDojo/0.4/PBE/SR/standard/upf

This will generate a JSON file in the `src/aiida_wannier90_workflows/utils/pseudo/data/semicore` directory.

.. important::

    The script currently only supports the PseudoDojo pseudopotentials.
