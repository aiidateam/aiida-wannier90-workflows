from upf_tools import UPFDict
from aiida_wannier90_workflows.utils.projectors.projectors import newProjector, newProjectors

import numpy as np
import warnings

class newUPFDict(UPFDict):
    """UPFDict for spin orbit coupling cases."""

    def to_dat(self) -> str:
        """"Override to_dat to fit soc.
        
        Generate a ``.dat`` file from a :class:`UPFDict` object.

        These files contain projectors that ``wannier90.x`` can read.

        :raises ValueError: The pseudopotential does not contain the pseudo-wavefunctions necessary to generate
            a ``.dat`` file

        :returns: the contents of a ``.dat`` file
        """
        # Fetch the r-mesh
        rmesh = self["mesh"]["r"]

        # Construct a logarithmic mesh
        min_r = 1e-10
        rmesh = [max(r, min_r) for r in rmesh]
        xmesh = np.log(rmesh)

        # Extract the pseudo wavefunctions, sorted by l and n
        if "chi" not in self["pswfc"]:
            raise ValueError("This pseudopotential does not contain any pseudo-wavefunctions")
        soc = self.has_so()
        # If soc: add j info to self
        if soc:
            if not "jchi" in self["pswfc"]["chi"]:
                wfclist = self["spin_orb"]["relwfc"]
                # wfc list should have same order as pswfc.chi
                if not isinstance(wfclist, list):
                    wfclist = [wfclist]
                for i, wfc in enumerate(wfclist):
                    if self["pswfc"]["chi"][i]["index"] != wfc["index"]:
                        raise ValueError("`PP_CHI/index` and `PP_RELWFC/index` do not match")
                    self["pswfc"]["chi"][i]["j"] = wfc["jchi"]
        if soc:
            chis = sorted(self["pswfc"]["chi"], key=lambda chi: (chi["l"], chi["j"], chi["n"]))
        else:
            chis = sorted(self["pswfc"]["chi"], key=lambda chi: (chi["l"], chi["n"]))
        data = np.transpose([chi["content"] for chi in chis])

        dat = [f"{len(rmesh)} {len(chis)}", " ".join([str(chi["l"]) for chi in chis])]
        if soc:
            dat += [" ".join([f"{chi['j']:4.1f}" for chi in chis])]
        dat += [
            f"{x:20.15f} {r:20.15f} " + " ".join([f"{v:25.15e}" for v in row])
            for x, r, row in zip(xmesh, rmesh, data)
        ]

        return "\n".join(dat)
    
    def to_projectors(self) -> newProjectors:
        """Generate newProjectors instance from upfDict.
        
        What's different: newProjector add support for soc and record labels of projectors"""
        # Fetch the r-mesh
        rmesh = self["mesh"]["r"]

        # Construct a logarithmic mesh
        min_r = 1e-10
        rmesh = [max(r, min_r) for r in rmesh]
        xmesh = np.log(rmesh)

        # Extract the pseudo wavefunctions, sorted by l and n
        if "chi" not in self["pswfc"]:
            raise ValueError("This pseudopotential does not contain any pseudo-wavefunctions")
        soc = self.has_so()
        # If soc: add j info to self
        if soc:
            if not "jchi" in self["pswfc"]["chi"]:
                wfclist = self["spin_orb"]["relwfc"]
                # wfc list should have same order as pswfc.chi
                if not isinstance(wfclist, list):
                    wfclist = [wfclist]
                for i, wfc in enumerate(wfclist):
                    if self["pswfc"]["chi"][i]["index"] != wfc["index"]:
                        raise ValueError("`PP_CHI/index` and `PP_RELWFC/index` do not match")
                    self["pswfc"]["chi"][i]["j"] = wfc["jchi"]
        if soc:
            chis = sorted(self["pswfc"]["chi"], key=lambda chi: (chi["l"], chi["j"], chi["n"]))
        else:
            chis = sorted(self["pswfc"]["chi"], key=lambda chi: (chi["l"], chi["n"]))
        # data = np.transpose([chi["content"] for chi in chis])

        if soc:
            projector = [
                newProjector(xmesh, chi["content"], int(chi["l"]), float(chi["j"]), label=chi["label"])
                for chi in chis
            ]
        else:
            projector = [
                newProjector(xmesh, chi["content"], int(chi["l"]), label=chi["label"])
                for chi in chis
            ]

        return newProjectors(projector)


    
    def has_so(self) -> bool:
        """Check if the system has spin orbit coupling."""

        try:
            has_so = self["header"]["has_so"]
        except KeyError:
            has_so = False
            warnings.warn("Can not find `have_so` in `PP_HEADER`,"
                          "assume the system as non soc")
        else:
            if isinstance(has_so, str):
                has_so = has_so[0].lower() == "t"
            
        return has_so
