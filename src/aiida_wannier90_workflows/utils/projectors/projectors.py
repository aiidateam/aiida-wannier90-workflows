from upf_tools.projectors import Projector, Projectors
from dataclasses import dataclass, field

from aiida_pseudo.data.pseudo.upf import UpfData
from aiida import orm, load_profile
from typing import Union
from pathlib import Path
import numpy as np

num2label = {
    0: "s",
    1: "p",
    2: "d",
    3: "f"
}


@dataclass
class newProjector(Projector):
    _x_min: float = field(default=-25, init=False, repr=False)
    j: float = field(init=False) # for spin orbit coupling
    _j: float = field(init=False, repr=False)
    label: str = field(init=True, default=None)
    _label: str = field(init=False, repr=False)
    alfa: str = field(init=True, default="UPF")

    @property
    def j(self):
        if np.abs(self._j) < 1e-8:
            raise AttributeError("j is not an attribute.")
        else:
            jout = self._j
        return jout
    
    @j.setter
    def j(self, value):
        if not isinstance(value, float):
            self._j = 0.0
        elif np.abs(self.l - value) - 0.5 < 1e-8:
            self._j = value
        else:
            raise ValueError(
                f"l={self.l}, j={self._j}"
            )
    
    @property
    def label(self):
        return self._label
    
    @label.setter
    def label(self, value):
        if not isinstance(value, str):
            self._label = "0" + num2label[self.l]
            return
        if len(value) != 2:
            raise ValueError(f"label<{value}> must be 2-digit")
        nshell = value[0]
        orbital = value[1]
        if not nshell.isdigit():
            raise ValueError(f"first digit of label<{nshell}> must be number")
        if orbital.isalpha():
            orbital = orbital.lower()
            if orbital not in ["s", "p", "d", "f"]:
                raise ValueError(f"second digit must be s, p, d or f")
        else:
            raise ValueError(f"second digit of label<{nshell}> must be alphabet")
        
        self._label = nshell + orbital
    # def __init__(self, x, y, l, j=0.0) -> None:
    #     super().__init__(x, y, l)
    #     self.j = j


class newProjectors(Projectors):
    """A list of projectors, with more extra functionalities."""

    def to_str_soc(self) -> str:
        """Convert the Projectors into a string following the format for ``pw2wannier90`` and ``Wannier90``."""
        lines = [
            f"{len(self.data[0].x)} {len(self.data)}",
            " ".join([f"{proj.l:3d}" for proj in self.data]),
            " ".join([f"{proj.j:.1f}" for proj in self.data])
        ]
        content = np.concatenate(
            [np.vstack([self.data[0].x, self.data[0].r]), [p.y for p in self.data]]
        ).transpose()
        lines += [" ".join([f"{v:18.12e}" for v in row]) for row in content]
        return "\n".join(lines)

    def to_file(self, filename: Path):
        """Dump the Projectors to a file following the format for ``pw2wannier90`` and ``Wannier90``."""
        with open(filename, "w") as fd:
            try:
                self[0].j
            except AttributeError:
                fd.write(self.to_str())
            else:
                fd.write(self.to_str_soc())

    #
    # Use myUPFDict.to_projectors instead of newProjectors.from_upfdata
    #
    # @classmethod
    # def from_upfdata(cls, upfdata:Union[UpfData, int]):
    #     """Create a Projectors object from an AiiDA UPFData"""

    #     load_profile()
    #     #check if upfdata is a formatted UPFData or the pk of upf
    #     if not isinstance(upfdata, UpfData):
    #         upfdata = orm.load_node(upfdata)
    #         if not isinstance(upfdata, UpfData):
    #             raise ValueError(
    #                 "Input must be either `UpfData` or its pk."
    #                 f"But the input is {upfdata.__class__}"
    #             )
            
    #     # get content from UpfData and convert it to Projectors
    #     upf_str = upfdata.get_content()
    #     upf_dict = myUPFDict.from_str(upf_str)

    #     if upf_dict.has_so():
    #         projector = cls.from_str_soc(upf_dict.to_dat())
    #     else:
    #         projector = cls.from_str(upf_dict.to_dat())

    #     return projector
        
    
    @classmethod
    def from_pao(cls, filename: Union[Path, str], n: int, l: int, label: str=None):
        """Create a Projectors object from an openMX flavor PAO file"""

        filename = filename if isinstance(filename, Path) else Path(filename)

        # copy from gen_pao.py
        with open(filename, "r") as fin:
            data = []
            in_tag = False
            for line in fin:
                if f"<pseudo.atomic.orbitals.L={l}" in line:
                    in_tag = True
                elif f"pseudo.atomic.orbitals.L={l}>" in line:
                    in_tag = False
                elif in_tag:
                    data.append(line)
        
        # have read the data from <pseudo.atomic.orbitals.L={l}>
        x = []
        y = []
        for line in data:
            line_pao = np.fromstring(line, sep=" ")
            # line_pao:
            # x  r  n=1 n=2 n=3 ...
            x.append(line_pao[0])
            y.append(line_pao[n+2])
        x = np.array(x)
        y = np.array(y)
        data_pao = [newProjector(x, y, l, label=label)]
        return cls(data_pao)

    @classmethod
    def from_str_soc(cls, string: str) -> "Projectors":
        """Create a Projectors object from a string that follows the format for ``pw2wannier90`` and ``Wannier90``."""
        lines = [l for l in string.split("\n") if l]
        content = np.array([[float(v) for v in row.split()] for row in lines[3:]]).transpose()
        lvals = [int(l) for l in lines[1].split()]
        jvals = [float(j) for j in lines[2].split()]
        data = [newProjector(content[0], y, l, j) for l, j, y in zip(lvals, jvals, content[2:])]

        return cls(data)

    def add_projector(self, projector: newProjector):
        """Add a projector element to self."""
        
        self.__iadd__([projector])

    def add_projector_soc(self, projector: newProjector):
        """Add a projector element to self.
        
        If the projector has only l, split it to different j with same radials funtction."""
        try:
            projector.j
        except AttributeError:
            if projector.l == 0:
                self.__iadd__(
                    [newProjector(
                        x=projector.x,
                        y=projector.y,
                        l=projector.l,
                        j=projector.l+0.5,
                        label=projector.label,
                        alfa=projector.alfa
                    )]
                )
            else:
                projector_minus = newProjector(
                    x=projector.x,
                    y=projector.y,
                    l=projector.l,
                    j=projector.l-0.5,
                    label=projector.label,
                    alfa=projector.alfa
                )
                projector_plus = newProjector(
                    x=projector.x,
                    y=projector.y,
                    l=projector.l,
                    j=projector.l+0.5,
                    label=projector.label,
                    alfa=projector.alfa
                )
                self.__iadd__([projector_minus, projector_plus])
        else:
            self.__iadd__([projector])

    def remove_low(self, l: int):
        """Remove the lowest orbital with specific l"""

        have_removed = False
        for i, projector in enumerate(self.data):
            # Because the projectors was sortted by l, n
            # It is safe to just remove the lowest pswfc 
            if projector.l == l and not have_removed:
                self.__delitem__(i)
                have_removed = True

        if not have_removed:
            raise ValueError(f"Can not remove pswfcs with l={l}")




