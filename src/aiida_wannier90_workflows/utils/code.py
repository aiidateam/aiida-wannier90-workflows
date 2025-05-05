"""Functions for codes."""

import typing as ty

from aiida import orm


def identify_codes(
    codes: ty.Iterable[ty.Union[str, int, orm.Code]]
) -> ty.Dict[str, orm.Code]:
    """Given a list of PK, label, or ``Code``, generate a dict for code.

    E.g.,
    ````
    codes = identify_codes([
        "qe-git-pw@localhost",
        "qe-git-projwfc@localhost",
        "qe-git-pw2wannier90@localhost",
        "wannier90-git-wannier90@localhost",
        "qe-git-open_grid@localhost",
    ])
    ```
    returns
    ```
    codes = {
        "pw": "qe-git-pw@localhost",
        "projwfc": "qe-git-projwfc@localhost",
        "pw2wannier90": "qe-git-pw2wannier90@localhost",
        "wannier90": "wannier90-git-wannier90@localhost",
        "open_grid": "qe-git-open_grid@localhost",
    }
    ```

    :param codes: a list of code identifier
    :return: a dict with plugin name as key, ``Code`` as value.
    """
    results = {}

    if codes is None:
        return results

    for code in codes:
        if not isinstance(code, orm.Code):
            code = orm.load_code(code)

        plugin_name = code.default_calc_job_plugin

        # For simplicity I just use the last part,
        # e.g., wannier90.wannier90 -> wannier90
        name = plugin_name.split(".")[-1]

        results[name] = code

    return results


def check_codes(codes: dict, required_codes: ty.List[str] = None) -> None:
    """Check ``codes`` contains required keys.

    :param codes: _description_
    :type codes: dict
    :raises ValueError: _description_
    """
    if required_codes is None:
        # For SCDM Wannier90BandsWorkChain
        required_codes = ["pw", "projwfc", "pw2wannier90", "wannier90"]

    for code in required_codes:
        if code not in codes:
            raise ValueError(f"Code {code} is required")
