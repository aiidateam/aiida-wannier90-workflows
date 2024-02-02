"""Functions for processing spin-separated calculations."""
import pathlib
import time
import typing as ty

import numpy as np

from aiida import load_profile, orm

load_profile()

num_bands = 0  # num of bands
num_kpts = 0  # num of k-points
num_wann = 0  # num of wannier funcitons
nn_tot = 0  # num of total neighbors we count


def read_eig_and_output_rank(
    up_path: ty.Union[str, orm.RemoteData],
    dn_path: ty.Union[str, orm.RemoteData],
    output_path: str,
    up_seedname: str = "aiida",
    dn_seedname: str = "aiida",
    output_seedname: str = "aiida",
):
    """Read the eig matrix from paths, then rank them and combine them into one.

    :param *_bands: path where eig matrix stored
    :param *_seedname: seedname for eig matrix
    """
    up_eig, dn_eig, output_eig = check_file_exist(
        up_path,
        dn_path,
        output_path,
        file_type="eig",
        up_seedname=up_seedname,
        dn_seedname=dn_seedname,
        output_seedname=output_seedname,
    )
    # up/dn_eig : eig files for up/down components

    # Read eig file into list of (ibands, ikpts, eig),
    # then turn it into np.array
    m_up = []
    with open(up_eig, encoding="utf-8") as up:
        # Seedname.eig file donot have headers
        for line in up:
            ib, ik, eig = (t(s) for t, s in zip((int, int, float), line.split()))
            m_up.append([ib, ik, eig])
    m_up = np.array(m_up)

    m_dn = []
    with open(dn_eig, encoding="utf-8") as dn:
        for line in dn:
            ib, ik, eig = (t(s) for t, s in zip((int, int, float), line.split()))
            m_dn.append([ib, ik, eig])
    m_dn = np.array(m_dn)

    global num_bands, num_kpts  # pylint: disable=global-statement
    assert all(m_up[:, 0] == m_dn[:, 0])
    num_bands = int(max(m_up[:, 0]))
    assert all(m_up[:, 1] == m_dn[:, 1])
    num_kpts = int(max(m_up[:, 1]))

    # Reshape the eig matrix to (num_bands, num_kpts), and then rank it and output
    eig_up = np.reshape(m_up[:, 2], (num_bands, num_kpts), order="F")
    eig_dn = np.reshape(m_dn[:, 2], (num_bands, num_kpts), order="F")
    eig_rank = np.zeros((num_kpts, num_bands * 2), int)
    with open(output_eig, "w", encoding="utf-8") as out:
        for ik in range(num_kpts):
            ind = np.argsort(np.append(eig_up[:, ik], eig_dn[:, ik]), kind="stable")
            # ind[ib] indicates the
            eig_rank[ik, :] = ind
            for ib in range(2 * num_bands):
                if ind[ib] < num_bands:
                    out.write(f"{ib+1:5d}{ik+1:5d}{eig_up[ind[ib], ik]:18.12f}\n")
                else:
                    out.write(
                        f"{ib+1:5d}{ik+1:5d}{eig_dn[ind[ib]-num_bands, ik]:18.12f}\n"
                    )

    return eig_rank


def read_amn_and_output(
    up_path: ty.Union[str, orm.RemoteData],
    dn_path: ty.Union[str, orm.RemoteData],
    output_path: str,
    rank: np.ndarray,
    up_seedname: str = "aiida",
    dn_seedname: str = "aiida",
    output_seedname: str = "aiida",
):
    """Read the amn matrix from paths, then combine them into one.

    :param *_bands: path where amn matrix stored
    :param *_seedname: seedname for amn matrix
    """
    up_amn, dn_amn, output_amn = check_file_exist(
        up_path,
        dn_path,
        output_path,
        file_type="amn",
        up_seedname=up_seedname,
        dn_seedname=dn_seedname,
        output_seedname=output_seedname,
    )

    with open(up_amn, encoding="utf-8") as up, open(
        dn_amn, encoding="utf-8"
    ) as dn, open(output_amn, "w", encoding="utf-8") as out:
        # Throw away the Header and write local time to out
        up.readline()
        dn.readline()
        date = time.strftime("%d%b%Y", time.localtime())
        hms = time.strftime("%H:%M:%S", time.localtime())
        out.write(f"Combine up and down amn file into one on {date} at {hms}\n")
        b_up, k_up, w_up = (
            t(s) for t, s in zip((int, int, int), up.readline().split())
        )
        b_dn, k_dn, w_dn = (
            t(s) for t, s in zip((int, int, int), dn.readline().split())
        )
        # check amn matrix have same shape
        assert (b_up == b_dn) and (k_up == k_dn) and (w_up == w_dn)
        assert (b_up == num_bands) and (k_up) == num_kpts
        global num_wann  # pylint: disable=global-statement
        num_wann = w_up

        out.write(f"{2*num_bands:12d}{num_kpts:12d}{2*num_wann:12d}\n")
        for ik in range(num_kpts):
            ind = rank[ik, :]
        for iw in range(num_wann):
            # Seperate proj orb into orb_up and orb_dn
            # First the orb_up
            for ib in range(2 * num_bands):
                # Seperate wfc from (up, dn)
                # into (up, 0) and (0, dn)
                if ind[ib] < num_bands:
                    # (up, 0) -> orb_up
                    _, _, _, a_r, a_i = (
                        t(s)
                        for t, s in zip(
                            (int, int, int, float, float), up.readline().split()
                        )
                    )
                    out.write(
                        f"{ib+1:5d}{2*iw+1:5d}{ik+1:5d}" + f"{a_r:18.12f}{a_i:18.12f}\n"
                    )
                else:
                    # (0, dn) -> orb_up
                    out.write(
                        f"{ib+1:5d}{2*iw+1:5d}{ik+1:5d}"
                        + f"{float(0.0):18.12f}{float(0.0):18.12f}\n"
                    )
            # Then the orb_dn
            for ib in range(2 * num_bands):
                if ind[ib] < num_bands:
                    # (up, 0) -> orb_dn
                    out.write(
                        f"{ib+1:5d}{2*iw+2:5d}{ik+1:5d}"
                        + f"{float(0.0):18.12f}{float(0.0):18.12f}\n"
                    )
                else:
                    # (0, dn) -> orb_dn
                    _, _, _, a_r, a_i = (
                        t(s)
                        for t, s in zip(
                            (int, int, int, float, float), dn.readline().split()
                        )
                    )
                    out.write(
                        f"{ib+1:5d}{2*iw+2:5d}{ik+1:5d}" + f"{a_r:18.12f}{a_i:18.12f}\n"
                    )


def read_mmn_and_output(
    up_path: ty.Union[str, orm.RemoteData],
    dn_path: ty.Union[str, orm.RemoteData],
    output_path: str,
    rank: np.ndarray,
    up_seedname: str = "aiida",
    dn_seedname: str = "aiida",
    output_seedname: str = "aiida",
):
    """Read the mmn matrix from paths, then combine them into one.

    :param *_bands: path where mmn matrix stored
    :param *_seedname: seedname for mmn matrix
    """
    up_mmn, dn_mmn, output_mmn = check_file_exist(
        up_path,
        dn_path,
        output_path,
        file_type="mmn",
        up_seedname=up_seedname,
        dn_seedname=dn_seedname,
        output_seedname=output_seedname,
    )
    # mmn file:
    # Header...
    # num_bands num_kpts nn_tot
    # (
    #   (
    #       kpt nn_kpt (3-int describe the relationship about kpt lattice and nn_kpt lattice)
    #       num_bands * num_bands matrix
    #   ) * nn_tot for same kpt
    # ) * num_kpts
    with open(up_mmn, encoding="utf-8") as up, open(
        dn_mmn, encoding="utf-8"
    ) as dn, open(output_mmn, "w", encoding="utf-8") as out:
        # Throw away the Header and write local time to out
        up.readline()
        dn.readline()
        date = time.strftime("%d%b%Y", time.localtime())
        hms = time.strftime("%H:%M:%S", time.localtime())
        out.write(f"Combine up and down amn file into one on {date} at {hms}\n")

        b_up, k_up, nn_up = (
            t(s) for t, s in zip((int, int, int), up.readline().split())
        )
        b_dn, k_dn, nn_dn = (
            t(s) for t, s in zip((int, int, int), dn.readline().split())
        )
        # check amn matrix have same shape
        assert (b_up == b_dn) and (k_up == k_dn) and (nn_up == nn_dn)
        assert (b_up == num_bands) and (k_up == num_kpts)
        global nn_tot  # pylint: disable=global-statement
        nn_tot = nn_up
        out.write(f"{2*num_bands:12d}{num_kpts:12d}{nn_tot:12d}\n")
        for ik in range(num_kpts):
            for _ in range(nn_tot):
                up_line = up.readline()
                dn_line = dn.readline()
                ik_up, nn_k_up, _, _, _ = (
                    t(s) for t, s in zip((int, int, int, int, int), up_line.split())
                )
                assert up_line == dn_line
                assert ik_up == ik + 1
                indi = rank[ik_up - 1]
                indj = rank[nn_k_up - 1]
                out.write(up_line)
                for j in range(2 * num_bands):
                    for i in range(2 * num_bands):
                        if indi[i] < num_bands and indj[j] < num_bands:
                            out.write(up.readline())
                        elif indi[i] >= num_bands and indj[j] >= num_bands:
                            out.write(dn.readline())
                        else:
                            out.write(f"{float(0.0):18.12f}{float(0.0):18.12f}\n")


def check_file_exist(
    up_path: ty.Union[str, orm.RemoteData],
    dn_path: ty.Union[str, orm.RemoteData],
    output_path: str,
    file_type: str,
    up_seedname: str = "aiida",
    dn_seedname: str = "aiida",
    output_seedname: str = "aiida",
):
    """Check the paths, then return the full path of the matrix files.

    :param *_bands: path where matrix stored
    :param *_seedname: seedname for matrix
    :param file_type: type(suffix) of matrix we want to find
    :return up, dn, output: full path of up/down/output matrix files
    """
    if isinstance(up_path, orm.RemoteData):
        up_path = pathlib.Path(up_path.get_remote_path())
    else:
        up_path = pathlib.Path(up_path)
    up = up_path / (up_seedname + "." + file_type)

    if isinstance(dn_path, orm.RemoteData):
        dn_path = pathlib.Path(dn_path.get_remote_path())
    else:
        dn_path = pathlib.Path(dn_path)
    dn = dn_path / (dn_seedname + "." + file_type)

    output_path = pathlib.Path(output_path)

    if not up.exists():
        raise FileNotFoundError(f"{up} does not exist")
    if not dn.exists():
        raise FileNotFoundError(f"{dn} does not exist")
    if not output_path.is_dir():
        raise FileNotFoundError(f"{output_path} can not be found")

    output = output_path / (output_seedname + "." + file_type)

    return up, dn, output


def main():
    """Combine spin up and down matrix into one."""
    up_dir_path = orm.load_node(285637)
    dn_dir_path = orm.load_node(285638)
    # up_path = "/home/jiang_y/Softwares/wannier90-3.1.0/examples/example08/IronTest"
    # dn_path = "/home/jiang_y/Softwares/wannier90-3.1.0/examples/example08/IronTest"
    output_dir_path = pathlib.Path.home() / "test/magnetic"
    # up_seedname = "iron_up"
    # dn_seedname = "iron_dn"
    # output_seedname = "iron"
    eig_rank = read_eig_and_output_rank(
        up_path=up_dir_path,
        dn_path=dn_dir_path,
        output_path=output_dir_path,
        # up_seedname=up_seedname,
        # dn_seedname=dn_seedname,
        # output_seedname=output_seedname,
    )
    read_amn_and_output(
        up_path=up_dir_path,
        dn_path=dn_dir_path,
        output_path=output_dir_path,
        rank=eig_rank,
        # up_seedname=up_seedname,
        # dn_seedname=dn_seedname,
        # output_seedname=output_seedname,
    )
    read_mmn_and_output(
        up_path=up_dir_path,
        dn_path=dn_dir_path,
        output_path=output_dir_path,
        rank=eig_rank,
        # up_seedname=up_seedname,
        # dn_seedname=dn_seedname,
        # output_seedname=output_seedname,
    )


if __name__ == "__main__":
    main()
