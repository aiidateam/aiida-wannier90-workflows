import numpy as np
from aiida import orm
from aiida.engine import calcfunction

@calcfunction
def convert_kpoints_mesh_to_list(kmesh):
    """works just like `kmesh.pl` in Wannier90
    
    :param kmesh: contains a N1 * N2 * N3 mesh
    :type kmesh: aiida.orm.KpointsData
    :raises AttributeError: if kmesh does not contains a mesh
    :return: an explicit list of kpoints
    :rtype: aiida.orm.KpointsData
    """
    try:  # test if it is a mesh
        results = kmesh.get_kpoints_mesh()
    except AttributeError as e:
        e.args = ('input does not contain a mesh!', )
        raise e
    else:
        # currently offset is ignored
        mesh = results[0]

        # following is similar to wannier90/kmesh.pl
        totpts = np.prod(mesh)
        weights = np.ones([totpts]) / totpts

        kpoints = np.zeros([totpts, 3])
        ind = 0
        for x in range(mesh[0]):
            for y in range(mesh[1]):
                for z in range(mesh[2]):
                    kpoints[ind, :] = [x / mesh[0], y / mesh[1], z / mesh[2]]
                    ind += 1
        klist = orm.KpointsData()
        klist.set_kpoints(kpoints=kpoints, cartesian=False, weights=weights)
        return klist