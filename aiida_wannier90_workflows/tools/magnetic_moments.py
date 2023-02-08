import numpy as np
from aiida.orm import StructureData
def validate_moments(structure, magmoms):
    if isinstance(structure,StructureData):
        natom=len(structure.sites)
    else:
        natom=len(structure)
    
    if natom != len(magmoms):
        raise ValueError(f'Number of magnetic moments {magmoms} did not match with the structure.')
    for i, mom in enumerate(magmoms):
        if len(mom) != 3:
            raise ValueError(f'Magnetic moment at site {i} was not a 3D vector')
    return 

def is_collinear(magmoms, mom_prec_muB=1e-4,angle_prec_degree=1):
    quant_axis = None
    if len(magmoms) > 0:
        #get the q-axis
        for magmom in magmoms:
            mnorm = np.linalg.norm(magmom)
            if mnorm > mom_prec_muB:
                quant_axis = magmom/mnorm
                break
    
    if quant_axis is None:
        # Non-magnetic material -> return True
        return True
    
    #check for collinearity
    # If one of magmom is deviate from quant_axis, meaning non-collinear and return False
    for magmom in magmoms:
        mnorm = np.linalg.norm(magmom)
        if mnorm > mom_prec_muB:
            unit_m = magmom/mnorm
            if np.arcsin(np.linalg.norm(np.cross(unit_m,quant_axis)))*360/(2*np.pi) > angle_prec_degree:
                return False
    return True

def get_moments_m_theta_phi(magmoms):
    """
    Return a dictionary with the magnetic moments in the QuantumEspresso format
    """
    moms=[]
    thetas=[]
    phis=[]
    for mom in magmoms:
        m = np.linalg.norm(mom,ord=2)
        moms.append(m)
        if m > 1e-6:
            mtheta=np.arccos(mom[2]/m)
            mphi=np.arctan2(mom[1],mom[0])
            thetas.append(mtheta)
            phis.append(mphi)
        else:
            thetas.append(0)
            phis.append(0)

    return moms,thetas,phis

    # for i,mom in enumerate(magmoms):
    #     num=str(i+1)
    #     [mx,my,mz]=mom
    #     m = np.linalg.norm(mom,ord=2)
    #     if m > 1e-6:
    #         param_system['starting_magnetization('+num+')'] = float(m)
    #         if noncollinear:
    #             mtheta=np.arccos(mz/m)
    #             mphi=np.arctan2(my,mx)
    #             param_system['angle1('+num+')'] = float(mtheta)
    #             param_system['angle2('+num+')'] = float(mphi)
    # return param_system
