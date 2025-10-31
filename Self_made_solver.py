from typing import List, Dict, Optional, Tuple
import numpy as np
from qutip import Qobj, liouvillian, operator_to_vector, vector_to_operator, expect
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import expm_multiply

def _qobj_super_to_scipy_csr(S: Qobj) -> csr_matrix:
   
    
    data = S.data
    # If bereits SciPy-sparse ist 
    if issparse(data):
        return data.tocsr()
    # QuTiP  Data zu ndarray
    to_array = getattr(data, "to_array", None)
    if callable(to_array):
        arr = to_array()
        return csr_matrix(arr.astype(np.complex128, copy=False))
    # 
    as_scipy = getattr(data, "as_scipy", None)
    if callable(as_scipy):
        return as_scipy().tocsr()
    #  versuche numpy-Array-Konstruktion
    return csr_matrix(np.array(data, dtype=np.complex128))

def solve_liouvillian_expm(
    H: Qobj,
    rho0: Qobj,
    tlist: np.ndarray,
    c_ops: List[Qobj],
    Auslassen:int
    #e_ops: Optional[List[Qobj]] = None,
) -> Tuple[List[Qobj], Optional[Dict[int, np.ndarray]]]:

    tlist = np.asarray(tlist, dtype=float)
    if tlist.ndim != 1 or (np.diff(tlist) < -1e-15).any():
        raise ValueError("tlist muss 1D und aufsteigend sein.")

    #  QuTiP to SciPy Konvertierung 
    L_sp = _qobj_super_to_scipy_csr(liouvillian(H, c_ops))

    # Anfangszustand als Liouville-Vektor
    v0_q = operator_to_vector(rho0)
    v = np.asarray(v0_q.full(), dtype=np.complex128).reshape(-1)
    vec_dims = v0_q.dims

    states: List[Qobj] = []
    #states_red: List[Qobj] = []
    #expt: Optional[Dict[int, np.ndarray]] = None
    #if e_ops:
    #    expt = {i: np.empty_like(tlist, dtype=float) for i in range(len(e_ops))}

    times0 = tlist - tlist[0]
    is_uniform = np.allclose(np.diff(tlist), np.diff(tlist)[0], rtol=1e-15, atol=1e-14)

    if is_uniform: # gleichmässige abstände 
        res = expm_multiply(L_sp, v, start=0.0, stop=times0[-1], num=len(tlist), endpoint=True)
        for k, vk in enumerate(res):
            rho = vector_to_operator(Qobj(vk.reshape(-1, 1), dims=vec_dims))
            #print(k)
            if k % Auslassen==0 :
                states.append(rho)
            #states_red.append(rho.ptrace(2))
            #if expt is not None:
            #    for i, op in enumerate(e_ops):
            #        expt[i][k] = float(expect(op, rho))
    else:
        t_prev = tlist[0]
        rho = vector_to_operator(Qobj(v.reshape(-1, 1), dims=vec_dims))
        #if expt is not None:
        #    for i, op in enumerate(e_ops):
        #        expt[i][0] = float(expect(op, rho))
        for k in range(0, len(tlist)):
            dt = tlist[k] - t_prev
            v = expm_multiply(L_sp, v, start=0.0, stop=dt, num=2, endpoint=True)[-1]
            t_prev = tlist[k]
            rho = vector_to_operator(Qobj(v.reshape(-1, 1), dims=vec_dims))
            if k % Auslassen==0 :
                states.append(rho)
            #states_red.append(rho.ptrace(2))
            #if expt is not None:
            #    for i, op in enumerate(e_ops):
            #        expt[i][k] = float(expect(op, rho))
    return states#, expt
