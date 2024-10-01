# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 02:12:01 2021

@author: Muqing Zheng
"""
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, vstack, hstack, diags, identity, block_diag
from scipy.sparse.linalg import spsolve

# import sympy as sym
# class QKalman2DT: # Specialize for linear state transition with quantum measurements
#     """
#     Based on the 2nd-order discrete time extended Kalman filter at book page 419-420 in the following book.
#     Simon, Dan. 2006. Optimal State Estimation: Kalman, H [Infinity] and Nonlinear Approaches. Hoboken, N.J: Wiley-Interscience.
#     ISBN: 978-0-471-70858-2
#     """
#     def __init__(self, x, P, Q, R, F):
#         self.k = -1 # number of iterations
        
#         self.x = x.copy() # State x_k, numpy array
#         self.nextXminus = None # x_{k+1},  a priori
#         self.y = None # observation y_k
#         self.m = self.x.size # the dimension of measurement vector(which is the same as it of state vector here)
        
#         self.P = P.copy() # covariance matrix
#         self.nextPminus = None # P_{k+1}^-
        
#         self.Q = Q.copy() # Process uncertainty covariance matrix
#         self.R = R.copy() # Measurement uncertainty covariance matrix
        
#         self.F = F.copy() # state transition matrix
        
#         # Observation function, its Jacobian and Hessians
#         self.var_x = sym.symbols('x_0:{}'.format(self.m))
#         self.h = sym.Matrix([self.var_x[j]**2/sum([self.var_x[i]**2 for i in range(self.m)]) for j in range(self.m)])
# #         self.h = sym.Matrix([self.var_x[j]**2 for j in range(self.m)]) # Not normalized observation function
#         self.h_jac = self.h.jacobian(self.var_x)
#         self.h_hess = [sym.hessian(self.h[j], self.var_x) for j in range(self.m)]

        
#     def phi_i(self,i):
#         res = np.zeros(self.x.shape)
#         res[i] = 1
#         return res
    
#     def time_update(self):
#         print("Start time update")
#         self.nextXminus = self.F.dot(self.x)
#         self.nextPminus = self.F.dot(self.P).dot(self.F.T) + self.Q
#         print("Finish time upate")
        
#     def meas_update(self, y):
#         print("Start measurement update")
#         H = np.array(self.h_jac.subs((self.var_x[j], self.nextXminus[j]) 
#                                      for j in range(self.m))).astype(np.float64) # still in real domain
#         temp = H.dot(self.nextPminus).dot(H.T) + self.R
#         K = self.nextPminus.dot(H.T).dot(np.linalg.inv(temp))

#         tempsum = np.zeros(self.m)
#         for i in range(self.m):
#             D = np.array(self.h_hess[i].subs((self.var_x[j], self.nextXminus[j]) 
#                                      for j in range(self.m))).astype(np.float64)  # still in real domain
#             tempsum += self.phi_i(i)*np.trace(D.dot(self.nextPminus))
#         pi = 0.5*K.dot(tempsum)
#         hxk = np.array(self.h.subs((self.var_x[j], self.nextXminus[j]) 
#                                    for j in range(self.m))).astype(np.float64).reshape(self.m)  # still in real domain
#         self.x = self.nextXminus + K.dot(y - hxk) - pi
#         self.P = (np.identity(self.m) - K.dot(H)).dot(self.nextPminus)
#         print("Finish measurement update")
        
#     def iterate(self, y):
#         self.k += 1
#         self.time_update() # get priori
#         self.meas_update(y) # get posteriori using the observation
#         return self.x # not normailized
    
    
    
    
def sparse_col_vec_dot(csc_mat, csc_vec):
    """
    From https://stackoverflow.com/questions/18595981/improving-performance-of-multiplication-of-scipy-sparse-matrices
    """
    # row numbers of vector non-zero entries
    v_rows = csc_vec.indices
    v_data = csc_vec.data
    # matrix description arrays
    m_dat = csc_mat.data
    m_ind = csc_mat.indices
    m_ptr = csc_mat.indptr
    # output arrays
    sizes = m_ptr.take(v_rows+1) - m_ptr.take(v_rows)
    sizes = np.concatenate(([0], np.cumsum(sizes)))
    data = np.empty((sizes[-1],), dtype=csc_mat.dtype)
    indices = np.empty((sizes[-1],), dtype=np.intp)
    indptr = np.zeros((2,), dtype=np.intp)

    for j in range(len(sizes)-1):
        slice_ = slice(*m_ptr[[v_rows[j] ,v_rows[j]+1]])
        np.multiply(m_dat[slice_], v_data[j], out=data[sizes[j]:sizes[j+1]])
        indices[sizes[j]:sizes[j+1]] = m_ind[slice_]
    indptr[-1] = len(data)
    ret = csc_matrix((data, indices, indptr),
                         shape=csc_vec.shape)
    ret.sum_duplicates()

    return ret




class QaugKF: # Specialize for linear state transition in quantum measurements
    """
     Augmented complex extended Kalman filter (ACEKF), 2nd order, discrete time.
     Linear state transition function.
     Observation function is squared complex norm.
     
     CR Calculus refers to https://arxiv.org/abs/0906.4835
     KF filter refers to ACEKF in 10.1109/TNNLS.2012.2189893
     
     x is the state to be estimated, dimension p*1, scipy sparse.csr_matrix array
     M is covariance matrix, M = E[(x-E[x])(x-E[x])^H], dimension p*p, scipy sparse.csr_matrix array
     Q is covariance of observation noise, dimension p*p, scipy sparse.csr_matrix array
     R is covariance of measurement noise, dimension p*p, scipy sparse.csr_matrix array
     P is pseudocovariance of observation noise, dimension p*p, scipy sparse.csr_matrix array
     U is pseudocovariance of measurement noise, dimension p*p, scipy sparse.csr_matrix array
    """
    def __init__(self, x, M, Q, R, P=None, U=None):
        self.k = 0 # number of iterations, start with 0      
        # State
        self.p = x.shape[0] # x \in C^p
        self.x = csc_matrix(x.reshape((self.p,1)),dtype=np.complex) # original x_k
        self.xa = vstack([self.x, self.x.conjugate()], format='csc')# Augmented x, x^a = [x^T, x^H]^T
        self.xa_prio = None # priori
        # Covariance of state
        self.M = csc_matrix(M,dtype=np.complex) # covariance matrix
        self.Ma = block_diag((self.M, self.M.conjugate()), format='csc') # argumented covariance matrix
        self.Ma_prio = None # priori
        # Noise of observation
        self.Q = csc_matrix(Q,dtype=np.complex)
        if P is None:
            self.P = csc_matrix((self.p, self.p), dtype=np.complex)
        else:
            self.P = csc_matrix(P,dtype=np.complex)
        self.Qa = vstack([hstack([self.Q, self.P], format='csc'),
                          hstack([self.P.conjugate(), self.Q.conjugate()], format='csc')],
                         format='csc')
        # Noise of measurement
        self.R = csc_matrix(R,dtype=np.complex)
        if U is None:
            self.U = csc_matrix((self.p, self.p), dtype=np.complex)
        else:
            self.U = csc_matrix(U,dtype=np.complex)
        self.Ra = vstack([hstack([self.R, self.U], format='csc'),
                          hstack([self.U.conjugate(), self.R.conjugate()], format='csc')],
                         format='csc')
        # State transition function
        self.Fa = None
        # Kalman Gain
        self.Ga = None
        
        
    def meas_func(self, state): # real-valued, so does not care about augmentation
        probs_vec = state.multiply(state.conjugate())
        return probs_vec/probs_vec.sum() # i.e. h(z) = zz^* for z \in C
        # return probs_vec
    
    
    def meas_Ha(self, state): # matrix H
        denom = state.multiply(state.conjugate()).sum()
        denom_sq = denom**2
        
        dense_state = np.array(state.todense())
        dfdz = dense_state[self.p:2*self.p].flatten()
        dfdzc = dense_state[0:self.p].flatten()
        # H = (1/denom) * diags(dfdz.flatten(), shape=(self.p, self.p), format='csc', dtype=np.complex) 
        # B = (1/denom) * diags(dfdzc.flatten(), shape=(self.p,self.p), format='csc', dtype=np.complex)
        H = np.zeros((self.p, self.p), dtype=np.complex)
        B = np.zeros((self.p, self.p), dtype=np.complex)
        for i in range(self.p):
            for j in range(self.p):
                if i == j:
                    H[i,j] = dfdz[i]/denom - dfdzc[i]*(dfdz[i]/denom)**2
                    B[i,j] = dfdzc[i]/denom - dfdz[i]*(dfdzc[i]/denom)**2
                else:
                    H[i,j] = dfdzc[i]*dfdz[i]*dfdzc[j]/denom_sq
                    H[i,j] = dfdz[i]*dfdzc[i]*dfdz[j]/denom_sq
        H = csc_matrix(H)
        B = csc_matrix(B)
        Ha =  vstack([hstack([H, B], format='csc'),
                      hstack([B.conjugate(), H.conjugate()], format='csc')],
                     format='csc')
        return Ha
    
    
    def meas_Hes(self, state): # Hessian matrices
        hess = []
        for i in range(self.p):
            row = np.array([i, self.p+i])
            col = np.array([i, self.p+i])
            data = np.array([1,1])
            hes = csc_matrix((data, (row, col)), shape=(2*self.p, 2*self.p))
            hess.append(hes)
        return hess
    
    
    def augF(self, F): # Augmented F, Fa = [[F,0],[0,F^H]]
        # A = csc_matrix((self.p, self.p))
        return block_diag((F, F.conjugate()), format='csc')
        
    
    def time_update(self, F):
        self.Fa = self.augF(F)
        # Prediction
        self.xa_prio = sparse_col_vec_dot(self.Fa, self.xa)
        # Prediction Covariance matrix
        self.Ma_prio = self.Fa.dot(self.Ma).dot(self.Fa.getH()) + self.Qa
     
        
    def meas_update(self, F, y):
        ya = vstack([y, y.conjugate()], format='csc')
        Ha = self.meas_Ha(self.xa_prio)
        Ha_Hem = Ha.getH().tocsc()
        # Kalman Gain, # G = MH(HMH + R)^-1 => (HMH + R)^T G^T = (MH)^T
        # So do not need to take inverse, faster!
        hmhr = Ha.dot(self.Ma_prio).dot(Ha_Hem) + self.Ra
        mh = self.Ma_prio.dot(Ha_Hem)
        self.Ga = spsolve(hmhr.transpose(), mh.transpose()).transpose() 
        ##################
        #2nd-order correction
        hesians = self.meas_Hes(self.xa_prio)
        trace_entries = np.array([0]*self.p,dtype=np.complex)
        for i in range(self.p):
            trace_entries[i] = hesians[i].dot(self.Ma_prio).diagonal().sum()
        aug_trace = np.append(trace_entries, trace_entries.conjugate())
        temp_vec = csc_matrix(aug_trace.reshape((2*self.p, 1)))
        pi = 0.5*sparse_col_vec_dot(self.Ga, temp_vec)
        #################
        # Correction
        self.xa = self.xa_prio + sparse_col_vec_dot(self.Ga, ya-self.meas_func(self.xa_prio)) # STILL FIRST ORDER!
        # Covariance matrix
        self.Ma = (identity(2*self.p, dtype=np.complex, format='csc') - self.Ga.dot(Ha)).dot(self.Ma_prio)
        
        
    def filtering(self, F, y):
        F = csc_matrix(F,dtype=np.complex)
        y = csc_matrix(y.reshape((self.p,1)),dtype=np.complex) 
        self.k += 1
        self.time_update(F) # update self.Fa in this function
        self.meas_update(F, y)
        
        # update non-augmented variables
        self.x = self.xa[0:self.p]
        self.M = self.Ma[np.array(range(self.p)),:]
        return self.x
    
    
    def cov_mat(self): # covariance matrix
        return self.M


    def itr_num(self): # number of iterations that measurement has done
        return self.k
        
    
    
    
if __name__ == "__main__":
    
    # Initial state
    # # Cannot initialize at [1,0] because the Jacobian of observation function will be zero matrix (due to normalization)
    # x = np.ones(2)*np.sqrt(0.01)
    # x[0]=np.sqrt(0.99)
    x = np.zeros(2)
    x[0] = 1
    
    # Other inputs
    P = np.identity(2)*0.01 # covariance matrix, E[(x0-xhat0^+)(x0-xhat0^+)^T]
    Q = np.identity(2)*0.01 # Process uncertainty/noise
    R = np.identity(2)*0.01 # measurement uncertainty/noise
    
    F = np.identity(2)  # state transition matrix
        
        
    filter_obj = QaugKF(x, P, Q, R)
    # State from the 1st iteration
    print(filter_obj.filtering(F, np.array([0.9,0.1])))
    
    
    
    
    
    
    
    
    
    