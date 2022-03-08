# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 02:04:03 2021

@author: Muqing Zheng
"""
import numpy as np
import sympy as sym
from scipy.sparse import csc_matrix, csr_matrix, vstack, hstack, diags, identity, block_diag, kron
from scipy.sparse.linalg import spsolve, splu, norm
from scipy.linalg import det, eig

def dictToVec(counts):
    """ Transfer counts to probabilities

    Args:
      counts: dict
        an dictionary in the form {basis string: frequency}. E.g.
        {"01": 100
         "11": 100}
        dict key follow little-endian convention

    Returns: numpy array
      an probability vector (array). E.g.
      [0, 0.5, 0, 0.5] is the result from example above.
    """
    nQubits = len(list(counts.keys())[0])
    vec = np.zeros(2**nQubits)
    form = "{0:0" + str(nQubits) + "b}"
    total_counts = 0
    for i in range(2**nQubits):
        key = form.format(i) # consider key = format(i,'0{}b'.format(nQubits))
                             # and delete variable form
        if key in counts.keys():
            vec[i] = int(counts[key])
            total_counts +=  int(counts[key]) 
        else:
            vec[i] = 0
    return vec/total_counts



def closed_den_mat(her_mat, norm_diag = False):
    """
    "Fast algorithm for Subproblem 1" in https://arxiv.org/abs/1106.5458
    Given a trace-one *Hermitian* matrix mu, find the closest density matrix rho
        
    density matrix: a trace-one Hermitian matrix with only non-negative egivenvalues (PSD)
    
    Parameters
    ----------
    her_mat : numpy.ndarray or scipy.csc_matrix
        A real-dignoal hermitian matrix, trace will be normalized to 1 in this function by setting norm_diag = False.
    norm_diag : boolean
        If norm_diag = True, the matrix her_mat has normalized diagonal. Otherwise, need to do normalization first

    Returns
    -------
    a density matrix in numpy.matrix

    """
    d, _ = her_mat.shape
    if not norm_diag:
        # normalize diagonal
        diags = her_mat.diagonal()
        diagsum = diags.sum()
        diags_normed = diags/diagsum
        try: 
            np.fill_diagonal(her_mat, diags_normed)
        except:
            her_mat.setdiag(diags_normed)
        
    # Get eigenvalues and eigenvectors
    try: # if numpy.ndarray
        eigvl, eigvc = eig(her_mat)
    except: # if scipy.csc_matrix
        eigvl, eigvc = eig(her_mat.toarray())
    eigvl = eigvl.real # just to not use complex form
    # Sort eigenvalues
    index = eigvl.argsort()[::-1]   
    eigvl = eigvl[index]
    
    acc = 0 # an accumulator
    final_eigvl = eigvl+0 # lambdas in the paper
    i = d # note that i is an index starting from 1, 1 <= i <= d
    while i > 0:
        temp_eigvl = eigvl[i-1] + acc/i
        if temp_eigvl >=0:
            for j in range(i):
                final_eigvl[j] = eigvl[j] + acc/i
            break
        else:
            final_eigvl[i-1] = 0
            acc += eigvl[i-1]
            i -= 1
    
    # Generate new density matrix
    final_den = np.zeros((d,d), dtype=complex)
    for k in range(d):
        vec_eigvc = np.matrix(eigvc[k], dtype=complex).reshape((d,1))
        final_den = final_den + final_eigvl[k] * vec_eigvc.dot(vec_eigvc.H)
        
    return final_den
        
    
    


def sparse_col_vec_dot(csc_mat, csc_vec):
    """
    Modify from
    https://stackoverflow.com/questions/18595981/improving-performance-of-multiplication-of-scipy-sparse-matrices
    Change the shape of ret
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
                         shape=(csc_mat.shape[0],1))
    ret.sum_duplicates()

    return ret


def hamming_weight(n,num_qubits):
    # Modified from https://www.tutorialspoint.com/number-of-1-bits-in-python
    form = "{0:0" + str(num_qubits) + "b}"
    n =  form.format(n)
    one_count = 0
    for i in n:
      if i == "1":
         one_count+=1
    return one_count


class KSQS: # quantum system kalman smoother, state is vectorized density matrix
    """
     Augmented complex extended Kalman smoother
     Linear state transition function.
     Observation function is element-wise squared complex norm.
     
     CR Calculus refers to https://arxiv.org/abs/0906.4835
     KF filter refers to ACEKF in 10.1109/TNNLS.2012.2189893
     Smoother refers to 
     1. **Smoothed State prior covariance matrix** -> Appendix A in R. H. Shumway and D. S. Stoffer, “AN APPROACH TO TIME SERIES SMOOTHING AND FORECASTING USING THE EM ALGORITHM,” doi: 10.1111/j.1467-9892.1982.tb00349.x
     2. **Time varying F and H** -> H. E. Rauch, F. Tung, and C. T. Striebel, “Maximum likelihood estimates of linear dynamic systems,” AIAA Journal, vol. 3, no. 8, pp. 1445–1450, Aug. 1965, doi: 10.2514/3.3166.
     3. **additional recources**Algorithm 4.1 (page 88) in S. Särkkä, “Bayesian Estimation of Time-Varying Systems: Discrete-Time Systems.” 

     
     x_t = F x_t-1 + w_t, w_t \sim N(0, Q, P)
     y_t = h(x_t) + v_t, v_t \sim N(0, R, U) # U should be a zero matrix since measurement noise is real
     
     x is the state to be estimated, dimension p*1, scipy sparse.csc_matrix array
     M is covariance matrix, M = E[(x-E[x])(x-E[x])^H], dimension p*p, scipy sparse.csc_matrix array
     Q is covariance of observation noise, dimension p*p, scipy sparse.csc_matrix array
     R is covariance of measurement noise, dimension p*p, scipy sparse.csc_matrix array
     P is pseudocovariance of observation noise, dimension p*p, scipy sparse.csc_matrix array
     U is pseudocovariance of measurement noise, dimension p*p, scipy sparse.csc_matrix array
    """
    def __init__(self, ys, F, x, M, Q, R, P=None, Augmented = False):
        self.k = 0 # number of iterations, start with 0      
        self.ys = ys
        self.xa_prio = None # priori
        self.Ma_prio = None # priori
        if Augmented:
            self.num_qubits = int(np.log2(np.sqrt(x.shape[0])))
            self.p = int(x.shape[0]/2) # x \in C^p
            self.q = ys[0].shape[0]
            self.xa = x.copy()
            self.xa_initial = x.copy()
            self.Ma = M.copy()
            self.Ma_initial = M.copy()
            self.Qa = Q.copy()
            self.Ra = R.copy()
            self.Fa = F.copy()
        else:
            # If not augumented, automatically do augmentation
            # State
            self.num_qubits = int(np.log2(np.sqrt(x.shape[0])))
            self.p = x.shape[0] # x \in C^p
            self.q = ys[0].shape[0]
            self.x = csc_matrix(x.reshape((self.p,1)),dtype=complex) # original x_k
            self.xa = vstack([self.x, self.x.conjugate()], format='csc')# Augmented x, x^a = [x^T, x^H]^T
            self.xa_initial = self.xa.copy()
            # Covariance of state
            self.M = csc_matrix(M) # covariance matrix
            self.Ma = block_diag((self.M, self.M.conjugate()), format='csc') # argumented covariance matrix
            self.Ma_initial = self.Ma.copy()
            # Noise of observation
            self.Q = csc_matrix(Q)
            if P is None:
                self.P = csc_matrix((self.p, self.p))
            else:
                self.P = csc_matrix(P)
            self.Qa = vstack([hstack([self.Q, self.P], format='csc'),
                              hstack([self.P.conjugate(), self.Q.conjugate()], format='csc')],
                             format='csc')
            # Noise of measurement
            self.R = csc_matrix(R)
            self.U = csc_matrix((self.q, self.q))
            self.Ra = vstack([hstack([self.R, self.U], format='csc'),
                              hstack([self.U.conjugate(), self.R.conjugate()], format='csc')],
                             format='csc')
            # State transition function
            self.F = csc_matrix(F,dtype=complex)
            self.Fa = block_diag((self.F, self.F.conjugate()), format='csc')
        # Kalman Gain (for measurement update)
        self.Ga = None
        # For smoother
        self.xa_prio_seq = []
        self.xa_seq = []
        
        self.Ma_prio_seq = []
        self.Ma_seq = []
        
        self.xa_smooth_seq = []
        self.Ma_prio_smooth_seq = [] # for EM algorithm
        self.Ma_smooth_seq = []
        
        self.H = self.meas_mat(self.num_qubits)
        self.Ha = block_diag((self.H, self.H.conjugate()), format='csc')
        
    def meas_mat(self, num_qubits):# H, measurement matrix for vectorized density matrix
        nrows = 2**num_qubits
        ncols = nrows**2
        mat = csc_matrix((nrows, ncols), dtype=complex)
        for k in range(nrows):
            mat[k, nrows*k+k] = 1 # take out the diagonal terms in vectorized density matrix
        return mat
        
    def meas_func(self, state): # real-valued, so does not care about augmentation
        prob_vec = sparse_col_vec_dot(self.Ha, state) 
        return prob_vec.real
        
    
    def time_update(self):
        # Non-zero mean for state transition noise
        # noise_scale = 0.0
        # noise_mean = np.zeros(self.p*2)
        # for i in range(int(len(noise_mean)/2)):
        #     # if abs(sparse_col_vec_dot(self.Fa, self.xa)[i]) > 0:
        #     noise_mean[i] = (int(np.log2(self.p)) - 2*hamming_weight(i,int(np.log2(self.p)))) * noise_scale
        #     noise_mean[self.p+i] = noise_mean[i]+0
        # noise_mean = csc_matrix(noise_mean.reshape((self.p*2,1)),dtype=complex)
        # Prediction
        # self.xa_prio = sparse_col_vec_dot(self.Fa, self.xa)
        # self.xa_prio = sparse_col_vec_dot(self.Fa, self.xa) - np.linalg.norm(np.array(self.xa.todense()[:self.p]), ord=2)*noise_mean
        self.xa_prio = sparse_col_vec_dot(self.Fa, self.xa)
        # Prediction Covariance matrix
        self.Ma_prio = self.Fa.dot(self.Ma).dot(self.Fa.getH().tocsc()) + self.Qa
     
        
    def meas_update(self,y):
        ya = vstack([y, y.conjugate()], format='csc')
        Ha_Hem = self.Ha.getH().tocsc()
        # Kalman Gain, # G = MH^H(HMH^H + R)^-1 => (HMH^H + R)^T G^T = (MH)^T
        # So do not need to take inverse, faster!
        hmhr = self.Ha.dot(self.Ma_prio).dot(Ha_Hem) + self.Ra
        mh = self.Ma_prio.dot(Ha_Hem)
        self.Ga = spsolve(hmhr.transpose(), mh.transpose()).transpose() 
        # Correction
        self.xa = self.xa_prio + sparse_col_vec_dot(self.Ga, ya-self.meas_func(self.xa_prio)) # STILL FIRST ORDER!
        # Covariance matrix
        self.Ma = (identity(2*self.p, dtype=complex, format='csc') - self.Ga.dot(self.Ha)).dot(self.Ma_prio)
        
        
    def filtering(self, y):
        y = csc_matrix(y.reshape((self.q,1)),dtype=complex) 
        self.k += 1
        self.time_update() # update self.Fa in this function
        self.meas_update(y)
        
        # update non-augmented variables
        self.x = self.xa[0:self.p]
        self.M = self.Ma[np.array(range(self.p)),:][:,np.array(range(self.p))]
        return self.xa, self.xa_prio, self.Ma, self.Ma_prio
    
    #### NOTE smoothed x and M includes BOTH AT TIME 0, BUT filter outputs do NOT  #####
    def smooth(self): # for all time together
        T = len(self.ys) # number of time step
        ##### Kalman filter (forward pass) ######
        self.xa_seq.append(self.xa_initial)
        self.Ma_seq.append(self.Ma_initial)
        self.xa_prio_seq.append(None) # does not have prior for time 0
        self.Ma_prio_seq.append(None) # does not have prior for time 0
        for t in range(T):
            xa_post, xa_prio, Ma_post, Ma_prio = self.filtering(self.ys[t])
            # Record state, priori M, and posteriori M
            self.xa_seq.append(xa_post)
            self.xa_prio_seq.append(xa_prio)
            self.Ma_prio_seq.append(Ma_post)
            self.Ma_seq.append(Ma_prio)
        last_gain_meas = self.Ga.copy()
        last_Ha = self.Ha.copy()
        ###########
        # Initialize
        self.xa_smooth_seq.append(self.xa_seq[-1].copy()) # at the last time step, smoothed one is the posterior
        self.Ma_smooth_seq.append(self.Ma_seq[-1].copy())
        
        iden_mat = identity(2*self.p, dtype=complex, format='csc') 
        last_Mk_prio_smooth = (iden_mat - last_gain_meas.dot(self.Ha)).dot(self.Fa).dot(self.Ma_seq[T-1])
        self.Ma_prio_smooth_seq.append(last_Mk_prio_smooth)
        
        # Backward iteration
        for t_inv in range(T, 0, -1): # the real time, not index in python
            xk_prior = self.xa_prio_seq[t_inv]
            Mk_prior = self.Ma_prio_seq[t_inv]
            Mkm1FHem_transpose = self.Fa.conjugate().dot(self.Ma_seq[t_inv-1].transpose())
            Gkm1 = spsolve(Mk_prior.transpose(), Mkm1FHem_transpose) # new Gain from x (instead of from y, like the one in the filter)
            
            xk_smooth = self.xa_seq[t_inv-1] + sparse_col_vec_dot(Gkm1, (self.xa_smooth_seq[0] - xk_prior))
            Mk_smooth = self.Ma_seq[t_inv-1] + Gkm1.dot(self.Ma_smooth_seq[0] - Mk_prior).dot(Gkm1.getH().tocsc())
            self.xa_smooth_seq.insert(0,xk_smooth)
            self.Ma_smooth_seq.insert(0,Mk_smooth)
            
            # also compute smoothed prior M for EM algorithm
            # Need to compute Gk(gain from x) that is one step before
            if t_inv > 1:
                Gkm2 = spsolve(self.Ma_prio_seq[t_inv-1].transpose(), 
                                   self.Fa.conjugate().dot(self.Ma_seq[t_inv-2].transpose()))
                Gkm2_Hem = Gkm2.getH().tocsc()
                Mk_prio_smooth = self.Ma_seq[t_inv-1].dot(Gkm2_Hem) +\
                                Gkm1.dot(self.Ma_prio_smooth_seq[0] - self.Fa.dot(self.Ma_seq[t_inv-1])).dot(Gkm2_Hem)
                self.Ma_prio_smooth_seq.insert(0,Mk_prio_smooth)
        return self.xa_smooth_seq, self.Ma_smooth_seq, self.Ma_prio_smooth_seq
    
    
    
    
class EMLearn:
    """
    EM algorithm for augmented complex first-order Kalman filter with RTS smoother.
    Refers to R. H. Shumway and D. S. Stoffer, “AN APPROACH TO TIME SERIES SMOOTHING AND FORECASTING USING THE EM ALGORITHM,” doi: 10.1111/j.1467-9892.1982.tb00349.x
    
     x_t = F x_t-1 + w_t, w_t \sim N(0, Q, P)
     y_t = h(x_t) + v_t, v_t \sim N(0, R, U) --> linearized to y_t = H_t x_t + v_t,
     where H_t is the Jacobian matrix of h evaluated at x_{t|t-1} (i.e., prior estimate of x_t)
     
     xhat0 is E[x0], dimension p*1, scipy sparse.csr_matrix array
     M0 is covariance matrix, M = E[(x0-E[x0])(x0-E[x0])^H], dimension p*p, scipy sparse.csr_matrix array
     Q is covariance of observation noise, dimension p*p, scipy sparse.csr_matrix array
     R is covariance of measurement noise, dimension p*p, scipy sparse.csr_matrix array
     P is pseudocovariance of observation noise, dimension p*p, scipy sparse.csr_matrix array
     U is pseudocovariance of measurement noise (should be a zero matrix), dimension p*p, scipy sparse.csr_matrix array
    
    The purpose is to maximize the log-likelihood, i.e.,
                        max log(p(y0:T|M0, Q, R, P, U))
    
    The following notations are all augmented.
    Start with y0. Because y0 = H0 x0 + v0 (linearized), x0 ~ N(x0hat, M0), v0 ~ N(0, R),we have
    y0 ~ N(H0 x0hat, H0 M0 H0^H + R).
    
    In general, yt|y0:t-1 ~ N(H_t x_{t|t-1}, H_t M_{t|t-1} H_t^H + R)
    So using conditional probability, p(y0:T) = p(y0)p(y1|y0)p(y2|y1,y2)...p(yT|y0:T-1)
    i.e., p(y0:T) = p(y0)\prod_{t = 1}^{T} p(yt|y0:t-1) = \prod_{t = 0}^{T}  N(H_t x_{t|t-1}, H_t M_{t|t-1} H_t^H + R)
    The log-likelihood is simply the log of p(y0:T)
    
    Recall for x ~ N(mu, Sigma), log(p(x)) = -d/2 log|Sigma| - 1/2 [(x-mu)^T Sigma^{-1} (x-mu)] + constant
                                           ∝ -d/2 log|Sigma| - 1/2 Trace[Sigma^{-1} (x-mu) (x-mu)^T]
    
    If does not have y0, then p(y1:T) = \prod_{t = 1}^{T}  N(H_t x_{t|t-1}, H_t M_{t|t-1} H_t^H + R)
    
    Compute dlog(|X^-1|)/dX: https://math.stackexchange.com/questions/38701/how-to-calculate-the-gradient-of-log-det-matrix-inverse
    """
    def __init__(self, ys, F, xhat0, M0, Q, R, P=None):
        # Use the smoother class to build augmented parameters to save
        smoother = KSQS(ys, F, xhat0, M0, Q, R, P)
        self.T = len(ys) # number of time steps
        self.p = smoother.p # dimension of x
        self.q = smoother.q # dimension of y
        self.ys = ys
        self.yas = [] # augmented measurements for time step in 1, 2, ... T (NOT INCLUDE TIME 0)
        for y in ys:
            ya = csc_matrix(np.hstack([y,y.conjugate()]).reshape((2*self.q,1)))
            self.yas.append(ya)
        self.x0 = smoother.xa.copy() # augmented initial expected state
        self.x0_smoothed = None
        self.M0 = smoother.Ma.copy() # augmented covariance matrix of errors of initial state
        self.M0_smoothed = None
        self.Qa = smoother.Qa.copy() # augmented covariance matrix for state noise
        self.Ra = smoother.Ra.copy() # augmented covarance matrix for measurement noise
        self.Fa = smoother.Fa.copy() # state transition matrix, TODO: make it indexed by time
        self.Ha = smoother.Ha.copy()
        
        
        self.xa_seq = []
        self.Ma_seq = []
        self.Ma_prio_seq = []
        self.xa_ns_prio_seq = [] # not smoothed prior
        self.Ma_ns_prio_seq = [] # not smoothed prior
        
    def mat_det(self, mat): # log determinant, from https://stackoverflow.com/questions/19107617/how-to-compute-scipy-sparse-matrix-determinant-without-turning-it-to-dense
        lu = splu(mat)
        diagL = lu.L.diagonal()
        diagU = lu.U.diagonal()
        logdet = np.log(diagL).sum() + np.log(diagU).sum()
        return logdet
        
    def log_likelihood(self): # see Equation (8)
        Fhem = self.Fa.getH().tocsc()
        # for mean and covariance of x0
        diff_x = self.x0_smoothed - self.x0
        x_cov = diff_x.dot(diff_x.getH().tocsc())
        mat_temp0 = self.M0_smoothed + x_cov
        mat_temp1 = spsolve(self.M0, mat_temp0)
        
        # intermediate terms for state x and observation y
        mat_A = csc_matrix((2*self.p, 2*self.p))
        mat_B = csc_matrix((2*self.p, 2*self.p))
        mat_C = csc_matrix((2*self.p, 2*self.p))
        for t in range(1, self.T+1): # real time step, not python index
            xtm1_smoothed = self.xa_seq[t-1]
            xt_smoothed = self.xa_seq[t]
            mat_A += self.Ma_seq[t-1] + xtm1_smoothed.dot(xtm1_smoothed.getH().tocsc()) # Equation (9)
            mat_B += self.Ma_prio_seq[t-1] + xt_smoothed.dot(xtm1_smoothed.getH().tocsc()) # Equation (10)
            mat_C += self.Ma_seq[t] + xt_smoothed.dot(xt_smoothed.getH().tocsc()) # Equation (11)
        
        # state x
        BFhem = mat_B.dot(Fhem)
        FAFhem = self.Fa.dot(mat_A).dot(Fhem)
        mat_temp2 = spsolve(self.Qa, mat_C - BFhem - BFhem.getH().tocsc() + FAFhem)
        
        # observation y
        mat_temp3 = csc_matrix((2*self.q, 2*self.q))
        for t in range(1, self.T+1): # real time step, not python index
            ymHx = self.yas[t-1] - sparse_col_vec_dot(self.Ha, self.xa_seq[t])
            mat_temp3 += ymHx.dot(ymHx.getH().tocsc()) +\
                        self.Ha.dot(self.Ma_seq[t]).dot(self.Ha.getH().tocsc())
        mat_temp4 = spsolve(self.Ra, mat_temp3)
        
        ll = -0.5*self.mat_det(self.M0).real - 0.5*spsolve(self.M0, mat_temp0).diagonal().sum().real -\
              0.5*self.T*self.mat_det(self.Qa).real - 0.5*mat_temp2.diagonal().sum().real - \
              0.5*self.T*self.mat_det(self.Ra).real -0.5*mat_temp4.diagonal().sum().real # Equation (8)
              
        # ll = 0 # Equation (18)
        # for t in range(1, self.T+1):
        #     try:
        #         HPHR = self.Ha.dot(self.Ma_ns_prio_seq[t-1]).dot(self.Ha.getH().tocsc()) + self.Ra
        #         HPHR_inv = spsolve(HPHR, identity(2*self.q, dtype=complex, format='csc') )
        #     except:
        #         print("Error")
        #         print(t)
        #         print(self.Ha)
        #         print(self.Ma_ns_prio_seq[t-1])
        #     ymHx = self.yas[t-1] - sparse_col_vec_dot(self.Ha, self.xa_ns_prio_seq[t-1])
        #     # ll += -0.5 * np.log(self.mat_det(HPHR)) - 0.5*spsolve(HPHR, ymHx.dot(ymHx.getH().tocsc())).diagonal().sum()
        #     ll += -0.5 * self.mat_det(HPHR).real - 0.5*(ymHx.getH().dot(HPHR_inv).dot(ymHx)).data.real[0]
        
        return ll, mat_A, mat_B, mat_C, mat_temp0, mat_temp3
        

    def learn(self, eps=1e-8):
        # intialize parameters
        smoother = KSQS(self.ys, 
                          self.Fa, 
                          self.x0, self.M0, 
                          self.Qa, 
                          self.Ra, 
                          Augmented = True)
        self.xa_seq, self.Ma_seq, self.Ma_prio_seq = smoother.smooth() # NOTE: self.xa_seq, self.Ma_seq contains elements at time 0, while the last two does not
        self.x0_smoothed = self.xa_seq[0].copy()
        self.M0_smoothed = self.Ma_seq[0].copy()
        self.xa_ns_prio_seq = smoother.xa_prio_seq[1:]
        self.Ma_ns_prio_seq = smoother.Ma_prio_seq[1:]
        
        this_ll, mat_A, mat_B, mat_C, mat_int1, mat_int2= self.log_likelihood()
        last_ll = this_ll - 0.1 - 10*eps
        
        counter = 0
        while this_ll > last_ll:
            counter += 1
            last_sol = (self.x0.copy(), self.M0.copy(), self.Qa.copy(), self.Ra.copy(), self.Fa.copy())
            # update parameters
            BAinv = spsolve(mat_A.transpose(), mat_B.transpose()).transpose()
            self.x0 = self.xa_seq[0].copy() # update E[x0]
            # self.x0 = smoother.xa_initial
            # self.M0 = mat_int1.copy() # suppose to be positive definite
            self.M0 = self.Ma_seq[0].copy()
            # self.Fa = BAinv.copy()
            # self.Qa = 1/self.T * (mat_C - BAinv.dot(mat_B.getH().tocsc()))
            ##x
            Fhem = self.Fa.getH().tocsc()
            BFhem = mat_B.dot(Fhem)
            FAFhem = self.Fa.dot(mat_A).dot(Fhem)
            self.Qa = 1/self.T * (mat_C - BFhem - BFhem.getH().tocsc() + FAFhem)
            ##
            self.Ra = 1/self.T * (mat_int2)
            
            # compute new likelihood
            smoother = KSQS(self.ys, 
                              self.Fa, 
                              self.x0, self.M0, 
                              self.Qa, 
                              self.Ra, 
                              Augmented = True)
            self.xa_seq, self.Ma_seq, self.Ma_prio_seq = smoother.smooth() # NOTE: self.xa_seq, self.Ma_seq contains elements at time 0, while the last two does not
            self.x0_smoothed = self.xa_seq[0].copy()
            self.M0_smoothed = self.Ma_seq[0].copy()
            self.xa_ns_prio_seq = smoother.xa_prio_seq[1:]
            self.Ma_ns_prio_seq = smoother.Ma_prio_seq[1:]
            
            last_ll = this_ll + 0
            this_ll, mat_A, mat_B, mat_C, mat_int1, mat_int2= self.log_likelihood()
            print("Iteration {:5d}, New log-likelihood {:.5e}, Last log-likelihood {:.5e}, Change {:.5e}".format(counter, this_ll, last_ll, this_ll - last_ll))
        
        return last_sol
        
    
if __name__ == "__main__":  
    num_dim = 4
    num_meas_dim = 2
    x = np.zeros(num_dim)
    x[0] = np.sqrt(0.5)
    x[3] = np.sqrt(0.5)
    
    # inputs
    x0 = x.copy() # E[xhat_0^+]
    Mplus = np.identity(num_dim)*1 # covariance matrix, E[(x0-xhat0^+)(x0-xhat0^+)^T]
    Q = np.identity(num_dim)*1 # Process uncertainty/noise
    R = np.identity(num_meas_dim)*1 # measurement uncertainty/noise
    
    Gate = np.array([[1,1],[1,-1]])*(1/np.sqrt(2))
    
    F = np.kron(Gate.conjugate(), Gate)
    #F = np.identity(2)
    ys = [np.array([0.5, 0.5]), np.array([0.9, 0.1]), np.array([0.5, 0.5]), np.array([0.8, 0.2])]
    # learn_obj = EMLearn(ys, F, xhatplus, Mplus, Q, R)
    # F, Q, R, x0, M0 = learn_obj.learn(np.array([0.99, 0.01]), 1e-5)
    smoother = KSQS(ys, F, x0, Mplus, Q, R)
    x_seq, M_seq, M_prio_seq = smoother.smooth()
    print(len(x_seq), len(M_seq), len(M_prio_seq))
    for i in range(len(x_seq)):
        print(i)
        # print(x_seq[i].toarray())
        # print()
        print()
        real_state = x_seq[i].toarray()[:num_dim]
        real_state = real_state/np.sqrt(np.sum(np.abs(real_state)**2))
        print(np.abs(real_state)**2)
        # print("M")
        # print(M_seq[i].toarray())
        # if i >= 1:
        #     print("M prior")
        #     print(M_prio_seq[i-1].toarray())
    print("="*100)
    
    # EM algorithm
    esp = 1e-6
    param_estor = EMLearn(ys, F, x0, Mplus, Q, R)
    estX0, estM0, estQ, estR, estF = param_estor.learn(esp)
    print("-"*100)
    # print(estX0.toarray())
    # print()
    # print(estM0.toarray())
    # print()
    # print(estQ.toarray())
    # print()
    # print(estR.toarray())
    
    print("="*100)
    realX0 = estX0.toarray()[:num_dim]
    print("New x0\n", realX0)
    realM0 = estM0.toarray()[range(num_dim),:][:,range(num_dim)]
    realF = estF.toarray()[range(num_dim),:][:,range(num_dim)]
    print("New F\n", realF)
    print("FF^*\n", realF.dot(np.matrix(realF).H))
    realQ = estQ.toarray()[range(num_dim),:][:,range(num_dim)]
    realR = estR.toarray()[range(num_meas_dim),:][:,range(num_meas_dim)]
    realP = estQ.toarray()[range(num_dim),:][:,range(num_dim, 2*num_dim)]
    realU = estR.toarray()[range(num_meas_dim),:][:,range(num_meas_dim, 2*num_meas_dim)]
    smoother = KSQS(ys, realF, realX0, realM0, realQ, realR, realP, realU)
    x_seq, M_seq, M_prio_seq = smoother.smooth()
    print(len(x_seq), len(M_seq), len(M_prio_seq))
    for i in range(len(x_seq)):
        print(i)
        # print(x_seq[i].toarray())
        print()
        real_state = x_seq[i].toarray()[:num_dim]
        real_state = real_state/np.sqrt(np.sum(np.abs(real_state)**2))
        print(np.abs(real_state)**2)
        # print("M")
        # print(M_seq[i].toarray())
        # if i >= 1:
        #     print("M prior")
        #     print(M_prio_seq[i-1].toarray())
    




        