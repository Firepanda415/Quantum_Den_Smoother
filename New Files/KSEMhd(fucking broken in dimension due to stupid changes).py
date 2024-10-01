import numpy as np
import sympy as sym
import scipy.sparse as ss
from scipy.sparse import csc_matrix, csr_matrix, vstack, hstack, diags, identity, block_diag, kron
from scipy.sparse.linalg import spsolve, splu, norm, inv
from scipy.linalg import det, eig

H_GATE = 1/np.sqrt(2)*np.matrix([[1,1],[1,-1]])
H_GATE_v = np.kron(H_GATE, H_GATE)
YB_GATE = 1/np.sqrt(2)*np.matrix([[1,1],[1j,-1j]])  # y-basis measurement gate \0><R| + |1><L|, |R> = 1/sqrt(2) * (|0>+i|1>), |L> = 1/sqrt(2) * (|0>-i|1>)
YB_GATE_v = np.kron(YB_GATE, YB_GATE)

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
            self.Ra = csc_matrix(R)
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
        
        z_basis_meas_mat = self.meas_mat(self.num_qubits)
        # H_GATE_3q = np.kron(np.kron(H_GATE, H_GATE), H_GATE) #WARNING: Only for 3-qubit example
        # x_basis_meas_mat = z_basis_meas_mat.dot(np.kron(H_GATE_3q, H_GATE_3q))
        # YB_GATE_3q = np.kron(np.kron(YB_GATE, YB_GATE), YB_GATE)
        # y_basis_meas_mat = z_basis_meas_mat.dot(np.kron(YB_GATE_3q, YB_GATE_3q))
        self.H = z_basis_meas_mat
        # self.H = vstack([z_basis_meas_mat, x_basis_meas_mat], format='csc')
        # self.H = vstack([z_basis_meas_mat, x_basis_meas_mat, y_basis_meas_mat], format='csc')
        # self.H = self.meas_mat(self.num_qubits)
        # nrows = 2**self.num_qubits
        # ncols = nrows**2
        # mat = csc_matrix((nrows, ncols), dtype=complex)
        self.Ha = hstack([self.H, 0*self.H], format='csc')
        
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
        # ya = vstack([y, y.conjugate()], format='csc')
        ya = csc_matrix(y)
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
            self.Ma_prio_seq.append(Ma_prio)
            self.Ma_seq.append(Ma_post)
        last_gain_meas = self.Ga.copy()
        last_Ha = self.Ha.copy()
        ###########
        # Initialize
        self.xa_smooth_seq.append(self.xa_seq[-1].copy()) # at the last time step, smoothed one is the posterior
        self.Ma_smooth_seq.append(self.Ma_seq[-1].copy())
        
        iden_mat = identity(2*self.p, dtype=complex, format='csc') 
        last_Mk_prio_smooth = (iden_mat - last_gain_meas.dot(self.Ha)).dot(self.Fa).dot(self.Ma_seq[T-1])
        # last_Mk_prio_smooth[last_Mk_prio_smooth < 0] = 0 # WARNING
        self.Ma_prio_smooth_seq.append(last_Mk_prio_smooth)
        
        # Backward iteration
        for t_inv in range(T, 0, -1): # the real time, not index in python
            xk_prior = self.xa_prio_seq[t_inv]
            Mk_prior = self.Ma_prio_seq[t_inv]
            Mkm1FHem_transpose = self.Fa.conjugate().dot(self.Ma_seq[t_inv-1].transpose())
            # Gkm1 = spsolve(Mk_prior.transpose(), Mkm1FHem_transpose) # new Gain from x (instead of from y, like the one in the filter)
            Gkm1 =  self.Ma_seq[t_inv-1].dot(self.Fa.conjugate().dot(ss.linalg.inv(self.Ma_prio_seq[t_inv])))
            
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
            ya = csc_matrix(y.reshape((self.q,1)))
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
        
    # def log_likelihood(self): # see Equation (8)
    #     Fhem = self.Fa.getH().tocsc()
    #     # for mean and covariance of x0
    #     diff_x = self.x0_smoothed - self.x0
    #     x_cov = diff_x.dot(diff_x.getH().tocsc())
    #     mat_temp0 = self.M0_smoothed + x_cov
    #     mat_temp1 = spsolve(self.M0, mat_temp0)
        
    #     # intermediate terms for state x and observation y
    #     mat_A = csc_matrix((2*self.p, 2*self.p))
    #     mat_B = csc_matrix((2*self.p, 2*self.p))
    #     mat_C = csc_matrix((2*self.p, 2*self.p))
    #     for t in range(1, self.T+1): # real time step, not python index
    #         xtm1_smoothed = self.xa_seq[t-1]
    #         xt_smoothed = self.xa_seq[t]
    #         mat_A += self.Ma_seq[t-1] + xtm1_smoothed.dot(xtm1_smoothed.getH().tocsc()) # Equation (9)
    #         mat_B += self.Ma_prio_seq[t-1] + xt_smoothed.dot(xtm1_smoothed.getH().tocsc()) # Equation (10)
    #         mat_C += self.Ma_seq[t] + xt_smoothed.dot(xt_smoothed.getH().tocsc()) # Equation (11)
        
    #     # state x
    #     BFhem = mat_B.dot(Fhem)
    #     FAFhem = self.Fa.dot(mat_A).dot(Fhem)
    #     mat_temp2 = spsolve(self.Qa, mat_C - BFhem - BFhem.getH().tocsc() + FAFhem)
        
    #     # observation y
    #     mat_temp3 = csc_matrix((self.q, self.q))
    #     for t in range(1, self.T+1): # real time step, not python index
    #         ymHx = self.yas[t-1] - self.Ha.dot(self.xa_seq[t])
    #         mat_temp3 += ymHx.dot(ymHx.getH().tocsc()) +\
    #                     self.Ha.dot(self.Ma_seq[t]).dot(self.Ha.getH().tocsc())
    #     mat_temp4 = spsolve(self.Ra, mat_temp3)
        
    #     ll = -0.5*self.mat_det(self.M0).real - 0.5*spsolve(self.M0, mat_temp0).diagonal().sum().real -\
    #           0.5*self.T*self.mat_det(self.Qa).real - 0.5*mat_temp2.diagonal().sum().real - \
    #           0.5*self.T*self.mat_det(self.Ra).real -0.5*mat_temp4.diagonal().sum().real # Equation (8)
              
    #     # ll = 0 # Equation (18)
    #     # for t in range(1, self.T+1):
    #     #     try:
    #     #         HPHR = self.Ha.dot(self.Ma_ns_prio_seq[t-1]).dot(self.Ha.getH().tocsc()) + self.Ra
    #     #         HPHR_inv = spsolve(HPHR, identity(2*self.q, dtype=complex, format='csc') )
    #     #     except:
    #     #         print("Error")
    #     #         print(t)
    #     #         print(self.Ha)
    #     #         print(self.Ma_ns_prio_seq[t-1])
    #     #     ymHx = self.yas[t-1] - sparse_col_vec_dot(self.Ha, self.xa_ns_prio_seq[t-1])
    #     #     # ll += -0.5 * np.log(self.mat_det(HPHR)) - 0.5*spsolve(HPHR, ymHx.dot(ymHx.getH().tocsc())).diagonal().sum()
    #     #     ll += -0.5 * self.mat_det(HPHR).real - 0.5*(ymHx.getH().dot(HPHR_inv).dot(ymHx)).data.real[0]
        
        
    #     return ll, mat_A, mat_B, mat_C, mat_temp0, mat_temp3

    # def learn(self, eps=1e-3):
    #     # intialize parameters
    #     smoother = KSQS(self.ys, 
    #                       self.Fa, 
    #                       self.x0, self.M0, 
    #                       self.Qa, 
    #                       self.Ra, 
    #                       Augmented = True)
    #     self.xa_seq, self.Ma_seq, self.Ma_prio_seq = smoother.smooth() # NOTE: self.xa_seq, self.Ma_seq contains elements at time 0, while the last two does not
    #     self.x0_smoothed = self.xa_seq[0].copy()
    #     self.M0_smoothed = self.Ma_seq[0].copy()
    #     self.xa_ns_prio_seq = smoother.xa_prio_seq[1:]
    #     self.Ma_ns_prio_seq = smoother.Ma_prio_seq[1:]
        
    #     this_ll, mat_A, mat_B, mat_C, mat_int1, mat_int2= self.log_likelihood()
    #     # update parameters
    #     BAinv = spsolve(mat_A.transpose(), mat_B.transpose()).transpose()
    #     self.x0 = self.xa_seq[0].copy() # update E[x0]
    #     self.M0 = self.Ma_seq[0].copy()
    #     # self.Fa = BAinv.copy()
    #     # self.Qa = 1/self.T * (mat_C - BAinv.dot(mat_B.getH().tocsc()))
    #     Fhem = self.Fa.getH().tocsc()
    #     BFhem = mat_B.dot(Fhem)
    #     FAFhem = self.Fa.dot(mat_A).dot(Fhem)
    #     self.Qa = 1/self.T * (mat_C - BFhem - BFhem.getH().tocsc() + FAFhem)
    #     self.Ra = 1/self.T * (mat_int2)
        
    #     last_sol = (self.x0.copy(), self.M0.copy(), self.Qa.copy(), self.Ra.copy(), self.Fa.copy())
        

        
    #     last_ll = this_ll - 10
    #     print("Iteration statrts", 'New ll {:.2f}, Last ll {:.2f}'.format(this_ll, last_ll))
        
    #     counter = 0
    #     while np.abs(this_ll - last_ll) > 1e-6:
    #         counter += 1            
    #         # compute new likelihood
    #         try:
    #             smoother = KSQS(self.ys, 
    #                               self.Fa, 
    #                               self.x0, self.M0, 
    #                               self.Qa, 
    #                               self.Ra, 
    #                               Augmented = True)
    #             self.xa_seq, self.Ma_seq, self.Ma_prio_seq = smoother.smooth() # NOTE: self.xa_seq, self.Ma_seq contains elements at time 0, while the last two does not
    #             self.x0_smoothed = self.xa_seq[0].copy()
    #             self.M0_smoothed = self.Ma_seq[0].copy()
    #             self.xa_ns_prio_seq = smoother.xa_prio_seq[1:]
    #             self.Ma_ns_prio_seq = smoother.Ma_prio_seq[1:]
    #         except:
    #             break
            
    #         last_ll = this_ll + 0
    #         this_ll, mat_A, mat_B, mat_C, mat_int1, mat_int2= self.log_likelihood()
            
    #         # if np.abs((this_ll - last_ll)/last_ll) >  10:
    #         #     print("Ends", 'New ll {:.2f}, Last ll {:.2f}'.format(this_ll, last_ll))
    #         #     break
    #         if this_ll < last_ll or this_ll > 1e10:
    #             break
            
            
    #         # update parameters
    #         BAinv = spsolve(mat_A.transpose(), mat_B.transpose()).transpose()
    #         self.x0 = self.xa_seq[0].copy() # update E[x0]
    #         # self.x0 = smoother.xa_initial
    #         # self.M0 = mat_int1.copy() # suppose to be positive definite
    #         self.M0 = self.Ma_seq[0].copy()
    #         # self.Fa = BAinv.copy()
    #         # self.Qa = 1/self.T * (mat_C - BAinv.dot(mat_B.getH().tocsc()))
    #         ##x
    #         Fhem = self.Fa.getH().tocsc()
    #         BFhem = mat_B.dot(Fhem)
    #         FAFhem = self.Fa.dot(mat_A).dot(Fhem)
    #         self.Qa = 1/self.T * (mat_C - BFhem - BFhem.getH().tocsc() + FAFhem)
    #         ##
    #         self.Ra = 1/self.T * (mat_int2)
            
    #         last_sol = (self.x0.copy(), self.M0.copy(), self.Qa.copy(), self.Ra.copy(), self.Fa.copy())
            
            
    #         print("Iteration {:5d}, New log-likelihood {:.5e}, Last log-likelihood {:.5e}, Change {:.5e}".format(counter, this_ll, last_ll, this_ll - last_ll))
        
    #     return last_sol

    def log_likelihood(self): # see Equation (8)
        Fhem = self.Fa.getH().tocsc()
        Hhem = self.Ha.transpose().tocsc()
        Qainv = inv(self.Qa).tocsc()
        Rainv = inv(self.Ra).tocsc()
        # for mean and covariance of x0
        diff_x = self.x0_smoothed - self.x0
        x_cov = diff_x.dot(diff_x.getH().tocsc())
        mat_temp0 = self.M0_smoothed + x_cov
        mat_temp1 = spsolve(self.M0, mat_temp0)
        
        # intermediate terms for state x and observation y
        mat_A = csc_matrix((2*self.p, 2*self.p))
        mat_B = csc_matrix((2*self.p, 2*self.p))
        mat_C = csc_matrix((2*self.p, 2*self.p))
        mat_D = csc_matrix((self.q, self.q)) 
        # mat_E = csc_matrix((self.q, 2*self.p)) # sum of y_t x_t^T
        for t in range(1, self.T+1): # real time step, not python index
            xtm1_smoothed = self.xa_seq[t-1]
            xt_smoothed = self.xa_seq[t]
            yt = self.yas[t-1]
            ymHxsm = yt - self.Ha.dot(xt_smoothed)
            mat_A += self.Ma_seq[t-1] + xtm1_smoothed.dot(xtm1_smoothed.getH().tocsc()) # Equation (9)
            mat_B += self.Ma_prio_seq[t-1] + xt_smoothed.dot(xtm1_smoothed.getH().tocsc()) # Equation (10)
            mat_C += self.Ma_seq[t] + xt_smoothed.dot(xt_smoothed.getH().tocsc()) # Equation (11)
            # mat_D += yt.dot(yt.transpose())
            # mat_E += yt.dot(xt_smoothed.getH().tocsc())
            mat_D += ymHxsm.dot(ymHxsm.getH().tocsc())+self.Ha.dot(self.Ma_seq[t]).dot(Hhem)
            # mat_D += ymHxsm.dot(ymHxsm.getH().tocsc())


        
        temp_mat1 = mat_C - mat_B.dot(Fhem) - self.Fa.dot(mat_B.getH().tocsc()) + self.Fa.dot(mat_A).dot(Fhem)
        # temp_mat2 = mat_D - mat_E.dot(Hhem) - self.Ha.dot(mat_E.getH().tocsc()) + self.Ha.dot(mat_C).dot(Hhem)
        
        ll =  -0.5*self.T*self.mat_det(self.Qa) - 0.5*(Qainv.dot(temp_mat1)).diagonal().sum() \
              -0.5*self.T*self.mat_det(self.Ra) - 0.5*(Rainv.dot(mat_D)).diagonal().sum() # Equation (8)
        # ll =  -0.5*self.T*self.mat_det(self.Qa) - 0.5*(spsolve(self.Qa,temp_mat1)).diagonal().sum() \
        #       -0.5*self.T*self.mat_det(self.Ra) - 0.5*(spsolve(self.Ra,mat_D)).diagonal().sum() # Equation (8)

        if ll > 1e10:
            print("WARNING: ll={:.4e}, ||Q||={:.4e}, ||R||={:.4e}, ||Q^-1A||={:.4e}, ||R^-1B||={:.4e}".format(ll, 
                                                            det(self.Qa.todense()),
                                                            det(self.Ra.todense()),
                                                            det(spsolve(self.Qa,temp_mat1).todense()),
                                                            det(spsolve(self.Ra,mat_D).todense())))
            
        
        return ll, temp_mat1, mat_D

        

    def learn(self, eps=1e-3):
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
        # compute log-likelihood
        last_ll, temp_mat1, temp_mat2= self.log_likelihood()
        # update parameters
        self.x0 = self.xa_seq[0].copy() # update E[x0]
        self.M0 = self.Ma_seq[0].copy()
        self.Qa = 1/self.T * temp_mat1
        self.Ra = 1/self.T * temp_mat2
        
        last_sol = (self.x0.copy(), self.M0.copy(), self.Qa.copy(), self.Ra.copy(), self.Fa.copy())
        this_ll, temp_mat1, temp_mat2= self.log_likelihood()
        print("Iteration statrts", 'New ll {:.2f}, Last ll {:.2f}'.format(this_ll, last_ll))
        
        counter = 0
        while np.abs((this_ll - last_ll)/last_ll) > 1e-2:
            last_ll = this_ll
            counter += 1            
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
            # compute log-likelihood
            this_ll, temp_mat1, temp_mat2= self.log_likelihood()
            # update parameters
            self.x0 = self.xa_seq[0].copy() # update E[x0]
            self.M0 = self.Ma_seq[0].copy()
            self.Qa = 1/self.T * temp_mat1
            self.Ra = 1/self.T * temp_mat2
            last_sol = (self.x0.copy(), self.M0.copy(), self.Qa.copy(), self.Ra.copy(), self.Fa.copy())
            print("Iteration {:5d}, New log-likelihood {:.5e}, Last log-likelihood {:.5e}, Change {:.5e}".format(counter, this_ll, last_ll, this_ll - last_ll))
        
        return last_sol
    
    
####################################################################################
def single_iter(n_qubits=2):
    # iterate = QuantumCircuit(n_qubits)
    # iterate.h(0)
    # iterate.cx(0,1)
    # iterate.cx(1,2)
    # iterate.ccx(0,1,2)
    # iterate.barrier()
    iterate = QuantumCircuit(n_qubits)
    iterate.h(0)
    iterate.cx(0,1)
    iterate.cx(1,2)
    iterate.barrier()
    # iterate.cx(1,2)
    # iterate.cx(0,1)
    # iterate.h(0)
    # iterate.barrier()
    return iterate



def iterative_circ(num_itrs, n_qubits=2, save_den = True, meas_basis='z'):   
    total_circ = QuantumCircuit(n_qubits)
    for i in range(num_itrs):
        total_circ.compose(single_iter(n_qubits), inplace=True)
    if meas_basis == 'x':
        for i in range(n_qubits):
            total_circ.h(i)
    if meas_basis == 'y':
        for i in range(n_qubits):
            total_circ.sdg(i)
            total_circ.h(i)
    if save_den:
        total_circ.save_density_matrix(pershot=False)
        return total_circ
        
    total_circ.measure_all()
    return total_circ

from scipy.linalg import sqrtm
def state_fid(m1,m2):
    sqm1 = sqrtm(m1)
    temp = sqm1.dot(m2).dot(sqm1)
    temp2 = sqrtm(temp)
    return np.real(np.trace(temp2))**2
####################################################################################        
    
if __name__ == "__main__":  


    # from collections import Counter
    # from qiskit import IBMQ,Aer,schedule, execute, QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
    # from qiskit.tools.visualization import plot_histogram
    # from qiskit.visualization import timeline_drawer
    # from qiskit.visualization.pulse_v2 import draw, IQXDebugging
    # from qiskit.tools.monitor import job_monitor
    # from qiskit.providers.aer.noise import NoiseModel
    # from qiskit.providers.aer import AerSimulator
    # import qiskit.quantum_info as qi
    # from qiskit.providers.aer.noise import QuantumError, ReadoutError
    
    # # Tomography functions
    # from qiskit_experiments.framework import ParallelExperiment
    # from qiskit_experiments.library import StateTomography
    
    # # Seeds
    # from numpy.random import Generator, PCG64
    # rng = Generator(PCG64(1897))
    # MY_SEEDS = rng.integers(0,high=10**8,size=100)
    
    # import matplotlib.pyplot as plt
    # from matplotlib.pyplot import figure
    
    # # plt.rcParams['text.usetex'] = True
    # fig_size = (8,6)
    # fig_dpi = 150    
    # # IBMQ.load_account()


    # # provider = IBMQ.get_provider(hub="ibm-q-pnnl", group="internal", project="default")
    # # name = "ibmq_brooklyn"
    # # backend = provider.get_backend(name)
    # # backend_noise_model = NoiseModel.from_backend(backend)
    # den_simu = AerSimulator(method='density_matrix')
    
    # n_qubits = 3
    # reps = 8
    # max_num_itrs = 10

    # unitary_simulator = Aer.get_backend('aer_simulator')
    # unitary_circ = transpile(single_iter(n_qubits), backend=den_simu)
    # unitary_circ.save_unitary()
    # unitary_result = unitary_simulator.run(unitary_circ).result()
    # unitary = unitary_result.get_unitary(unitary_circ)
    
    # unitaries = []
    # for i in range(1, max_num_itrs+1):
    #     gate = unitary.data
    #     F = np.kron(gate.conjugate(), gate)
    #     unitaries.append(F)
        
    # total_simu_dens = [] # quantum state in density-matrix form
    # total_simu_probs = [] # measurement result
    # total_simu_purs = [] # purity
    # for i in range(1, max_num_itrs+1):
    #     my_seed = MY_SEEDS[i]
    #     trans_circ = transpile(iterative_circ(i, n_qubits, save_den = True), seed_transpiler=my_seed, backend=den_simu,optimization_level=0)
    #     iter_res = den_simu.run(trans_circ,shots=8192*reps,seed_simulator=my_seed).result()
    #     iter_den = iter_res.data()['density_matrix']
    #     total_simu_dens.append(iter_den)
        
    #     trans_circ = transpile(iterative_circ(i, n_qubits, save_den = False), seed_transpiler=my_seed, backend=den_simu,optimization_level=0)
    #     iter_res = den_simu.run(trans_circ,shots=8192*reps,seed_simulator=my_seed).result()
    #     total_simu_probs.append(dictToVec(iter_res.get_counts()))
    #     total_simu_purs.append(np.real(iter_den.purity()))
        
    # total_simu_probs_x = [] # measurement result, x-basis
    # for i in range(1, max_num_itrs+1):
    #     my_seed = MY_SEEDS[i]
    #     trans_circ = transpile(iterative_circ(i, n_qubits, save_den = False, meas_basis = 'x'), seed_transpiler=my_seed, backend=den_simu,optimization_level=0)
    #     iter_res = den_simu.run(trans_circ,shots=8192*reps,seed_simulator=my_seed).result()
    #     total_simu_probs_x.append(dictToVec(iter_res.get_counts()))
        
    # observs = []
    # for i in range(len(total_simu_probs)):
    #     z_meas_p = total_simu_probs[i]
    #     x_meas_p = total_simu_probs_x[i]
    # #     y_meas_p = total_simu_probs_y[i]
    # #     observs.append(np.append(np.append(z_meas_p, x_meas_p),y_meas_p))
    #     observs.append(np.append(z_meas_p, x_meas_p))

    # initial_state = np.array([0]*((2**3)**2), dtype=complex)
    # initial_state[0] = 1
    
    # # initial state
    # num_dim = initial_state.size
    # x =initial_state
    # x[0]-= 0.01/num_dim
    # nrows = int(x.size-1)
    # for k in range(1,nrows+1):
    #     x[k] += 1/(num_dim*nrows)
        
    # # Other variance parameters
    # num_dim_state = initial_state.size
    # num_dim_obs = observs[0].size
    
    # M = np.identity(num_dim_state, dtype=complex)* 0.02 * (1) # a guess for covariance matrix, E[(x0-xhat0^+)(x0-xhat0^+)^T]
    # Q = np.identity(num_dim_state, dtype=complex)* 0.2 * (1) # state covariance
    # R = np.identity(num_dim_obs, dtype=complex)* 0.1 * (1) # meas covariance
    # P = np.identity(num_dim_state, dtype=complex)* 0.05 * (1)# 
    
    
    # total_smoother_dens = []
    # total_smoother_purs = []
    
    # # observs = total_simu_probs
    # learn_obj = EMLearn(observs, unitaries[0], x, M, Q, R, P)
    # estX0, estM0, estQ, estR, estF = learn_obj.learn() # they are all arguemented
    
    # # Slice from argumented system
    # realX0 = estX0.toarray()[:num_dim_state]
    
    # realM0 = estM0.toarray()[range(num_dim_state),:][:,range(num_dim_state)]
    # realF = estF.toarray()[range(num_dim_state),:][:,range(num_dim_state)]
    # realQ = estQ.toarray()[range(num_dim_state),:][:,range(num_dim_state)]
    # realR = estR.toarray()[range(num_dim_obs),:][:,range(num_dim_obs)]
    # realP = estQ.toarray()[range(num_dim_state),:][:,range(num_dim_state, 2*num_dim_state)]
    
    # smoother = KSQS(observs, realF, realX0, realM0, realQ, realR, realP)
    # x_seq, M_seq, M_prio_seq = smoother.smooth() 
    
    # for j in range(max_num_itrs):   
    #     x_est = np.matrix(x_seq[j+1][:num_dim_state].todense()).flatten().reshape((int(np.sqrt(num_dim_state)), int(np.sqrt(num_dim_state))), order='F')
    #     final_den = closed_den_mat(x_est)
    #     total_smoother_dens.append(final_den)
    #     total_smoother_purs.append(np.real(qi.DensityMatrix(final_den).purity()))
    
    # print(np.sum(realQ.real<0))
    
    # for den in total_smoother_dens:
    #     print("Is state a valid density matrix:", qi.DensityMatrix(den).is_valid())
        
    # # Compare fidelity, use Qiskit API (when every state from KS is a valid density matrix)
    # diff_fed_all= []
    # for i in range(max_num_itrs):
    #     qis_den_all = qi.DensityMatrix(total_smoother_dens[i])
    #     fed_difference_all =  qi.state_fidelity(total_simu_dens[i], qis_den_all)
    #     diff_fed_all.append(fed_difference_all)
    #     print("Iteration",i+1, "KS Fid:", fed_difference_all)
        
    # iter_range = range(max_num_itrs)
    # plt.plot(np.array(iter_range)+1, np.array(diff_fed_all)[iter_range], '+-', color='red', label='KS')
    # plt.xlabel("Number of Iterations")
    # plt.ylabel("Fidelity")
    # plt.xticks((np.array(iter_range)+1))
    # plt.legend()
    # plt.tight_layout()
    # plt.show()


    import numpy as np
    from collections import Counter
    from qiskit import IBMQ,Aer,schedule, execute, QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
    from qiskit.tools.visualization import plot_histogram
    from qiskit.visualization import timeline_drawer
    from qiskit.visualization.pulse_v2 import draw, IQXDebugging
    from qiskit.tools.monitor import job_monitor
    from qiskit.providers.aer.noise import NoiseModel
    from qiskit.providers.aer import AerSimulator
    import qiskit.quantum_info as qi
    from qiskit.providers.aer.noise import QuantumError, ReadoutError

    # Tomography functions
    from qiskit_experiments.framework import ParallelExperiment
    from qiskit_experiments.library import StateTomography


    import KSEMhd
    from importlib import reload  
    KSEMhd = reload(KSEMhd)
    from KSEMhd import KSQS, EMLearn

    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure

    IBMQ.load_account()
    provider = IBMQ.get_provider(hub="ibm-q-pnnl", group="internal", project="default")
    name = "ibmq_montreal"
    backend = provider.get_backend(name)
    backend_noise_model = NoiseModel.from_backend(backend)
    # # Remove readout errros
    # p0given1 = 0
    # p1given0 = 1
    # rde = ReadoutError([[1 - p1given0, p1given0], [p0given1, 1 - p0given1]])
    # for i in range(backend.configuration().num_qubits):
    #     backend_noise_model._local_readout_errors[(i,)] = rde
    den_simu = AerSimulator(method='density_matrix',noise_model=backend_noise_model)

    n_qubits = 3
    reps = 8
    max_num_itrs = 10

    from qiskit import Aer
    def single_iter(n_qubits=2):
        iterate = QuantumCircuit(n_qubits)
        iterate.h(0)
        iterate.cx(0,1)
        iterate.cx(1,2)
        iterate.barrier()
        # iterate.cx(1,2)
        # iterate.cx(0,1)
        # iterate.h(0)
        # iterate.barrier()
        return iterate

    def iterative_circ(num_itrs, n_qubits=2, save_den = True, meas_x = False):   
        total_circ = QuantumCircuit(n_qubits)
        for i in range(num_itrs):
            total_circ.compose(single_iter(n_qubits), inplace=True)
        if meas_x:
            for i in range(n_qubits):
                total_circ.h(i)
        if save_den:
            total_circ.save_density_matrix(pershot=False)
        total_circ.measure_all()
        return total_circ

    unitary_simulator = Aer.get_backend('aer_simulator')
    unitary_circ = transpile(single_iter(n_qubits), backend=den_simu)
    unitary_circ.save_unitary()
    unitary_result = unitary_simulator.run(unitary_circ).result()
    unitary = unitary_result.get_unitary(unitary_circ)

    unitaries = []
    for i in range(1, max_num_itrs+1):
        gate = unitary.data
        F = np.kron(gate.conjugate(), gate)
        unitaries.append(F)
    trans_circ = transpile(iterative_circ(3, n_qubits), backend=den_simu)
    iter_res = den_simu.run(trans_circ,shots=8192*reps,seed_simulator=np.random.randint(10**8)).result()

    np.random.seed(7)
    total_simu_dens = [] # quantum state in density-matrix form
    total_simu_probs = [] # measurement result
    total_simu_purs = [] # purity
    for i in range(1, max_num_itrs+1):
        trans_circ = transpile(iterative_circ(i, n_qubits), backend=den_simu)
        iter_res = den_simu.run(trans_circ,shots=8192*reps,seed_simulator=np.random.randint(10**8)).result()
        iter_den = iter_res.data()['density_matrix']
        total_simu_dens.append(iter_den)
        total_simu_probs.append(KSEMhd.dictToVec(iter_res.get_counts()))
        total_simu_purs.append(np.real(iter_den.purity()))

    def meas_mat(num_qubits):# H, measurement matrix for vectorized density matrix
        nrows = 2**num_qubits
        ncols = nrows**2
        mat = csc_matrix((nrows, ncols), dtype=complex)
        for k in range(nrows):
            mat[k, nrows*k+k] = 1 # take out the diagonal terms in vectorized density matrix
        return mat

    backend_noise_model_for_tomo = NoiseModel.from_backend(backend)
    # Remove readout errros
    p0given1 = 0
    p1given0 = 0
    rde = ReadoutError([[1 - p1given0, p1given0], [p0given1, 1 - p0given1]])
    for i in range(backend.configuration().num_qubits):
        backend_noise_model_for_tomo._local_readout_errors[(i,)] = rde
    den_simu_for_tomo = AerSimulator(method='density_matrix',noise_model=backend_noise_model_for_tomo)

    np.random.seed(7)

    tomo_dens = [] # quantum state in density-matrix form
    tomo_probs = [] # measurement result
    tomo_purs = [] # purity
    for i in range(1, max_num_itrs+1):
        tomo_target_circ = transpile(iterative_circ(i, n_qubits, save_den = False), backend=den_simu_for_tomo)
        qstexp = StateTomography(tomo_target_circ)
        qstdata = qstexp.run(den_simu, seed_simulation=np.random.randint(10**8)).block_for_results()
        tomo_state =  qstdata.analysis_results("state")
        
        tomo_dens.append(tomo_state.value.data)
        tomo_probs.append(tomo_state.value.probabilities())
        tomo_purs.append(np.real(tomo_state.value.purity()))

    def vecden_meas(state):# H, measurement matrix for vectorized density matrix
        num_qubits = int(np.log2(np.sqrt(state.shape[0])))
        nrows = 2**num_qubits
        ncols = nrows**2
        mat = np.zeros((nrows, ncols), dtype=np.float64)
        for k in range(nrows):
            mat[k, nrows*k+k] = 1.0 # take out the diagonal terms in vectorized density matrix
        return np.real(mat.dot(state))

    total_simu_probs_x = [] # measurement result, x-basis
    for i in range(1, max_num_itrs+1):
        trans_circ = transpile(iterative_circ(i, n_qubits, save_den = False, meas_x = True), backend=den_simu)
        iter_res = den_simu.run(trans_circ,shots=8192*reps,seed_simulator=np.random.randint(10**8)).result()
        total_simu_probs_x.append(KSEMhd.dictToVec(iter_res.get_counts()))

    observs = []
    for i in range(len(total_simu_probs)):
        z_meas_p = total_simu_probs[i]
        x_meas_p = total_simu_probs_x[i]
        observs.append(np.append(z_meas_p, x_meas_p))

    initial_state = np.array([0]*(total_simu_probs[0].size**2), dtype=complex)
    initial_state[0] = 1

    # initial state
    num_dim = initial_state.size
    x =initial_state
    x[0]-= 0.01/num_dim
    nrows = int(x.size-1)
    for k in range(1,nrows+1):
        x[k] += 1/(num_dim*nrows)

    # Other variance parameters
    num_dim_state = initial_state.size
    num_dim_obs = observs[0].size

    M = np.identity(num_dim_state, dtype=complex)* 0.01 * (1) # a guess for covariance matrix, E[(x0-xhat0^+)(x0-xhat0^+)^T]
    Q = np.identity(num_dim_state, dtype=complex)* 0.2 * (1) # state covariance
    R = np.identity(num_dim_obs)* 0.2 * (1) # meas covariance
    P = np.identity(num_dim_state, dtype=complex)* 0.1 * (1)# 
    # U = np.identity(num_dim_obs, dtype=complex)* 0.0

    total_smoother_dens = []
    total_smoother_purs = []

    # observs = total_simu_probs
    learn_obj = EMLearn(observs, unitaries[0], x, M, Q, R, P)
    estX0, estM0, estQ, estR, estF = learn_obj.learn() # they are all arguemented

    # Slice from argumented system
    realX0 = estX0.toarray()[:num_dim_state]
    realX0norm = np.sqrt(np.sum(np.abs(realX0)**2))
    realM0 = estM0.toarray()[range(num_dim_state),:][:,range(num_dim_state)]
    realF = estF.toarray()[range(num_dim_state),:][:,range(num_dim_state)]
    realQ = estQ.toarray()[range(num_dim_state),:][:,range(num_dim_state)]
    realR = estR.toarray()[range(num_dim_obs),:][:,range(num_dim_obs)]
    realP = estQ.toarray()[range(num_dim_state),:][:,range(num_dim_state, 2*num_dim_state)]

    smoother = KSQS(observs, realF, realX0, realM0, realQ, realR, realP)
    x_seq, M_seq, M_prio_seq = smoother.smooth() 

    for j in range(max_num_itrs):
        x_est = np.matrix(x_seq[j+1][:num_dim_state].todense()).flatten().reshape((int(np.sqrt(num_dim_state)), int(np.sqrt(num_dim_state))), order='F')
        x_est = (x_est+x_est.H)/2
        final_den = KSEMhd.closed_den_mat(x_est)
        total_smoother_dens.append(final_den)
        total_smoother_purs.append(np.real(qi.DensityMatrix(final_den).purity()))

    # Compare fidelity, use Qiskit API (when every state from KS is a valid density matrix)
    diff_fed_all= []
    diff_fed_tomo= []
    for i in range(max_num_itrs):
        qis_den_tomo = qi.DensityMatrix(tomo_dens[i])
        qis_den_all = qi.DensityMatrix(total_smoother_dens[i])
        fed_difference_tomo =  qi.state_fidelity(total_simu_dens[i], qis_den_tomo)
        fed_difference_all =  qi.state_fidelity(total_simu_dens[i], qis_den_all)
        diff_fed_all.append(fed_difference_all)
        diff_fed_tomo.append(fed_difference_tomo)
        print("Iteration",i+1, "KS Fid:", fed_difference_all, "Tomo Fid:", fed_difference_tomo)
        
    iter_range = range(max_num_itrs)
    plt.plot(np.array(iter_range)+1, np.array(diff_fed_all)[iter_range], '+-', color='red', label='KS')
    plt.plot(np.array(iter_range)+1, np.array(diff_fed_tomo)[iter_range], '*-', color='blue', label='Tomo')
    # plt.plot(np.array(iter_range)+1, np.array([0.9]*max_num_itrs)[iter_range], '--', color='lightgray')
    plt.xlabel("Number of Iterations")
    plt.ylabel("Fidelity")
    plt.xticks((np.array(iter_range)+1))
    plt.legend()
    plt.tight_layout()
    # plt.savefig("diff_fed_qis.svg")
    plt.show()