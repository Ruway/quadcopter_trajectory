from __future__ import division

import glog as log
from quadcopter_trajectory.poly_trajectory_generation import symbols
from scipy.linalg import block_diag
import numpy as np
import casadi as ca
from quadcopter_trajectory.poly_trajectory_generation.trajectoryTools import TrajectoryTools

class Polytrajectory(object):
    def __init__(self, N, dim):
        log.check_eq(N % 2, 0)
        # super(TrajectoryTools, self).__init__()
        self.N = N 
        self.dim = dim
        self.max_derivative_to_optimize = N / 2.0 -1
            
    def setup_from_vertice(self, vertices, time_segment, derivative_to_optimise, max_continuity, weights = np.ones(10)):
        """
        Setup polynomial trajectory generation problem using input vertice.

        :param vertice: input vertice, set of vertex node containing constraints.
        :type vertice: vertice class
        :param time_segment: Target time stamp for each vertex.
        :type time_segment: list, np.array
        :param derivative_to_optimise: Max derivative to take into account.
        :type derivative_to_optimise: int
        :param max_continuity: Derivative degree until which continuity will be ensure.
        :type max_continuity: int
        :param weights: Weight of each derivative.
        :type weights: list, np.array 

        """
        log.check(derivative_to_optimise >= symbols.POSITION and derivative_to_optimise <= self.max_derivative_to_optimize)

        self.vertices = vertices
        self.time_segment = time_segment

        self.Nv = vertices.size() # Nb of vertex
        self.Ns = self.Nv - 1 # Nb of segment

        self.derivative_to_optimize = derivative_to_optimise
        self.nb_variables = self.Ns * (self.N + 1)
        self.max_continuity = max_continuity

        self.polyCoeffSet = np.zeros((self.dim, self.N+1, self.Ns))
        self.set_weight_mask(weights)
        self.Q = self._setup_cost_matrix(derivative_to_optimise)
        self.A, self.b = self._setup_constraints_matrix(max_continuity)
        
        self.opts_settings = {'ipopt.max_iter' : 100,
                         'ipopt.print_level':0, 
                         'print_time':0, 
                         'ipopt.acceptable_tol':1e-8, 
                         'ipopt.acceptable_obj_change_tol':1e-6
                         }
        
    def solve(self,):
        """
        Solve polynomial trajectory generation problem and store coefficients for further evalutation.

        """
        x_sym = ca.SX.sym('x', self.Q[0].shape[0])

        for dd in range(self.dim):
            print("Solving {}th dimension ...".format(dd))

            obj = ca.mtimes([x_sym.T, self.Q[dd], x_sym])

            A = self.A[dd].copy()
            A_sym = ca.mtimes([A, x_sym])

            b_u = self.b[dd]
            b_l = self.b[dd]
            
            nlp_prob = {'f': obj, 'x': x_sym, 'g':A_sym}
            solver = ca.nlpsol('solver', 'ipopt', nlp_prob, self.opts_settings)

            try:
                result = solver(lbg=b_l, ubg=b_u,)
                status = solver.stats()['return_status']
                Phat_ = result['x']
                flag_ = True
            except:
                Phat_ = None
                flag_ = False

            if flag_:
                print("success !")
                self.isSolved = True
                P_ = np.dot(self.scaleMatBigInv(), Phat_)
                self.polyCoeffSet[dd] = P_.reshape(-1, self.N+1).T
            

        print("Done")


    def set_weight_mask(self, weights):
        """
        Set weight mask for derivation importance in cost matrix

        Parameters 
        ----------
        weight : 1-D Numpy array of size derivative_to_optimize + 1
 
        """
        if weights.shape[0] > self.N : 
            log.warn("To many weights, the higher terms will be ignored.")
            self.weight_mask = weights[:self.derivative_to_optimize]
        else:
            self.weight_mask = weights
        
    
    def eval(self, t_fine, derivative):
        """
        Evaluate one trajectory derivative using a fine temporal vector. 

        Parameters 
        ----------
        t_fine : 1-D Numpy array time time points 
        derivative : Trajectory derivative to evaluate

        Return
        ------
        val : Dim x size_t_fine Numpy array with trajectory evaluation
 
        """
        time_stamp = [sum(self.time_segment[:i]) for i in range(len(self.time_segment) + 1)]

        val = np.zeros((self.dim, t_fine.shape[0]))
       
        for dd in range(self.dim):
            for idx in range(t_fine.shape[0]):
                t_i = t_fine[idx]
                
                if t_i < time_stamp[0] or t_i > time_stamp[-1]:
                    log.warn("WARNING: Eval of t: out of bound. Extrapolation.")

                m = self._find_pts_seg(t_i)
                val[dd, idx] = np.dot(self._build_time_vect(t_i - time_stamp[m], derivative).T, self.polyCoeffSet[dd, :, m])
        
        return val

    def eval_one(self, t_i, derivative):
        """
        Evaluate one trajectory derivative using a fine temporal vector. 

        Parameters 
        ----------
        t_fine : 1-D Numpy array time time points 
        derivative : Trajectory derivative to evaluate

        Return
        ------
        val : Dim x size_t_fine Numpy array with trajectory evaluation
 
        """
        time_stamp = [sum(self.time_segment[:i]) for i in range(len(self.time_segment) + 1)]
        
        val = np.zeros((self.dim, 1))
        for dd in range(self.dim):

            if t_i < time_stamp[0] or t_i > time_stamp[-1]:
                log.warn("WARNING: Eval of t: out of bound. Extrapolation.")

            m = self._find_pts_seg(t_i)
            val[dd, 0] = np.dot(self._build_time_vect(t_i - time_stamp[m], derivative).T, self.polyCoeffSet[dd, :, m])
        
        return val


    def _setup_constraints_matrix(self, max_continuity):
        """
        Set constraints based on A * q = b. Taking into account the fixed constraints and the continuity constraints.

        Parameters 
        ----------
        max_continuity : Max continuity dimension. 

        Return
        ------
        A matrix : Dim x (nb_constraints * (nb_seg - 1) * (max_continuity + 1)) x (nb_segment * (poly_deg + 1))
        b vector : (nb_constraints * (nb_seg - 1) * (max_continuity + 1))
 
        """
        
        A = None
        b = None
    
        for v in range(self.Nv):
            seg, tau =  self._find_v_seg(v)
            av = self._get_av_constraints(self.vertices[v], seg, tau)
            A = av if A is None else np.concatenate((A, av), axis=1)

            bv = self._get_bv(self.vertices[v])
     
            b = bv.reshape(self.dim, -1, 1) if b is None else np.concatenate((b, bv.reshape(self.dim, bv.shape[1], -1)), axis=1)

            if v < self.Ns - 1:
                av_eq = self._get_av_continuity(seg, max_continuity )
                A = np.concatenate((A, av_eq.reshape(1, -1, self.nb_variables).repeat(self.dim, axis=0)), axis=1)
                bv_eq = np.zeros((max_continuity + 1, 1)) 
                b = np.concatenate((b, bv_eq.reshape(1, -1, 1).repeat(self.dim, axis=0)), axis=1)
        
        return A, b

    def _setup_cost_matrix(self, derivative_to_optimise):
        """
        Set cost matrix Q 

        Parameters 
        ----------
        derivative_to_optimise : Derive number for the optimization process 

        Return
        ------
        Q matrix : Dim x (nb_seg * (poly_deg + 1)) x (nb_seg * (poly_deg + 1))
 
        """
        Q_set = np.zeros((self.dim, self.nb_variables, self.nb_variables))
        
        for dd in range(self.dim):
            Q_dim = np.zeros((self.nb_variables, self.nb_variables))

            for derivative in range(self.derivative_to_optimize+1):
                Q_derive = None

                for s in range(self.Ns):
                    Q_s = self._build_Q_sub(derivative + 1) / self.time_segment[s]**(2*(derivative + 1) - 1)
                    Q_derive = Q_s.copy() if Q_derive is None else block_diag(Q_derive, Q_s)

                Q_dim = self.weight_mask[derivative] * Q_derive
                
            Q_set[dd] = Q_dim
        
        return Q_set

    def _build_Q_sub(self, derivative_to_optimise):
        """
        Set sub-cost matrix Q for one segment 

        Parameters 
        ----------
        derivative_to_optimise : Target derivative for Q sub-matrix generation 

        Return
        ------
        Q matrix : (poly_deg + 1)) x (poly_deg + 1)
 
        """
        Q = np.zeros((self.N + 1, self.N+1))

        if derivative_to_optimise > self.N :
            log.warn("Order of derivative > poly order, return zeros-matrix.")
            return Q

        for i in range(derivative_to_optimise, self.N+1):
            for j in range(derivative_to_optimise, self.N+1):
                Q[i,j] = 2 * self._get_nth_coeff(i, derivative_to_optimise) * self._get_nth_coeff(j, derivative_to_optimise) \
                             / (i + j - 2 * derivative_to_optimise + 1)

        return Q

    def _get_bv(self, vertex):
        """
        Build b vector for optimization constraints A * x = b 

        Parameters 
        ----------
        vertex : Vertex containing constraints to translate to the b vector 

        Return
        ------
        b : Dim x nb_constraints
 
        """
        const_number = vertex.get_constraints_number()

        if(const_number > self.max_derivative_to_optimize + 1):
            log.warn("Too many constraints, the higher ones will be ignored.")

        bv = np.zeros((self.dim, const_number))

        for c in range(const_number):
            const = vertex.get_constraint(c)
            for d in range(self.dim):
                bv[d, c] = const[d]

        return bv

    def _get_av_constraints(self, vertex, seg, tau):
        """
        Build constraint part of A matrix for optimization constraints A * x = b 

        Parameters 
        ----------
        vertex : Vertex containing constraints to translate to the b vector 
        seg : Target segment
        tau : ?

        Return
        ------
        av : Dim x nb_constraints x (polynome_coeff + 1) * segment
 
        """
        const_number = vertex.get_constraints_number()
        av = np.zeros((self.dim, const_number, self.nb_variables))

        idxStart = seg * (self.N + 1)
        idxEnd = (seg + 1) * (self.N + 1)

        dt = self.time_segment[seg]

        for c in range(const_number):
            for dd in range(self.dim):
                av[dd, c, idxStart:idxEnd] = self._build_time_vect(tau, c).flatten() / dt**c

        return av

    def _get_av_continuity(self, seg, continuity):
        """
        Build continuity part of A matrix for optimization constraints A * x = b 

        Parameters 
        ----------
        continuity : Derivative to ensure continuity 
        seg : Target segment

        Return
        ------
        av_eq : continuity x (polynome_coeff + 1) * segment
 
        """
        av_eq = np.zeros((continuity + 1, self.nb_variables))

        idxStart = seg * (self.N + 1)
        idxStop = (seg + 2) * (self.N + 1)

        dt1 = self.time_segment[seg]
        dt2 = self.time_segment[seg + 1]

        for d in range(continuity + 1):
            av_eq[d, idxStart:idxStop] = np.concatenate((self._build_time_vect(1, d)/dt1**d, - self._build_time_vect(0, d)/dt2**d), axis=0).flatten()
        
        return av_eq

    def _build_time_vect(self, t, derivative):
        vect = np.zeros((self.N+1, 1))

        for i in range(derivative, self.N+1):
            vect[i] = self._get_nth_coeff(i, derivative) * t**(i - derivative)

        return vect

    def _get_nth_coeff(self, p_coeff, d):
        if d == 0:
            coeff = 1
        else:
            accum_prod = np.cumprod(np.arange(p_coeff, p_coeff - d, -1))
            coeff = accum_prod[-1] * ( p_coeff>=d )

        return coeff

    def _find_v_seg(self, vertex_id):
        return vertex_id if vertex_id < self.Ns else self.Ns - 1, 0 if vertex_id < self.Ns else 1

    def _find_pts_seg(self, pts):
        time_stamp = [sum(self.time_segment[:i]) for i in range(len(self.time_segment) + 1)]
        idx_ = np.where(time_stamp<=pts)[0]

        if idx_.shape[0]>0:
            seg = np.max(idx_)
            if seg >= self.Ns:
                if pts != time_stamp[-1]:
                    log.warn('Eval of t : geq TM. eval target = last segment')
                seg = self.Ns-1
        else:
            log.warn('Eval of t : leq T0. eval target = 1st segment')
            seg = 0
        return seg

    '''
    Math function to reshape matrix for optimisation
    '''   
    def scaleMat(self, delT):
        mat_ = np.diag([delT**i for i in range(self.N+1)])
        return mat_

    def scaleMatBigInv(self,):
        mat_ = None
        for seg in range(self.Ns):
            matSet_ = self.scaleMat(1/(self.time_segment[seg]))
            if mat_ is None:
                mat_ = matSet_.copy()
            else:
                mat_ = block_diag(mat_, matSet_)
        return mat_
    '''
    End
    '''











